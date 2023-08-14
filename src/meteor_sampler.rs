use std::cmp::min;
use std::fmt::{Debug, Formatter};
use std::sync::Mutex;

use aes_gcm::aead::rand_core::Error;
use anyhow::Context;
use bitvec::field::BitField;
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use bitvec::view::BitView;
use llm::{Sampler, TokenBias, TokenId, Tokenizer};
use log::{debug, info};
use partial_sort::PartialSort;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::StdRng;
use rand::{Rng, RngCore};

use crate::token_trie::{TokenTrie, Trie};
use crate::util::{cumsum_rescale, prefix_bits, Cumsum, Rescale};

pub(crate) struct MeteorDecodeSampler {
    sampler_config: TopKTopPConfig,
    /// the token trie to sample from
    trie: TokenTrie,
    /// the byte array to embed (e.g. a ciphertext of a hidden message)
    context: Vec<TokenId>,
    /// the stego text to decode
    stego_text: Vec<u8>,
    /// a tokenizer
    tokenizer: Tokenizer,
    pub recovered_bits: BitVec<u8>,
    special_token_ids: Vec<TokenId>,
}

impl Debug for MeteorDecodeSampler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeteorDecodeSampler")
    }
}

impl MeteorDecodeSampler {
    pub fn new(
        trie: TokenTrie,
        tokenizer: &Tokenizer,
        context: Vec<TokenId>,
        stego_text: Vec<u8>,
        special_token_ids: Vec<TokenId>,
    ) -> MeteorDecodeSampler {
        MeteorDecodeSampler {
            sampler_config: TopKTopPConfig::default(),
            trie,
            context,
            stego_text,
            tokenizer: match tokenizer {
                Tokenizer::Embedded(t) => Tokenizer::Embedded(t.clone()),
                Tokenizer::HuggingFace(t) => Tokenizer::HuggingFace(t.clone()),
            },
            recovered_bits: BitVec::new(),
            special_token_ids,
        }
    }
}

#[derive(Debug)]
struct TopKTopPConfig {
    /// The top K words by score are kept during sampling.
    pub top_k: usize,
    /// The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature (randomness) used for sampling. A higher number is more random.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// The number of tokens to consider for the repetition penalty.
    pub repetition_penalty_last_n: usize,
}

impl Default for TopKTopPConfig {
    fn default() -> Self {
        TopKTopPConfig {
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::empty(),
            repetition_penalty_last_n: 512,
        }
    }
}

#[derive(Debug)]
/// A sampler based on TopKTopP. Instead of an rng, this will use hidden message ciphertexts to select a msg
pub(crate) struct MeteorEncodeSampler {
    sampler_config: TopKTopPConfig,
    /// the token trie to sample from
    trie: TokenTrie,
    /// the byte array to embed (e.g. a ciphertext of a hidden message)
    ciphertext: BitVecSampler,
}

impl MeteorEncodeSampler {
    pub fn new(trie: TokenTrie, msg: Vec<u8>, key_rng: StdRng, pad_rng: StdRng) -> Self {
        MeteorEncodeSampler {
            sampler_config: TopKTopPConfig::default(),
            trie,
            ciphertext: BitVecSampler::new(msg, key_rng, pad_rng),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SamplerContainer<T> {
    pub(crate) inner: Mutex<T>,
}

impl<T: Default> Default for SamplerContainer<T> {
    fn default() -> Self {
        SamplerContainer {
            inner: Mutex::new(Default::default()),
        }
    }
}

impl<S> SamplerContainer<S> {
    pub fn new(sampler: S) -> Self {
        SamplerContainer {
            inner: Mutex::new(sampler),
        }
    }
}

impl<S: MutableSampler + Debug + Send + Sync> Sampler for SamplerContainer<S> {
    fn sample(
        &self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> TokenId {
        let mut sampler = self.inner.lock().unwrap();
        sampler.sample(previous_tokens, logits, rng)
    }
}

//region Preprocess Logits (Top-P Top-K)
struct PreprocessedLogits {
    token_ids: Vec<TokenId>,
    probs: Vec<f32>,
}
fn preprocess_logits<'a>(
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    temperature: f32,
    repetition_penalty_last_n: usize,
    bias_tokens: &TokenBias,
    logits: &'a [f32],
    previous_tokens: &[TokenId],
) -> PreprocessedLogits {
    let n_logits = logits.len();
    let mut logits_id = Vec::<(f32, TokenId)>::with_capacity(n_logits);

    // TODO: consider if this can be modularized and this sampler can be composed out of multiple pieces,
    // instead of having this monolithic function that embeds the repetition penalty and token bias
    {
        let scale = 1.0 / temperature;
        for (i, &logit) in logits.iter().enumerate() {
            let tid = i as TokenId;

            let val = if let Some(logit_override) = bias_tokens.get(tid) {
                logit_override
            } else if previous_tokens[previous_tokens
                .len()
                .saturating_sub(repetition_penalty_last_n)..]
                .contains(&(i as TokenId))
            {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if logits[i] < 0.0 {
                    logit * scale * repeat_penalty
                } else {
                    logit * scale / repeat_penalty
                }
            } else {
                logit * scale
            };
            logits_id.push((val, tid));
        }
    }

    // find the top K tokens
    {
        logits_id.partial_sort(top_k, |a, b| {
            // Sort descending
            b.0.total_cmp(&a.0)
        });
        logits_id.truncate(top_k);
    }

    let token_weights = logits_id.iter().map(|x| x.0);
    let mut token_ids: Vec<TokenId> = logits_id.iter().map(|x| x.1).collect();

    let maxl = token_weights.max_by(f32::total_cmp).unwrap();

    // compute probs for the top K tokens
    let mut probs: Vec<f32> = logits_id
        .iter()
        .copied()
        .map(|(k, _)| (k - maxl).exp())
        .collect();
    let sum: f32 = probs.iter().copied().sum();

    // Normalize the probs
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top p sampling
    if top_p < 1.0 {
        let mut cumsum = 0.0;
        for i in 0..probs.len() {
            cumsum += probs[i];
            if cumsum >= top_p {
                probs.truncate(i + 1);
                token_ids.truncate(i + 1);
                logits_id.truncate(i + 1);
                break;
            }
        }

        cumsum = 1.0 / cumsum;
        for p in probs.iter_mut() {
            *p *= cumsum;
        }
    }
    PreprocessedLogits { token_ids, probs }
}
//endregion

trait MutableSampler {
    fn sample(
        &mut self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> TokenId;
}

impl MutableSampler for MeteorEncodeSampler {
    fn sample(
        &mut self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> TokenId {
        let bits_remaining = self.ciphertext.bits_remaining();
        if bits_remaining == 0 {
            info!("Ciphertext embedded, returning EOT token");
            return 2; // EOT token
        }
        let TopKTopPConfig {
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            ..
        } = self.sampler_config;
        let bias_tokens = &self.sampler_config.bias_tokens;

        let PreprocessedLogits { token_ids, probs } = preprocess_logits(
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            bias_tokens,
            logits,
            previous_tokens,
        );
        assert_eq!(token_ids.len(), probs.len());

        let Rescale { probs, cumsum: _ } = cumsum_rescale(probs);

        let update_vec: Vec<(TokenId, u64)> = token_ids
            .iter()
            .zip(&probs)
            .map(|(x, y)| (*x, *y))
            .collect();
        self.trie.update(&update_vec);
        //debug!("{:?}", self.trie);

        let dist = self.trie.distribution();
        //let weighted = dist.weighted_index().expect("WeightedIndex error");
        //let idx = weighted.sample(&mut self.ciphertext);
        let cumsum = dist.probs.cumsum(0);
        let coins: u64 = self.ciphertext.gen();
        let idx = cumsum
            .iter()
            .enumerate()
            .find(|(_, &e)| e > coins)
            .map(|(idx, _)| idx)
            .expect("Sampling failed");
        let tokens = &dist.tokens[idx];

        let lower = if idx > 0 { cumsum[idx - 1] } else { 0 };
        let upper = cumsum[idx];
        let bits_encoded = prefix_bits(lower, upper);
        let num_bits_encoded = bits_encoded.len();
        if num_bits_encoded > 0 {
            info!("Encoded bits {}", bits_encoded);
        }
        self.ciphertext.offset -= 64 - num_bits_encoded;
        info!(
            "{} bits encoded, {} bits remaining, offset {}",
            num_bits_encoded,
            bits_remaining as i32 - num_bits_encoded as i32,
            self.ciphertext.offset,
        );

        return if tokens.len() == 1 {
            debug!("Simple distr: {}", tokens[0]);
            tokens[0]
        } else {
            // resample
            let subtrie = self.trie.lookup(&dist.reprs[idx]).expect("Lookup failed");
            let st_tokens = subtrie.tokens();
            let probs = subtrie.probabilities();
            assert_eq!(st_tokens.len(), probs.len());
            let weighted = WeightedIndex::new(probs).expect("WeightedIndex error");
            let st_idx = weighted.sample(rng);
            assert_eq!(&st_tokens, tokens);
            let token = st_tokens[st_idx];
            debug!("Resampled {} from subtrie {}", token, &dist.reprs[idx]);
            token
        };
    }
}

impl MutableSampler for MeteorDecodeSampler {
    fn sample(
        &mut self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> TokenId {
        if self.context.eq(previous_tokens) {
            info!("starting new stego decoder session");
            self.recovered_bits.clear();
        }
        if self.stego_text.is_empty() {
            info!("Decoding complete: EOT");
            return 2;
        }
        let TopKTopPConfig {
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            ..
        } = self.sampler_config;
        let bias_tokens = &self.sampler_config.bias_tokens;

        let PreprocessedLogits { token_ids, probs } = preprocess_logits(
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            bias_tokens,
            logits,
            previous_tokens,
        );
        assert_eq!(token_ids.len(), probs.len());

        let Rescale { probs, cumsum: _ } = cumsum_rescale(probs);

        let update_vec: Vec<(TokenId, u64)> = token_ids
            .iter()
            .zip(&probs)
            .map(|(x, y)| (*x, *y))
            .collect();
        self.trie.update(&update_vec);

        //debug!("{:?}", self.trie);

        let dist = self.trie.distribution();
        let get_token = |idx| self.tokenizer.token(idx);
        let idx = dist
            .find_token_prefix_index(get_token, &self.stego_text, &self.special_token_ids)
            .with_context(|| {
                format!(
                    "Token ID lookup failed for {:?} in distribution {:?} with reprs {:?}",
                    self.stego_text[0],
                    dist,
                    dist.reprs
                        .iter()
                        .map(|r| get_token(*r as usize))
                        .collect::<Vec<Vec<u8>>>(),
                )
            })
            .unwrap();
        let cumsum = dist.probs.cumsum(0);
        let lower = if idx > 0 { cumsum[idx - 1] } else { 0 };
        let upper = cumsum[idx];
        let bits_encoded = prefix_bits(lower, upper);
        if !bits_encoded.is_empty() {
            info!("Recovered bits {}", bits_encoded)
        }
        self.recovered_bits.extend(&bits_encoded);
        let tokens = &dist.tokens[idx];

        let token = if tokens.len() == 1 {
            debug!("Simple distr: {}", tokens[0]);
            tokens[0]
        } else {
            // resample
            let subtrie = self.trie.lookup(&dist.reprs[idx]).expect("Lookup failed");
            let st_tokens = subtrie.tokens();
            let probs = subtrie.probabilities();
            assert_eq!(st_tokens.len(), probs.len());
            let weighted = WeightedIndex::new(probs).expect("WeightedIndex error");
            let st_idx = weighted.sample(rng);
            assert_eq!(&st_tokens, tokens);
            let token = st_tokens[st_idx];
            debug!("Resampled {} from subtrie {}", token, &dist.reprs[idx]);
            token
        };
        let token_bytes = self.tokenizer.decode(vec![token], false);
        self.stego_text
            .drain(..min(self.stego_text.len(), token_bytes.len()));

        info!(
            "{} bits decoded ({} bits total, {} bytes of stego text remaining)",
            &bits_encoded.len(),
            self.recovered_bits.len(),
            self.stego_text.len()
        );
        token
    }
}

//region BitVecSampler
/// An RNG that uses a pre-initialized BitVector as source of randomness
#[derive(Debug, Default)]
struct BitVecSampler<R1: Rng = StdRng, R2: Rng = StdRng> {
    source: BitVec<u8, Msb0>,
    key_rng: R1,
    pad_rng: R2,
    pub offset: usize,
}

impl<R1: Rng, R2: Rng> BitVecSampler<R1, R2> {
    fn new(source: Vec<u8>, key_rng: R1, pad_rng: R2) -> Self {
        BitVecSampler {
            source: BitVec::<u8, Msb0>::from_vec(source.into_iter().collect::<Vec<u8>>()),
            key_rng,
            pad_rng,
            offset: 0,
        }
    }

    fn bits_remaining(&self) -> usize {
        self.source.len() - min(self.offset, self.source.len())
    }
}

impl<R1: Rng, R2: Rng> RngCore for BitVecSampler<R1, R2> {
    fn next_u32(&mut self) -> u32 {
        panic!("use next_u64");
    }

    fn next_u64(&mut self) -> u64 {
        let source_len = self.source[self.offset..].len();
        let source = if source_len < 64 {
            let mut source = self.source[self.offset..].to_bitvec();
            let pad: BitVec<u8> = unsafe {
                self.pad_rng.next_u64().view_bits().align_to::<u8>().1[..64 - source_len]
                    .to_bitvec()
            };
            //debug!("pad source \n{} with \n{}", source, pad);
            source.extend(pad);
            //debug!("padded source: \n{}", source);
            source
        } else {
            self.source[self.offset..self.offset + 64].to_bitvec()
        };
        let next: u64 = source.load_be();
        let key: u64 = self.key_rng.gen();
        // TODO encrypt
        // let next = next ^ key;
        //debug!("sampled {:02X}", next);
        self.offset += 64;
        next
    }

    fn fill_bytes(&mut self, _: &mut [u8]) {
        panic!("use next_u64");
    }

    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Error> {
        panic!("use next_u64");
    }
}
//endregion

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use crate::meteor_sampler::BitVecSampler;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_bitvec_sampler() -> anyhow::Result<()> {
        init();
        let vec: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let key_rng = StdRng::seed_from_u64(0x00C0FFEE);
        let pad_rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut sampler = BitVecSampler::new(vec.clone(), key_rng, pad_rng);
        let next = sampler.next_u64();
        let expect = (vec[0] as u64) << 8 * 7
            | (vec[1] as u64) << 8 * 6
            | (vec[2] as u64) << 8 * 5
            | (vec[3] as u64) << 8 * 4
            | (vec[4] as u64) << 8 * 3
            | (vec[5] as u64) << 8 * 2
            | (vec[6] as u64) << 8 * 1
            | (vec[7] as u64);
        assert_eq!(next, expect);
        Ok(())
    }
}
