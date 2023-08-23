use std::cmp::min;
use std::fmt::{Debug, Formatter};

use anyhow::Context;
use bitvec::field::BitField;
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use bitvec::view::BitView;
use llm::{TokenId, Tokenizer};
use llm_samplers::prelude::*;
use log::{debug, info};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::StdRng;
use rand::{Error, Rng, RngCore};

use crate::token_trie::{TokenTrie, Trie};
use crate::util::{cumsum_rescale, prefix_bits, Cumsum, Rescale};

fn make_default_chain() -> SamplerChain {
    SamplerChain::<u32, f32>::new()
        // apply penalty for token repetition
        + SampleRepetition::default().penalty(1.30).last_n(64)
        // apply penalty for the last 64 tokens
        + SampleFreqPresence::default().last_n(64)
        // penalize repeating sequence of tokens
        + SampleSeqRepetition::default()
        // cut-off top 40 tokens
        + SampleTopK::default().k(40)
        // apply Tail-free sampling (https://www.trentonbricken.com/Tail-Free-Sampling/)
        + SampleTailFree::default()
        // maximize "human-like" output
        + SampleLocallyTypical::default()
        // cut-off top 95 % tokens
        + SampleTopP::default().p(0.95)
        // apply temperature
        + SampleTemperature::default().temperature(0.8)
}

pub(crate) struct MeteorDecodeSampler {
    chain: SamplerChain,
    /// the last sampled token ID
    token_id: Option<TokenId>,
    /// the token trie to sample from
    trie: TokenTrie,
    /// the byte array to embed (e.g. a ciphertext of a hidden message)
    context: Vec<TokenId>,
    /// the stego text to decode
    stego_text: Vec<u8>,
    /// the RNG used for decrypting the embedded message
    cipher_rng: StdRng,
    /// a tokenizer
    tokenizer: Tokenizer,
    pub recovered_bits: BitVec<u8, Msb0>,
    special_token_ids: Vec<TokenId>,
    /// the model's EOT token
    eot_token_id: TokenId,
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
        cipher_rng: StdRng,
        stego_text: Vec<u8>,
        special_token_ids: Vec<TokenId>,
        eot_token_id: TokenId,
    ) -> Self {
        let chain = make_default_chain();
        MeteorDecodeSampler {
            chain,
            token_id: None,
            trie,
            context,
            cipher_rng,
            stego_text,
            tokenizer: match tokenizer {
                Tokenizer::Embedded(t) => Tokenizer::Embedded(t.clone()),
                Tokenizer::HuggingFace(t) => Tokenizer::HuggingFace(t.clone()),
            },
            recovered_bits: BitVec::new(),
            special_token_ids,
            eot_token_id,
        }
    }
}

#[derive(Debug)]
pub struct MeteorEncodeStats {
    pub num_tokens: usize,
    pub num_bits_encoded: usize,
}

impl Default for MeteorEncodeStats {
    fn default() -> Self {
        MeteorEncodeStats {
            num_tokens: 0,
            num_bits_encoded: 0,
        }
    }
}

#[derive(Debug)]
/// A sampler based on TopKTopP. Instead of an rng, this will use hidden message ciphertexts to select a msg
pub(crate) struct MeteorEncodeSampler {
    chain: SamplerChain,
    /// last sampled token ID
    token_id: Option<TokenId>,
    /// the token trie to sample from
    trie: TokenTrie,
    /// the byte array to embed (e.g. a ciphertext of a hidden message)
    ciphertext: BitVecSampler,
    /// the model's EOT token
    eot_token_id: TokenId,
    /// stats
    pub stats: MeteorEncodeStats,
}

impl MeteorEncodeSampler {
    pub fn new(
        trie: TokenTrie,
        msg: Vec<u8>,
        key_rng: StdRng,
        pad_rng: StdRng,
        eot_token_id: TokenId,
    ) -> Self {
        let chain = make_default_chain();
        MeteorEncodeSampler {
            chain,
            token_id: None,
            trie,
            ciphertext: BitVecSampler::new(msg, key_rng, pad_rng),
            eot_token_id,
            stats: Default::default(),
        }
    }
}

impl Sampler<u32, f32> for MeteorEncodeSampler {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = u32>,
        logits: &'a mut Logits<u32, f32>,
    ) -> anyhow::Result<&'a mut Logits<u32, f32>> {
        let bits_remaining = self.ciphertext.bits_remaining();
        if bits_remaining == 0 {
            info!("Ciphertext embedded, returning EOT token");
            self.token_id = Some(self.eot_token_id);
            return Ok(logits);
        }
        let logits = self.chain.sample(res, logits)?;
        logits.softmax()?;
        let token_ids: Vec<u32> = logits.iter().map(|l| l.token_id).collect();
        let probs: Vec<f32> = logits.iter().map(|l| l.prob).collect();
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

        self.stats.num_tokens += 1;
        self.stats.num_bits_encoded += num_bits_encoded;

        self.token_id = if tokens.len() == 1 {
            debug!("Simple distr: {}", tokens[0]);
            Some(tokens[0])
        } else {
            // resample
            let subtrie = self.trie.lookup(&dist.reprs[idx]).expect("Lookup failed");
            let st_tokens = subtrie.tokens();
            let probs = subtrie.probabilities();
            assert_eq!(st_tokens.len(), probs.len());
            let weighted = WeightedIndex::new(probs).expect("WeightedIndex error");
            let mut st_idx = None;
            res.with_rng_mut(&mut |rng: &mut dyn RngCore| {
                st_idx = Some(weighted.sample(rng));
            })?;
            assert_eq!(&st_tokens, tokens);
            let token = st_tokens[st_idx.expect("sampling failed")];
            debug!("Resampled {} from subtrie {}", token, &dist.reprs[idx]);
            Some(token)
        };
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<u32> {
        self.token_id
    }
}

impl Sampler<u32, f32> for MeteorDecodeSampler {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = u32>,
        logits: &'a mut Logits<u32, f32>,
    ) -> anyhow::Result<&'a mut Logits<u32, f32>> {
        res.with_last_tokens(&mut |previous_tokens| {
            if self.context.eq(previous_tokens) {
                info!("starting new stego decoder session");
                self.recovered_bits.clear();
            }
        })?;
        if self.stego_text.is_empty() {
            info!("Decoding complete: EOT");
            self.token_id = Some(self.eot_token_id);
            return Ok(logits);
        }
        let logits = self.chain.sample(res, logits)?;
        logits.softmax()?;
        let token_ids: Vec<u32> = logits.iter().map(|l| l.token_id).collect();
        let probs: Vec<f32> = logits.iter().map(|l| l.prob).collect();
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
        let mut bits_encoded = prefix_bits(lower, upper);
        let prefix_len = bits_encoded.len();
        let key = self.cipher_rng.next_u64();
        let key = key.view_bits::<Msb0>().to_bitvec();
        bits_encoded.extend(BitVec::<u64, Msb0>::repeat(false, 64 - prefix_len));
        bits_encoded ^= key;
        bits_encoded = bits_encoded[..prefix_len].to_bitvec();
        if prefix_len > 0 {
            info!("Recovered bits {}", bits_encoded);
            self.recovered_bits.extend(&bits_encoded);
        }
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
            let mut st_idx = None;
            res.with_rng_mut(&mut |rng| {
                st_idx = Some(weighted.sample(rng));
            })?;
            assert_eq!(&st_tokens, tokens);
            let token = st_tokens[st_idx.expect("sample failed")];
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
        self.token_id = Some(token);
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<u32> {
        self.token_id
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
        let next = next ^ key;
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
    use rand::RngCore;

    use crate::meteor_sampler::BitVecSampler;

    struct MockSampler;

    impl RngCore for MockSampler {
        fn next_u32(&mut self) -> u32 {
            unimplemented!()
        }

        fn next_u64(&mut self) -> u64 {
            0
        }

        fn fill_bytes(&mut self, _: &mut [u8]) {
            unimplemented!()
        }

        fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), rand::Error> {
            unimplemented!()
        }
    }

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_bitvec_sampler() -> anyhow::Result<()> {
        init();
        let vec: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let key_rng = MockSampler;
        let pad_rng = MockSampler;
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
