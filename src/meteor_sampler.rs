use crate::token_trie::{Prob, TokenTrie, Trie};
use aes_gcm::aead::rand_core::{impls, Error};
use bitvec::field::BitField;
use bitvec::vec::BitVec;
use bitvec::view::BitView;
use llm::{Sampler, TokenBias, TokenId};
use log::{debug, info};
use partial_sort::PartialSort;
use rand::distributions::{Distribution, WeightedIndex};
use rand::RngCore;
use std::sync::Mutex;

#[derive(Debug)]
/// A sampler based on TopKTopP. Instead of an rng, this will use hidden message ciphertexts to select a msg
struct MeteorSampler {
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
    /// the token trie to sample from
    trie: TokenTrie,
    /// the byte array to embed (e.g. a ciphertext of a hidden message)
    ciphertext: BitVecSampler,
}

impl Default for MeteorSampler {
    fn default() -> Self {
        Self {
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::empty(),
            repetition_penalty_last_n: 512,
            trie: Default::default(),
            ciphertext: Default::default(),
        }
    }
}

fn common_bit_prefix(x: f32, y: f32) -> BitVec {
    todo!()
}

impl MeteorSampler {
    fn sample(
        &mut self,
        previous_tokens: &[TokenId],
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> TokenId {
        let Self {
            top_k,
            top_p,
            repeat_penalty,
            temperature,
            repetition_penalty_last_n,
            ..
        } = *self;
        let bias_tokens = &self.bias_tokens;

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

        assert_eq!(token_ids.len(), probs.len());
        let update_vec: Vec<(TokenId, Prob)> = token_ids
            .iter()
            .zip(&probs)
            .map(|(x, y)| (*x, y.clone()))
            .collect();
        self.trie.update(update_vec);
        debug!("{:?}", self.trie);

        let dist = self.trie.distribution();
        let weighted = dist.weighted_index().expect("WeightedIndex error");
        let idx = weighted.sample(&mut self.ciphertext);
        let tokens = &dist.tokens[idx];
        return if tokens.len() == 1 {
            info!("Simple distr: {}", tokens[0]);
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
            info!("Resampled {} from subtrie {}", token, &dist.reprs[idx]);
            token
        };
    }
}

/// An RNG that uses a pre-initialized BitVector as source of randomness
#[derive(Debug, Default)]
struct BitVecSampler {
    source: BitVec<u8>,
    pub offset: usize,
}

impl BitVecSampler {
    fn new(source: BitVec<u8>) -> Self {
        BitVecSampler { source, offset: 0 }
    }
}

impl RngCore for BitVecSampler {
    fn next_u32(&mut self) -> u32 {
        let range = self.offset..self.offset + 32;
        // TODO pad if offset
        let next = self
            .source
            .get(range)
            .map_or_else(|| rand::thread_rng().next_u32(), |x| x.load::<u32>());
        self.offset += 32;
        next
    }

    fn next_u64(&mut self) -> u64 {
        let mut next = self.next_u32() as u64;
        next = next << 32 | (self.next_u32() as u64);
        next
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Ok(self.fill_bytes(dest))
    }
}

#[derive(Debug)]
pub(crate) struct MeteorSamplerContainer {
    inner: Mutex<MeteorSampler>,
}

impl Default for MeteorSamplerContainer {
    fn default() -> Self {
        MeteorSamplerContainer {
            inner: Mutex::new(MeteorSampler::default()),
        }
    }
}

impl MeteorSamplerContainer {
    pub fn new(trie: TokenTrie, ciphertext: Vec<u8>) -> Self {
        MeteorSamplerContainer {
            inner: Mutex::new(MeteorSampler {
                trie,
                ciphertext: BitVecSampler::new(ciphertext.view_bits().to_bitvec()),
                ..Default::default()
            }),
        }
    }
}

impl Sampler for MeteorSamplerContainer {
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

#[cfg(test)]
mod tests {
    use crate::meteor_sampler::BitVecSampler;
    use bitvec::view::BitView;
    use rand::RngCore;

    #[test]
    fn test_meteor_sampler() -> anyhow::Result<()> {
        let vec: Vec<u8> = vec![0, 1, 2, 3];
        let mut sampler = BitVecSampler::new(vec.view_bits().to_bitvec());
        let next = sampler.next_u32();
        let expect =
            (vec[3] as u32) << 24 | (vec[2] as u32) << 16 | (vec[1] as u32) << 8 | (vec[0] as u32);
        assert_eq!(next, expect);
        Ok(())
    }
}
