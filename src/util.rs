use bitvec::order::Lsb0;
use bitvec::view::BitView;
use log::warn;
use std::ops::AddAssign;

/// Calculates the prefix bit length between `a' and `b'.
pub fn prefix_bit_length(a: u64, b: u64) -> u8 {
    let zipped = a
        .view_bits::<Lsb0>()
        .iter()
        .zip(b.view_bits::<Lsb0>())
        .rev()
        .enumerate();
    for (idx, (ai, bi)) in zipped {
        if ai != bi {
            return idx as u8;
        }
    }
    64
}

pub trait Cumsum {
    type Element;
    fn cumsum(&self, init: Self::Element) -> Vec<Self::Element>;
}

impl<T: AddAssign + Copy> Cumsum for Vec<T> {
    type Element = T;
    fn cumsum(&self, init: T) -> Vec<T> {
        return self
            .iter()
            .scan(init, |acc, x| {
                *acc += *x;
                Some(*acc)
            })
            .collect();
    }
}

#[derive(Debug)]
pub struct Rescale {
    pub probs: Vec<u64>,
    pub cumsum: Vec<u64>,
}

pub fn cumsum_rescale(probs: Vec<f32>) -> Rescale {
    let range = u64::MAX - 0;
    let threshold = 1.0 / range as f32;
    let probs = probs
        .into_iter()
        // cutoff probabilities that would be rounded to 0
        .filter(|p| *p >= threshold);
    let total_weights = probs.clone().sum::<f32>();
    let scaler = range as f32 / total_weights;
    let mut cumsum: Vec<u128> = probs
        // rescale to range
        .map(|p| p * scaler)
        .map(|p| p.round() as u128)
        .collect::<Vec<u128>>()
        .cumsum(0);
    // find overflow
    let mut cumsum_iter = cumsum.iter_mut();
    fn find_overfill(x: &&mut u128) -> bool {
        **x > u64::MAX as u128
    }
    let overfill = cumsum_iter.find(find_overfill);
    if let Some(overfill) = overfill {
        warn!(
            "Rescale caused overflow, remove weight {} from last element",
            (*overfill - range as u128) as f32 / *overfill as f32
        );
        *overfill = range as u128;
    }
    assert_eq!(cumsum_iter.find(find_overfill), None);
    let probs: Vec<u64> = cumsum
        .iter()
        .scan(0, |prev, curr| {
            let p = curr - *prev;
            *prev = *curr;
            Some(p as u64)
        })
        .collect();
    let cumsum = cumsum.into_iter().map(|v| v as u64).collect();
    Rescale { probs, cumsum }
}

#[cfg(test)]
mod tests {
    use crate::util::{cumsum_rescale, prefix_bit_length, Rescale};
    use approx::assert_abs_diff_eq;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_cumsum_rescale() -> anyhow::Result<()> {
        init();
        let vec = vec![0.5, 0.3, 0.2];
        let Rescale { probs, cumsum } = cumsum_rescale(vec);
        assert_eq!(cumsum.len(), 3);
        assert_eq!(cumsum[cumsum.len() - 1], u64::MAX);
        assert_eq!(cumsum[0], (0.5 * u64::MAX as f32).floor() as u64);
        assert_eq!(probs[0], (0.5 * u64::MAX as f32).floor() as u64);
        assert_eq!(cumsum[1], (0.8 * u64::MAX as f32).floor() as u64);
        assert_eq!(probs[1], (0.3 * u64::MAX as f32).floor() as u64);
        assert_eq!(cumsum[2], (1.0 * u64::MAX as f32).floor() as u64);
        // we can't directly assert the last element, as rescaling might have updated its weight
        assert_abs_diff_eq!(
            probs[2],
            (0.2 * u64::MAX as f32).floor() as u64,
            epsilon = (f32::EPSILON * u64::MAX as f32) as u64
        );
        println!("{:?}", cumsum);
        Ok(())
    }

    #[test]
    fn test_prefix_length() -> anyhow::Result<()> {
        init();
        #[derive(Debug)]
        struct TCase {
            a: u64,
            b: u64,
            res: u8,
        }
        let io: Vec<TCase> = vec![
            TCase {
                a: 0,
                b: 0,
                res: 64,
            },
            TCase {
                a: u64::MAX,
                b: u64::MAX,
                res: 64,
            },
            TCase {
                a: 0,
                b: u64::MAX,
                res: 0,
            },
            TCase {
                a: u64::MAX,
                b: 0,
                res: 0,
            },
            TCase {
                a: 0,
                b: u64::MAX,
                res: 0,
            },
            TCase {
                a: 1 << 63,
                b: 0,
                res: 0,
            },
            TCase {
                a: u64::MAX,
                b: 1 << 63,
                res: 1,
            },
            TCase {
                a: 0xFFFFFFFF00000000,
                b: 0xFFFFFFFFFFFFFFFF,
                res: 32,
            },
            TCase {
                a: 0xFFFFFFFE00000000,
                b: 0xFFFFFFFFFFFFFFFF,
                res: 31,
            },
        ];
        for p in io {
            let TCase { a, b, res } = p;
            assert_eq!(prefix_bit_length(a, b), res, "{:?}", p);
        }
        Ok(())
    }
}
