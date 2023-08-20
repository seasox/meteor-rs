use std::ops::AddAssign;

use bitvec::order::Msb0;
use bitvec::prelude::BitVec;
use bitvec::view::BitView;
use log::warn;

/// Calculates the prefix bit length between `a' and `b'.
pub fn prefix_bits(a: u64, b: u64) -> BitVec<u64, Msb0> {
    let a_vec = BitVec::from(a.view_bits::<Msb0>());
    let b_vec = BitVec::from(b.view_bits::<Msb0>());

    return a_vec
        .iter()
        .zip(b_vec)
        .scan(BitVec::new(), |acc, (ai, bi)| {
            return if ai != bi {
                None
            } else {
                acc.push(*ai);
                Some(acc.clone())
            };
        })
        .last()
        .unwrap_or(BitVec::new());
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
        .iter()
        // cutoff probabilities that would be rounded to 0
        .filter(|p| **p >= threshold);
    let total_weights = probs.clone().sum::<f32>();
    let scaler = range as f32 / total_weights;
    let mut cumsum: Vec<u128> = probs
        // rescale to range
        .map(|p| p * scaler)
        .map(|p| p.round() as u128)
        .collect::<Vec<u128>>()
        .cumsum(0);
    // find over/underflow
    fn find_overfill(x: &&u128) -> bool {
        **x > u64::MAX as u128
    }
    let last = cumsum.last_mut().expect("cumsum is empty");
    if *last > range as u128 {
        warn!(
            "Rescale caused overflow, remove weight {} from last element",
            (*last - range as u128) as f32 / *last as f32
        );
        *last = range as u128;
    } else if *last < range as u128 {
        warn!(
            "Rescale caused underflow, add weight {} to last element",
            (range as u128 - *last) as f32 / *last as f32
        );
        *last = range as u128;
    }
    assert_eq!(cumsum.iter().find(find_overfill), None);
    let probs: Vec<u64> = cumsum
        .iter()
        .scan(0, |prev, curr| {
            let p = curr - *prev;
            *prev = *curr;
            Some(p as u64)
        })
        .collect();
    let cumsum: Vec<u64> = cumsum.into_iter().map(|v| v as u64).collect();
    Rescale { probs, cumsum }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use bitvec::bitvec;
    use bitvec::prelude::Lsb0;
    use bitvec::vec::BitVec;

    use crate::util::{cumsum_rescale, prefix_bits, Rescale};

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
        assert_eq!(cumsum[2], u64::MAX);
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
    fn test_cumsum_rescale_weighted() -> anyhow::Result<()> {
        init();
        let vec = vec![0.8741235, 0.062710986, 0.04070479, 0.012961245, 0.009499585];
        let Rescale { probs, cumsum } = cumsum_rescale(vec);
        assert_eq!(cumsum.len(), 5);
        assert_eq!(*cumsum.last().unwrap(), u64::MAX);
        assert_eq!(probs.iter().sum::<u64>(), *cumsum.last().unwrap());
        Ok(())
    }

    #[test]
    fn test_prefix_bits() -> anyhow::Result<()> {
        init();
        #[derive(Clone, Debug)]
        struct TCase {
            a: u64,
            b: u64,
            res: BitVec,
            first: Option<bool>,
        }
        let io: Vec<TCase> = vec![
            TCase {
                a: 0,
                b: 0,
                res: bitvec![0; 64],
                first: Some(false),
            },
            TCase {
                a: u64::MAX,
                b: u64::MAX,
                res: bitvec![1; 64],
                first: Some(true),
            },
            TCase {
                a: 0,
                b: u64::MAX,
                res: BitVec::new(),
                first: None,
            },
            TCase {
                a: u64::MAX,
                b: 0,
                res: BitVec::new(),
                first: None,
            },
            TCase {
                a: 0,
                b: u64::MAX,
                res: BitVec::new(),
                first: None,
            },
            TCase {
                a: 1 << 63,
                b: 0,
                res: BitVec::new(),
                first: None,
            },
            TCase {
                a: u64::MAX,
                b: 1 << 63,
                res: bitvec![1u8; 1],
                first: Some(true),
            },
            TCase {
                a: 0xFFFFFFFF00000000,
                b: 0xFFFFFFFFFFFFFFFF,
                res: bitvec![1u8; 32],
                first: Some(true),
            },
            TCase {
                a: 0xFFFFFFFE00000000,
                b: 0xFFFFFFFFFFFFFFFF,
                res: bitvec![1u8; 31],
                first: Some(true),
            },
            TCase {
                a: 0xAA55555555555555,
                b: 0xAA10000000055555,
                res: bitvec![1, 0, 1, 0, 1, 0, 1, 0, 0],
                first: Some(true),
            },
        ];
        for (idx, p) in io.into_iter().enumerate() {
            let TCase { a, b, res, first } = p.clone();
            let prefix = prefix_bits(a, b);
            assert_eq!(prefix, res, "{}, {:?}", idx, p);
            assert_eq!(prefix.get(0).map(|r| *r.as_ref()), first);
        }
        Ok(())
    }
}
