use std::fs::File;
use std::io::{Read, Write};

use anyhow::Context;
use log::info;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};

const KEY_BYTES: usize = 3 * 32;
pub(crate) struct MeteorKey {
    pub cipher_rng: StdRng,
    pub resample_rng: StdRng,
    pub pad_rng: StdRng,
}

impl MeteorKey {
    pub(crate) fn load_key(filename: &str) -> anyhow::Result<Self> {
        let f = File::open(filename);
        let key_bytes = match f {
            Ok(mut f) => {
                info!("Key file loaded");
                let mut key_bytes = [0; KEY_BYTES];
                f.read_exact(&mut key_bytes)?;
                key_bytes
            }
            Err(_) => {
                info!("Key file does not exist. Creating random key");
                let mut f = File::create(filename)?;
                let mut rng = rand::thread_rng();
                let mut key_bytes = [0; KEY_BYTES];
                for i in 0..KEY_BYTES {
                    key_bytes[i] = rng.gen();
                }
                f.write_all(&key_bytes)?;
                key_bytes
            }
        };
        Ok(MeteorKey {
            cipher_rng: StdRng::from_seed(
                key_bytes[..32]
                    .try_into()
                    .with_context(|| "wrong key size")?,
            ),
            resample_rng: StdRng::from_seed(
                key_bytes[32..64]
                    .try_into()
                    .with_context(|| "wrong key size")?,
            ),
            pad_rng: StdRng::from_seed(
                key_bytes[64..]
                    .try_into()
                    .with_context(|| "wrong key size")?,
            ),
        })
    }
}
