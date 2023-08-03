mod token_trie;

use crate::token_trie::{TokenTrie, Trie};
use anyhow::Result;
use clap::Parser;
use env_logger;
use llm;
use llm::{
    InferenceFeedback, InferenceParameters, InferenceResponse, Model, ModelParameters, Prompt,
    Sampler, TokenId, Tokenizer, TokenizerSource,
};
use log::{debug, info};
use rand::{thread_rng, Rng, RngCore};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug, Parser)]
#[clap(version = "1.0", author = "Jeremy Boy <github@jboy.eu>")]
struct CliArgs {
    #[arg(long)]
    model: String,
    #[command(subcommand)]
    mode: ProgramMode,
}

#[derive(clap::Subcommand, Clone, Debug)]
enum ProgramMode {
    Inference {
        #[arg(long)]
        context: String,
    },
    #[command(arg_required_else_help = true)]
    Encode {
        #[arg(long)]
        context: String,
        #[arg(long)]
        key_file: String,
        msg: String,
    },
    #[command(arg_required_else_help = true)]
    Decode {
        #[arg(long)]
        context: String,
        #[arg(long)]
        key_file: String,
        stego_text: String,
    },
}

#[derive(Debug)]
enum InferenceCallbackError {}

impl Display for InferenceCallbackError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct MeteorSampler {
    state: u8,
}

impl MeteorSampler {
    fn sample(
        &mut self,
        _previous_tokens: &[TokenId],
        _logits: &[f32],
        _rng: &mut dyn RngCore,
    ) -> TokenId {
        self.state += 1;
        todo!()
    }
}

#[derive(Debug)]
struct MeteorSamplerContainer {
    inner: Mutex<MeteorSampler>,
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

impl Error for InferenceCallbackError {}

fn main() -> Result<()> {
    env_logger::init();
    debug!("Parse CLI args");
    let args = CliArgs::parse();
    debug!("args: {:?}", args);
    info!("Loading model {}", &args.model);
    let model_path = Path::new(&args.model);
    let llama = llm::load::<llm::models::Llama>(
        model_path,
        TokenizerSource::Embedded,
        ModelParameters {
            use_gpu: true,
            ..Default::default()
        },
        llm::load_progress_callback_stdout,
    )?;

    return match args.mode {
        ProgramMode::Inference { context } => mode_inference(llama, &context),
        ProgramMode::Encode {
            context,
            key_file,
            msg,
        } => mode_encode(llama, &context, &key_file, &msg),
        ProgramMode::Decode {
            context,
            key_file,
            stego_text,
        } => mode_decode(llama, &context, &key_file, &stego_text),
    };
}

fn mode_inference<M: Model>(model: M, context: &str) -> Result<()> {
    let mut session = model.start_session(Default::default());
    let mut s = String::new();
    let res = session.infer(
        &model,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: Prompt::Text(context),
            parameters: &InferenceParameters::default(),
            play_back_previous_tokens: true,
            maximum_token_count: None,
        },
        // llm::OutputRequest
        &mut Default::default(),
        |t| -> Result<InferenceFeedback, InferenceCallbackError> {
            match t {
                InferenceResponse::InferredToken(t) | InferenceResponse::PromptToken(t) => {
                    s.push_str(&t);
                }
                _ => {}
            }
            Ok(InferenceFeedback::Continue)
        },
    )?;
    println!("\n\nInference stats:\n{res}");
    println!("{}", s);
    Ok(())
}

const KEY_BYTES: usize = 128;
fn load_key(filename: &str) -> Result<[u8; KEY_BYTES]> {
    let f = File::open(filename);
    return match f {
        Ok(mut f) => {
            info!("Key file loaded");
            let mut key_bytes = [0; KEY_BYTES];
            f.read_exact(&mut key_bytes)?;
            Ok(key_bytes)
        }
        Err(_) => {
            info!("Key file does not exist. Creating random key");
            let mut f = File::create(filename)?;
            let mut rng = rand::thread_rng();
            let mut random_bytes = [0; KEY_BYTES];
            for i in 0..KEY_BYTES {
                random_bytes[i] = rng.gen();
            }
            f.write_all(&random_bytes)?;
            Ok(random_bytes)
        }
    };
}

fn mode_encode<M: Model>(model: M, context: &str, key_file: &str, msg: &str) -> Result<()> {
    info!("Loading key file {}...", key_file);
    let key = load_key(key_file)?;
    info!("Loading tokenizer...");
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.get_tokens();
    info!("Loaded tokenizer with {} tokens", tokens.len());
    let trie = TokenTrie::new(tokens.clone().into_iter().collect())?;
    todo!("Encode")
}

fn mode_decode<M: Model>(model: M, context: &str, key_file: &str, stego_text: &str) -> Result<()> {
    info!("Loading key file {}...", key_file);
    let key = load_key(key_file)?;
    info!("Loading tokenizer...");
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.get_tokens();
    todo!("Decode")
}

trait GetTokens<TokenId, Token> {
    fn get_tokens(&self) -> BTreeMap<TokenId, Token>;
}

impl GetTokens<TokenId, Vec<u8>> for Tokenizer {
    fn get_tokens(&self) -> BTreeMap<TokenId, Vec<u8>> {
        let token_lookup = match self {
            Tokenizer::Embedded(tokenizer) => tokenizer.token_to_id.clone(),
            Tokenizer::HuggingFace(_) => todo!(),
        };
        let map = token_lookup.iter().map(|(k, v)| (v.clone(), k.clone()));
        return BTreeMap::from_iter(map);
    }
}
