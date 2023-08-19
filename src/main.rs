use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use clap::Parser;
use env_logger;
use llm;
use llm::samplers::default_samplers;
use llm::{
    InferenceFeedback, InferenceParameters, InferenceResponse, InferenceStats, Model,
    ModelParameters, Prompt, TokenId, Tokenizer, TokenizerSource,
};
use llm_samplers::prelude::Sampler;
use log::{debug, error, info};
use rand::Rng;

use crate::key::MeteorKey;
use crate::meteor_sampler::{MeteorDecodeSampler, MeteorEncodeSampler};
use crate::token_trie::{TokenTrie, Trie};

mod key;
mod meteor_sampler;
mod token_trie;
mod util;

#[derive(Debug, Parser)]
#[clap(version = "1.0", author = "Jeremy Boy <github@jboy.eu>")]
struct CliArgs {
    #[arg(long)]
    model: String,
    #[command(subcommand)]
    mode: ProgramMode,
    #[clap(long)]
    model_type: ModelType,
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
    EncodeDecode {
        #[arg(long)]
        context: String,
        #[arg(long)]
        key_file: String,
        msg: String,
    },
}

#[derive(Clone, Debug, Parser)]
enum ModelType {
    Chat,
    TextCompletion,
}

impl ModelType {
    fn context(&self, user_context: &str) -> String {
        match self {
            ModelType::Chat => format!("SYSTEM: You are Bob, a friend of Alice. Alice and Bob are best friends having a casual conversation. Alice is the user, and you are Bob, the AI assistant. NEVER reveal that you are an AI assistant. Instead, have a casual, respectful conversation with Alice about everyday topics. IMPORTANT: DO NOT REVEAL YOU ARE NOT HUMAN!\nUser: {}\nASSISTANT:", user_context),
            ModelType::TextCompletion => user_context.to_string(),
        }
    }
}

impl FromStr for ModelType {
    type Err = fmt::Error;
    fn from_str(s: &str) -> std::result::Result<Self, fmt::Error> {
        match s.to_ascii_lowercase().as_str() {
            "chat" => Ok(ModelType::Chat),
            "text-completion" => Ok(ModelType::TextCompletion),
            _ => Err(std::fmt::Error),
        }
    }
}
#[derive(Debug)]
enum InferenceCallbackError {}

impl Display for InferenceCallbackError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
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
        ProgramMode::Inference { context } => {
            let rng = rand::thread_rng();
            mode_inference(llama, &args.model_type.context(&context), rng)
        }
        ProgramMode::Encode {
            context,
            key_file,
            msg,
        } => mode_encode(&llama, &args.model_type.context(&context), &key_file, &msg).map(|_| ()),
        ProgramMode::Decode {
            context,
            key_file,
            stego_text,
        } => mode_decode(
            &llama,
            &args.model_type.context(&context),
            &key_file,
            &stego_text,
        )
        .map(|_| ()),
        ProgramMode::EncodeDecode {
            context,
            key_file,
            msg,
        } => mode_decode(
            &llama,
            &args.model_type.context(&context),
            &key_file,
            &mode_encode(&llama, &args.model_type.context(&context), &key_file, &msg)?,
        )
        .map(|_| ()),
    };
}

fn mode_inference<M: Model>(model: M, context: &str, rng: impl Rng) -> Result<()> {
    let context: Vec<TokenId> = model
        .tokenizer()
        .tokenize(&context, false)?
        .iter()
        .map(|v| v.1)
        .collect();
    let (res, s) = infer(&model, &context, rng, default_samplers())?;
    println!("\n\nInference stats:\n{res}");
    println!("{}", s);
    Ok(())
}

fn mode_encode<M: Model>(model: &M, context: &str, key_file: &str, msg: &str) -> Result<String> {
    info!("Loading key file {}...", key_file);
    let key = MeteorKey::load_key(key_file)?;
    info!("Loading tokenizer...");
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.get_tokens();
    info!("Loaded tokenizer with {} tokens", tokens.len());
    info!("EOT: {:?}", tokenizer.token(model.eot_token_id() as usize));
    info!(
        "BOT: {:?}",
        model.bot_token_id().map(|t| tokenizer.token(t as usize))
    );
    let context: Vec<TokenId> = tokenizer
        .tokenize(&context, false)?
        .iter()
        .map(|v| v.1)
        .collect();
    let trie = TokenTrie::new(tokens.clone().into_iter().collect())?;
    assert!(trie.lookup(&model.eot_token_id()).is_some());
    let sampler = Arc::new(Mutex::new(MeteorEncodeSampler::new(
        trie,
        msg.as_bytes().to_vec(),
        key.cipher_rng,
        key.pad_rng,
    )?));
    let (res, s) = infer(model, &context, key.resample_rng, sampler)?;
    info!("Inference stats: {}", res);
    println!("{}", "=".repeat(80));
    println!("{}", s);
    println!("{}", "=".repeat(80));
    Ok(s)
}

fn mode_decode<'a, M: Model>(
    model: &M,
    context: &str,
    key_file: &str,
    stego_text: &str,
) -> Result<String> {
    let mut stego_text = String::from(stego_text);
    if stego_text.eq("-") {
        stego_text.clear();
        // load stego text from stdin
        std::io::stdin().read_to_string(&mut stego_text)?;
    }
    info!("Loaded stego text \"{}\"", &stego_text);

    assert_eq!(&stego_text[..context.len()], context);
    stego_text.drain(..context.len());

    info!("Loading key file {}...", key_file);
    let key = MeteorKey::load_key(key_file)?;
    info!("Loading tokenizer...");
    let tokenizer = model.tokenizer();
    let tokens = tokenizer.get_tokens();
    info!("Loaded tokenizer with {} tokens", tokens.len());
    let trie = TokenTrie::new(tokens.clone().into_iter().collect())?;
    let context_tokens: Vec<TokenId> = tokenizer
        .tokenize(&context, false)?
        .iter()
        .map(|v| v.1)
        .collect();
    let mut special_token_ids = vec![model.eot_token_id()];
    if let Some(bot) = model.bot_token_id() {
        special_token_ids.push(bot);
    }
    let sampler = Arc::new(Mutex::new(MeteorDecodeSampler::new(
        trie,
        tokenizer,
        context_tokens.clone(),
        key.cipher_rng,
        stego_text.as_bytes().to_vec(),
        special_token_ids,
    )?));
    let (stats, recovered_stego) =
        infer(model, &context_tokens, key.resample_rng, sampler.clone())?;
    info!("{:?}", stats);
    let recovered_msg = sampler.lock().unwrap().recovered_bits.clone();
    debug!("{}", recovered_msg);
    let (_, byte_aligned, _) = unsafe { recovered_msg.align_to::<u8>() };
    let bytes = byte_aligned.to_bitvec();
    let bytes = bytes.as_raw_slice();
    debug!("{:?}", bytes);
    let recovered_msg = match String::from_utf8(bytes.to_vec()) {
        Ok(s) => s,
        Err(e) => {
            error!("msg is not a string: {}", e);
            recovered_msg.to_string()
        }
    };
    println!("{}", recovered_msg);
    assert_eq!(stego_text, recovered_stego[context.len()..]);
    Ok(recovered_msg)
}

fn infer<M: Model>(
    model: &M,
    context: &[TokenId],
    mut rng: impl Rng,
    sampler: Arc<Mutex<dyn Sampler<TokenId, f32>>>,
) -> Result<(InferenceStats, String)> {
    let mut session = model.start_session(Default::default());
    let mut s = String::new();
    let res = session.infer(
        model,
        &mut rng,
        &llm::InferenceRequest {
            prompt: Prompt::Tokens(context),
            parameters: &InferenceParameters {
                sampler,
                ..Default::default()
            },
            play_back_previous_tokens: true,
            maximum_token_count: None,
        },
        // llm::OutputRequest
        &mut Default::default(),
        |t| -> Result<InferenceFeedback, InferenceCallbackError> {
            match &t {
                InferenceResponse::InferredToken(t) | InferenceResponse::PromptToken(t) => {
                    s.push_str(t);
                }
                _ => {}
            }
            Ok(InferenceFeedback::Continue)
        },
    )?;
    Ok((res, s))
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

/*
struct MockModel {}

impl Model for MockModel {
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        todo!()
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        todo!()
    }

    fn tokenizer(&self) -> &Tokenizer {
        todo!()
    }

    fn context_size(&self) -> usize {
        todo!()
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        todo!()
    }

    fn eot_token_id(&self) -> TokenId {
        todo!()
    }

    fn supports_rewind(&self) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{mode_decode, mode_encode};
    use llm::{ModelParameters, TokenizerSource};
    use std::path::Path;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }
    #[test]
    fn test_open_llama_encode_decode() -> anyhow::Result<()> {
        init();
        let model_path = Path::new("models/open_llama_7b-q4_0-ggjt.bin");
        let llama = llm::load::<llm::models::Llama>(
            model_path,
            TokenizerSource::Embedded,
            ModelParameters {
                use_gpu: true,
                ..Default::default()
            },
            llm::load_progress_callback_stdout,
        )?;
        let key = "key.bin";
        let context = "Unit Tests are important for software quality because";
        let msg = "hello world";
        let stegotext = mode_encode(&llama, context, key, msg)?;
        let recovered = mode_decode(&llama, context, key, &stegotext)?;
        assert_eq!(recovered, msg);
        Ok(())
    }
}
*/
