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
    model_family: ModelFamily,
    #[arg(long)]
    model: String,
    #[command(subcommand)]
    mode: ProgramMode,
    #[clap(long)]
    model_type: ModelType,
}

#[derive(Clone, Debug, Parser)]
enum ModelFamily {
    GptJ,
    Llama,
}

impl FromStr for ModelFamily {
    type Err = fmt::Error;
    fn from_str(s: &str) -> std::result::Result<Self, fmt::Error> {
        match s.to_ascii_lowercase().as_str() {
            "gptj" => Ok(ModelFamily::GptJ),
            "llama" => Ok(ModelFamily::Llama),
            _ => Err(std::fmt::Error),
        }
    }
}

#[derive(clap::Subcommand, Clone, Debug)]
enum ProgramMode {
    #[command(arg_required_else_help = true)]
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
    #[command(arg_required_else_help = true)]
    EncodeDecode {
        #[arg(long)]
        context: String,
        #[arg(long)]
        key_file: String,
        msg: String,
    },
    Reproduce,
}

#[derive(Clone, Debug, Parser)]
enum ModelType {
    Chat,
    TextCompletion,
}

const SYSTEM_PROMPT: &str =
    "A chat between Alice and Bob. Alice and Bob are good friends talking about daily topics.";
const HUMAN_IDENTIFIER: &str = "Alice:";
const ASSISTANT_IDENTIFIER: &str = "Bob:";

impl ModelType {
    fn context(&self, user_context: &str) -> String {
        match self {
            ModelType::Chat => format!(
                "{}\n{} {}\n{}",
                SYSTEM_PROMPT, HUMAN_IDENTIFIER, user_context, ASSISTANT_IDENTIFIER
            ),
            ModelType::TextCompletion => user_context.to_string(),
        }
    }

    fn full_output(&self, user_context: &str, inferred: &str) -> String {
        match self {
            ModelType::Chat => format!("{}{}", self.context(user_context), inferred),
            ModelType::TextCompletion => format!("{}{}", self.context(user_context), inferred),
        }
    }

    /// a sampler callback that appends inferred and prompt tokens to `out_str`
    fn callback<'a>(
        &self,
        out_str: &'a mut String,
    ) -> Box<dyn FnMut(InferenceResponse) -> Result<InferenceFeedback, InferenceCallbackError> + 'a>
    {
        return match self {
            ModelType::Chat => Box::new(llm::conversation_inference_callback(
                HUMAN_IDENTIFIER,
                |s| out_str.push_str(&s),
            )),
            ModelType::TextCompletion => {
                Box::new(|t| -> Result<InferenceFeedback, InferenceCallbackError> {
                    match &t {
                        InferenceResponse::InferredToken(t) | InferenceResponse::PromptToken(t) => {
                            out_str.push_str(t)
                        }
                        _ => {}
                    }
                    Ok(InferenceFeedback::Continue)
                })
            }
        };
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
    let model: Box<dyn Model> = match args.model_family {
        ModelFamily::Llama => Box::new(llm::load::<llm::models::Llama>(
            model_path,
            TokenizerSource::Embedded,
            ModelParameters {
                use_gpu: true,
                ..Default::default()
            },
            llm::load_progress_callback_stdout,
        )?),
        ModelFamily::GptJ => Box::new(llm::load::<llm::models::GptJ>(
            model_path,
            TokenizerSource::Embedded,
            ModelParameters {
                use_gpu: true,
                ..Default::default()
            },
            llm::load_progress_callback_stdout,
        )?),
    };
    return match args.mode {
        ProgramMode::Inference { context } => {
            let rng = rand::thread_rng();
            mode_inference(&*model, &args.model_type, &context, rng)
        }
        ProgramMode::Encode {
            context,
            key_file,
            msg,
        } => mode_encode(&*model, &args.model_type, &context, &key_file, &msg).map(|_| ()),
        ProgramMode::Decode {
            context,
            key_file,
            stego_text,
        } => mode_decode(&*model, &args.model_type, &context, &key_file, &stego_text).map(|_| ()),
        ProgramMode::EncodeDecode {
            context,
            key_file,
            msg,
        } => mode_decode(
            &*model,
            &args.model_type,
            &context,
            &key_file,
            &mode_encode(&*model, &args.model_type, &context, &key_file, &msg)?,
        )
        .map(|_| ()),
        ProgramMode::Reproduce => mode_reproduce(&*model).map(|_| ()),
    };
}

fn mode_inference(
    model: &dyn Model,
    model_type: &ModelType,
    user_context: &str,
    rng: impl Rng,
) -> Result<()> {
    let context = model_type.context(user_context);
    let context: Vec<TokenId> = model
        .tokenizer()
        .tokenize(&context, false)?
        .iter()
        .map(|v| v.1)
        .collect();
    let mut s = String::new();
    let res = infer(
        model,
        &context,
        rng,
        model_type.callback(&mut s),
        default_samplers(),
    )?;
    println!("\n\nInference stats:\n{res}");
    println!("{}", model_type.full_output(user_context, &s));
    Ok(())
}

fn mode_encode(
    model: &dyn Model,
    model_type: &ModelType,
    context: &str,
    key_file: &str,
    msg: &str,
) -> Result<String> {
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
    let context = model_type.context(context);
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
        model.eot_token_id(),
    )));
    let mut s = String::new();
    let res = infer(
        model,
        &context,
        key.resample_rng,
        model_type.callback(&mut s),
        sampler.clone(),
    )?;
    println!("Inference stats: {}", res);
    println!("Encoder stats: {:?}", sampler.lock().unwrap().stats);
    println!("{}", "=".repeat(80));
    println!("{}", s);
    println!("{}", "=".repeat(80));
    Ok(s)
}

fn mode_decode(
    model: &dyn Model,
    model_type: &ModelType,
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

    let context = model_type.context(context);

    if stego_text.starts_with(&context) {
        stego_text.drain(..context.len());
    }

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
        model.eot_token_id(),
    )));
    let mut recovered_stego = String::new();
    let stats = infer(
        model,
        &context_tokens,
        key.resample_rng,
        model_type.callback(&mut recovered_stego),
        sampler.clone(),
    )?;
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
    Ok(recovered_msg)
}

struct Stats {}
fn mode_reproduce<'a>(model: &dyn Model) -> Result<Stats> {
    let questions = vec![
        "What is the derivative of f(x)=x^2?",
        "What is the capital city of France?",
        "Which planet is known as the \"Red Planet\"?",
        "What is the largest mammal in the world?",
        "Who wrote the play \"Romeo and Juliet\"?",
        "What is the chemical symbol for gold?",
        "Which famous scientist developed the theory of relativity?",
        "What is the tallest mountain in the world?",
        "What is the process by which plants make their own food using sunlight?",
        "What is the smallest prime number?",
        "In which year did the United States declare its independence from Great Britain?",
    ];
    let answers: Vec<&str> = vec![
        "2x+C",
        "Paris",
        "Mars",
        "The blue whale",
        "William Shakespeare",
        "Au",
        "Albert Einstein",
        "Mount Everest",
        "Photosynthesis",
        "2",
        "1776",
    ];
    for (q, _) in questions
        .into_iter()
        .zip(answers)
        .collect::<Vec<(&str, &str)>>()
    {
        let a2 = mode_encode(model, &ModelType::Chat, &q, "key.bin", "msg")?;
        println!("{}: {}", q, a2);
    }
    Ok(Stats {})
}

fn infer(
    model: &dyn Model,
    context: &[TokenId],
    mut rng: impl Rng,
    callback: impl FnMut(InferenceResponse) -> Result<InferenceFeedback, InferenceCallbackError>,
    sampler: Arc<Mutex<dyn Sampler<TokenId, f32>>>,
) -> Result<InferenceStats> {
    let mut session = model.start_session(Default::default());
    let res = session.infer(
        model,
        &mut rng,
        &llm::InferenceRequest {
            prompt: Prompt::Tokens(context),
            parameters: &InferenceParameters { sampler },
            play_back_previous_tokens: true,
            maximum_token_count: None,
        },
        // llm::OutputRequest
        &mut Default::default(),
        callback,
    )?;
    Ok(res)
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
