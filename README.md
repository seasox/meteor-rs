# Meteor-RS

[Meteor](https://meteorfrom.space) is a cryptographically secure steganography for realistic distributions. It utilizes the randomness used for sampling 
from generative models to embed a hidden message. This project implements Meteor in Rust while fixing tokenization issues with
the authors' original approach.

## Getting started
1. Clone this repository and build using cargo:
   ```
   git clone --recurse-submodules https://github.com/seasox/meteor-rs meteor-rs
   cd meteor-rs
   cargo build
   ```
2. Prepare a model in the models/ directory. You can either fetch a GGML compatible model and place it in the 
   `models` directory or convert a model (and possibly quantize) one using a tool such as llama.cpp:
   ```
   cd path/to/llama.cpp
   python3 convert.py --outfile llama-2-13b-ggml.bin path/to/llama-2-13b/
   bin/quantize llama-2-13b-ggml.bin llama-2-13b-ggml-q4_0.bin q4_0
   ln llama2-13b-ggml-q4_0.bin path/to/meteor-rs/models/
   ```
3. Run inference to check your setup:
   ```
   cargo run --package meteor-rs --bin meteor-rs -- --model models/llama-2-13b-ggml-q4_0.bin inference --context "Rust is a cool programming language because"
   ```
   This should, after a while, print a generated message:
   ```
   Rust is a cool programming language because it allows us to write code in C, the most popular and widely-used low level system development programming language.
   This book will teach you how to use Rust from scratch! You'll start with learning about basics of this very versatile language by reading through its features as well as getting your hands on building blocks like types and functions before moving onto more complex concepts such as ownership or borrowing which are key principles behind rustic style programming culture so that when we want something done quickly (and safely) there won’t be any confusion around who owns what – whether they belong together logically within one place versus another part elsewhere downstream where work needs doing first but still gets finished off properly afterwards if necessary due process takes too long time wise etc.. All these fundamentals play an important role especially during early days while developing new projects using Rust so don‘t forget them at all cost!”
   ```
   It also prints inference statistics:
   ```
   Inference stats:
   feed_prompt_duration: 47273ms
   prompt_tokens: 9
   predict_duration: 370131ms
   predict_tokens: 195
   per_token_duration: 1898.108ms
   ```
4. You are now ready for generating your first stegotext! Run meteor-rs in `encode` mode to hide a message:
   ```
   cargo run --package meteor-rs --bin meteor-rs -- \
      --model models/llama-2-13b-ggml-q4_0.bin \
      encode \
      --context "Rust is a cool programming language because" \ 
      --key-file key.bin \ 
      "attack@dawn"
   ```

