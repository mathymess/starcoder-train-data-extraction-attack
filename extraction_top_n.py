import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from tqdm import tqdm
import pandas as pd

import argparse
import math
import zlib
from collections import defaultdict


def load_tokenizer(model_name: str):
    """
    Load a tokenizer from HuggingFace, move onto a device.
    Set the padding side to left and the padding token to EOS.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, device: str):
    """
    Load a transformer model for CausalLM from HuggingFace.
    Move onto a device. Set eval mode.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def calculate_perplexity(input_sentence: str, model, tokenizer, device: str) -> float:
    """
    Calculate exp(loss), where loss is obtained py passing tokenized input sentence to the model
    with the labels set as the same tokenized input (the shifting of the labels is done internally)
    https://huggingface.co/docs/transformers/tasks/language_modeling
    """
    input_ids = tokenizer(input_sentence).input_ids
    input_ids = torch.tensor(input_ids).to(device)
    output = model(input_ids, labels=input_ids)
    return torch.exp(output.loss).cpu().item()


@torch.no_grad()
def calculate_perplexity_sliding(input_sentence, model, tokenizer, device, window_size=50):
    """
    Calculate min(exp(loss)) over a sliding window
    """
    input_ids = tokenizer(input_sentence).input_ids
    input_ids = torch.tensor(input_ids).to(device)
    min_perplexity = torch.inf
    for start_idx in range(input_ids.shape[0] - window_size):
        input_window = input_ids[start_idx : start_idx + window_size]
        output = model(input_window, labels=input_window)
        min_perplexity = min(min_perplexity, torch.exp(output.loss).cpu().item())
    return min_perplexity


def append_results_on_batch(
    all_scores, all_texts: list[str], batch: list[str], model3b, model1b, tokenizer, device
):
    for text in batch:
        # Calculate perplexity of StarCoder-3b and 1b on each generated text
        perplexity_3b = calculate_perplexity(text, model3b, tokenizer, device)
        perplexity_1b = calculate_perplexity(text, model1b, tokenizer, device)

        # Calculate perplexity of StarCoder-3b on each lower-cased text
        perplexity_3b_lowercase = calculate_perplexity(text.lower(), model3b, tokenizer, device)

        # Calculate zlib entropy of the generated code chunk
        zlib_entropy = len(zlib.compress(bytes(text, "utf-8")))

        # Calculate minimum perplexity of GPT2-XL across any sliding window of 50 tokens
        perplexity_3b_slidingwindow = calculate_perplexity_sliding(
            text.lower(), model3b, tokenizer, device
        )

        all_texts.append(text)
        all_scores["perp_1b"].append(perplexity_3b)
        all_scores["perp_3b"].append(perplexity_1b)
        all_scores["zlib_entropy"].append(zlib_entropy)
        all_scores["perp_3b_lowercase"].append(perplexity_3b_lowercase)
        all_scores["sliding_window"].append(perplexity_3b_slidingwindow)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("Loading models...")
    tokenizer = load_tokenizer("bigcode/starcoderbase-3b")
    model1b = load_model("bigcode/starcoderbase-1b", device)
    print("StarCoder 1b model loaded!")
    model3b = load_model("bigcode/starcoderbase-3b", device)
    print("StarCoder 3b model loaded!")

    max_sequence_length = 256  # number of tokens to generate (from paper)
    k_in_top_k_sampling = 40  # k in top_k sampling (from paper)

    num_batches = int(math.ceil(args.N / args.batch_size))
    new_tot = num_batches * args.batch_size

    generated_samples = []
    scores = defaultdict(list)
    with tqdm(total=new_tot) as pbar:
        for batch in range(num_batches):
            # Create empty prompts
            prompts = [tokenizer.eos_token] * args.batch_size
            encoded_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

            # Batched sequence generation
            generated_sequences = model3b.generate(
                input_ids=encoded_prompts.input_ids,
                attention_mask=encoded_prompts.attention_mask,
                max_length=max_sequence_length,
                do_sample=True,
                top_k=k_in_top_k_sampling,
                top_p=1.0,
            )
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

            append_results_on_batch(scores, generated_samples, generated_texts, model3b, model1b, tokenizer, device)
            pbar.update(args.batch_size)

    df = pd.DataFrame({"texts": generated_samples, **scores})
    df.drop_duplicates(subset="texts")
    df.to_csv(args.outfile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=20, type=int, help="Number of samples to generate")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument(
        "--random_seed", default=42, type=int, help="RNG Seed to ensure reproducible generation"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Path to file to which the CSV of pandas dataset with results will be written",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.random_seed)
    main(args)
