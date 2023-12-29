It has been shown that fragments of the data on which an LLM was trained can be recovered from the trained model.

    Extracting Training Data from Large Language Models
    Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel
    USENIX Security Symposium, 2021
    https://arxiv.org/abs/2012.07805

This repo focuses on LLMs for code generation.
Many companies are training their own LLMs on proprietary codebases nowadays.
Naturally, they might want to give public access to the model without leaking their codebase.

Let's implement the data extraction attack on [bigcode/starcoderbase-3b](https://huggingface.co/bigcode/starcoderbase-3b) which is a GPT2 architecture trained on [The Stack](https://huggingface.co/datasets/bigcode/the-stack), a 6TB dataset of permissive-licenced code scrapped from GitHub.

## How to run

1. Log into Hugging Face Hub, read and accept the licence agreement to get access
   to the [model](https://huggingface.co/bigcode/starcoderbase-3b).

2. Log into huggingface locally, e.g. with `huggingface-cli login` shell
   command.

3. Install requirements:
```
pip install -r requirements.txt
```

4. Let the model generate a lot of text, calculate some perplexity-based
    metrics by running the script (assumes Unix-like OS, check the file and adopt for your OS):
```
./generate_code_chunks.sh
```
This saves the code chunks and the respective perprexities to a CSV file.

5. Explore the saved results manually in the Jupyter notebook (see `TODO`).
    - To verify that the extracted code chunk was indeed present in the training data, we can search for it on GitHub using their API, since the original dataset is too large to search locally.
    - Obtain a GitHub API token, put it into the environment variable called `GITHUB_API_TOKEN`

## Code Adoption Disclaimer

This repo is heavily based on (https://github.com/shreyansh26/Extracting-Training-Data-from-Large-Langauge-Models/), which itself is based on the [original implementation](https://github.com/ftramer/LM_Memorization/extraction.py).
Both are MIT-licenced.
Both use GPT2 and non-specific text, while this repo uses a model specifically trained to generate code.
