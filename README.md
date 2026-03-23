# DATASHI: Orthography Normalization for Tashlhiyt

Code and prompts for the paper:

**DATASHI: A Parallel English–Tashlhiyt Corpus for Orthography Normalization and Low-Resource Language Processing**  
**Nasser-Eddine Monir, Zakaria Baou**

---

## Overview

This repository contains the code, prompts, and evaluation setup used for orthography normalization experiments on Tashlhiyt with large language models.

The task consists of mapping **non-standard Tashlhiyt (SHI-ns)** to **standardized Tashlhiyt (SHI-s)** under two prompting settings:

- zero-shot
- few-shot

The evaluated models are:

- GPT-5
- Claude Sonnet
- Gemini 2.5 Pro
- Mistral Large
- Qwen3-Max

---

## Repository Structure

```text
data/
  inputs/
    inputs.csv
  outputs/
    claude/
    gemini2.5/
    gpt5/
    mistral/
    qwen3-max/

prompt/
  few_shot_prompt.txt
  zero_shot_prompt.txt

src/
  evaluation.py
  run_inference.py

.env
README.md
```

---

## Data

The input CSV is structured as follows:

```text
CATEGORY,EN,SHI-s,SHI-ns
```

Model inputs use the `SHI-ns` column.

Each output file contains exactly one model inference per line, with no additional formatting.

---

## Run Inference

Install dependencies:

```bash
pip install openai anthropic google-generativeai mistralai python-dotenv
```

Create a `.env` file with the required API keys:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
QWEN_API_KEY=...
```

Run inference:

```bash
python src/run_inference.py
```

This produces:

```text
data/outputs/<model>/zero_shot_normalization.txt
data/outputs/<model>/few_shot_normalization.txt
```

---

## Evaluation

Run:

```bash
python src/evaluation.py
```

The evaluation follows the paper setup and reports:

* Word error rate (WER)
* Levenshtein distance (LD)

---

## Dataset Note

DATASHI contains 5,000 English–Tashlhiyt sentence pairs, including a 1,500-sentence subset with expert-standardized and non-standard versions for normalization experiments.

The remaining non-standard Tashlhiyt sentences will be added soon upon conference publication.

---

## Prompts

The repository includes:

* `prompt/zero_shot_prompt.txt`
* `prompt/few_shot_prompt.txt`

These are the prompts used for the normalization experiments described in the paper.

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{monir2025-datashi,
  title={DATASHI: A Parallel English--Tashlhiyt Corpus for Orthography Normalization and Low-Resource Language Processing},
  author={Monir, Nasser-Eddine and Baou, Zakaria},
  booktitle={Proceedings of LREC 2026},
  year={2026}
}
```

---

## Contact

Nasser-Eddine Monir  
Zakaria Baou