import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI as OpenAICompatible

# Load env vars
load_dotenv()

# CONFIG
INPUT_FILE = "data/inputs/inputs.csv"

OUTPUT_DIRS = {
    "claude": "data/outputs/claude",
    "gemini": "data/outputs/gemini2.5",
    "gpt": "data/outputs/gpt5",
    "mistral": "data/outputs/mistral",
    "qwen": "data/outputs/qwen3-max",
}

PROMPTS = {
    "zero_shot": "prompt/zero_shot_prompt.txt",
    "few_shot": "prompt/few_shot_prompt.txt",
}

# LOAD PROMPTS
def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

zero_prompt = load_prompt(PROMPTS["zero_shot"])
few_prompt = load_prompt(PROMPTS["few_shot"])

# LOAD DATA
def load_inputs():
    rows = []
    with open(INPUT_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row["SHI-ns"])
    return rows

sentences = load_inputs()

# OUTPUT CLEANING FUNCTION
def clean_output(text):
    if not text:
        return ""

    text = text.strip()

    # Split into lines
    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        # Remove labels
        if lower.startswith("output:"):
            line = line[len("output:"):].strip()
        elif lower.startswith("input:"):
            continue

        # Skip explanations
        if any(keyword in lower for keyword in [
            "explanation", "here is", "normalized sentence", "translation"
        ]):
            continue

        cleaned_lines.append(line)

    # Return ONLY first valid sentence
    if cleaned_lines:
        return cleaned_lines[0]

    return ""


# MODEL CALLS
# GPT
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


# Claude
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def call_claude(prompt):
    resp = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()


# Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def call_gemini(prompt):
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()


# Mistral
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

def call_mistral(prompt):
    resp = mistral_client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content=prompt)]
    )
    return resp.choices[0].message.content.strip()


# Qwen
qwen_client = OpenAICompatible(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def call_qwen(prompt):
    resp = qwen_client.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


# Dispatch tablle
MODEL_FUNCS = {
    "claude": call_claude,
    "gemini": call_gemini,
    "gpt": call_gpt,
    "mistral": call_mistral,
    "qwen": call_qwen,
}

# Pipeline
def build_prompt(template, sentence):
    return template.replace("{input}", sentence)


def run_model(model_name, prompt_type, template):
    output_path = Path(OUTPUT_DIRS[model_name])
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{prompt_type}_normalization.txt"

    call_fn = MODEL_FUNCS[model_name]

    with open(file_path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentences):
            full_prompt = build_prompt(template, sent)

            try:
                raw_output = call_fn(full_prompt)
                output = clean_output(raw_output)

                if not output:
                    output = "[EMPTY]"

            except Exception as e:
                output = f"[ERROR] {str(e)}"

            f.write(output + "\n")

            if i % 10 == 0:
                print(f"{model_name} | {prompt_type} | {i}/{len(sentences)}")

    print(f"Saved -> {file_path}")


# MAIN
if __name__ == "__main__":
    for model in MODEL_FUNCS.keys():
        run_model(model, "zero_shot", zero_prompt)
        run_model(model, "few_shot", few_prompt)

    print("All generations completed.")