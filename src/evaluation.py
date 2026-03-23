import pandas as pd 

from Levenshtein import distance as levenshtein_distance
from jiwer import wer

def evaluate_text(ground_truth: str, predicted_text: str):
    """
    Compute WER and Levenshtein distance between a predicted text and its ground truth.

    Args:
        ground_truth (str): Reference (true) text.
        predicted_text (str): Hypothesis (predicted) text.

    Returns:
        dict: {"Levenshtein_distance": int, "WER": float}
    """
    # Compute character-level Levenshtein distance
    lev_dist = levenshtein_distance(ground_truth, predicted_text)

    # Compute word-level Word Error Rate
    wer_score = wer(ground_truth, predicted_text)

    return {
        "Levenshtein_distance": lev_dist,
        "WER": wer_score
    }

df = pd.read_excel("data/dataset.xlsx")
ground_truths = df["SHI-1"].tolist() 


files = {
    "gpt5_zero": "data/exp/gpt5/zero_shot_normalization.txt",
    "gpt5_few": "data/exp/gpt5/few_shot_normalization.txt",
    "gemini2.5_zero": "data/exp/gemini2.5/zero_shot_normalization.txt",
    "gemini2.5_few": "data/exp/gemini2.5/few_shot_normalization.txt",
    "claude4_zero": "data/exp/claude4/zero_shot_normalization.txt",
    "claude4_few": "data/exp/claude4/few_shot_normalization.txt",
    "qwen3-max_zero": "data/exp/qwen3-max/zero_shot_normalization.txt",
    "qwen3-max_few": "data/exp/qwen3-max/few_shot_normalization.txt",
    "mistral_few": "data/exp/mistral/few_shot_normalization.txt",
    "mistral_zero": "data/exp/mistral/one_shot_normalization.txt",
}

for model, file in files.items(): 
    with open(file, "r", encoding="utf-8", errors="replace") as f:
        predicted_texts = f.readlines()

    results = []
    for gt, pred in zip(ground_truths, predicted_texts):
        results.append(evaluate_text(gt, pred.strip()))

    results_df = pd.DataFrame(results)

    # Compute average metrics to 2.f digits 
    average_metrics = results_df.mean().round(3).to_dict()
    print("Average Evaluation Metrics:", model, average_metrics)