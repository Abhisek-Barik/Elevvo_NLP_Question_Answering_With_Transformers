import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# -------------------------------
# ✅ Build a system that answers questions based on a given context or passage
# -------------------------------

# Load dataset (only first 10 rows for quick testing)
df = pd.read_csv("SQuAD-v1.1 (Stanford).csv").dropna().reset_index(drop=True).head(10)

# -------------------------------
# ✅ Use pre-trained transformer models (e.g., DistilBERT) fine-tuned for question answering
# -------------------------------
model_name = "distilbert-base-uncased-distilled-squad"  # Fine-tuned on SQuAD
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Move model to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# ✅ Feed the model both the context and the question, and extract the correct answer span
# -------------------------------
def get_answer(context, question):
    # Encode question + context together
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]

    # Predict start and end of answer span
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1
    answer_ids = input_ids[start_idx:end_idx]

    # Decode tokens to string answer
    return tokenizer.decode(answer_ids, skip_special_tokens=True)

# -------------------------------
# ✅ Evaluate with exact match and F1 score
# -------------------------------
def compute_exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())

def compute_f1(pred, truth):
    pred_tokens = pred.lower().split()
    truth_tokens = truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Run inference and evaluate on 10 samples
exact_matches = []
f1_scores = []

print("\n--- Top 10 QA Results ---\n")

for i in range(len(df)):
    context = df.loc[i, 'context']
    question = df.loc[i, 'question']
    ground_truth = df.loc[i, 'answer']

    predicted = get_answer(context, question)
    em = compute_exact_match(predicted, ground_truth)
    f1 = compute_f1(predicted, ground_truth)

    exact_matches.append(em)
    f1_scores.append(f1)

    print(f"\nQ{i+1}: {question}")
    print(f"Predicted Answer : {predicted}")
    print(f"Ground Truth     : {ground_truth}")
    print(f"Exact Match      : {'100.00%' if em else '0.00%'}")
    print(f"F1 Score         : {f1 * 100:.2f}%")

# Print average evaluation scores
print("\n--- Final Evaluation ---")
print(f"Exact Match (EM): {sum(exact_matches) / 10 * 100:.2f}%")
print(f"Average F1 Score: {sum(f1_scores) / 10 * 100:.2f}%")
