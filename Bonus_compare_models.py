import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import numpy as np

# Load cleaned dataset
df = pd.read_csv("SQuAD-v1.1 (Stanford).csv")
df = df.head(10)  # Use first 10 rows for quick comparison

# Models to compare
models = {
    "DistilBERT": "distilbert-base-uncased-distilled-squad",
    "BERT": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa": "deepset/roberta-base-squad2",
    "ALBERT": "twmkn9/albert-base-v2-squad2"
}

# Metrics
def compute_exact_match(pred, true):
    return int(pred.strip().lower() == true.strip().lower())

def compute_f1(pred, true):
    pred_tokens = pred.lower().split()
    true_tokens = true.lower().split()
    common = set(pred_tokens) & set(true_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

# For storing results
final_results = []

for model_name, model_path in models.items():
    print(f"\n🔍 Evaluating model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    em_scores = []
    f1_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        context = row['context']
        true_answer = row['answer']

        try:
            result = qa_pipeline(question=question, context=context)
            predicted_answer = result['answer']
        except:
            predicted_answer = ""

        em = compute_exact_match(predicted_answer, true_answer)
        f1 = compute_f1(predicted_answer, true_answer)

        em_scores.append(em)
        f1_scores.append(f1)

        print(f"\nQ{idx+1}: {question}")
        print(f"➡️  Predicted: {predicted_answer}")
        print(f"✅  Actual: {true_answer}")
        print(f"🎯 EM: {em * 100:.2f}%, F1: {f1 * 100:.2f}%")

    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    final_results.append({
        "Model": model_name,
        "Exact Match (%)": round(avg_em, 2),
        "F1 Score (%)": round(avg_f1, 2)
    })

    print(f"\n📊 {model_name} ➤ EM: {avg_em:.2f}% | F1: {avg_f1:.2f}%")

# 📋 Print final comparison summary table
print("\n\n========= ✅ FINAL SUMMARY COMPARISON =========\n")
summary_df = pd.DataFrame(final_results)
print(summary_df.to_string(index=False))
