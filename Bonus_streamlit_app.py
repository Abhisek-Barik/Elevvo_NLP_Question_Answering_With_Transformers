import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load default QA model
model_name = "distilbert-base-uncased-distilled-squad"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("🧠 Ask Me Anything - QA with Transformers")
context = st.text_area("📜 Enter Passage/Context", height=200)
question = st.text_input("❓ Enter Your Question")

if st.button("Get Answer") and context and question:
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_ids = input_ids[start_idx:end_idx]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    st.success(f"🗨️ Answer: **{answer}**")
