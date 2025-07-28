# 💬 NLP Task 6 – Question Answering with Transformers 

## 🚀 ELEVVO Internship | Abhisek Barik  
### 🌟 Level-3 → NLP Task 6 ✅ + Bonus 💡 Completed  

---

## 📌 Task Overview  
Build a robust **Question Answering (QA) system** using transformer-based models, enabling the model to extract accurate answers from context passages. Evaluated on **Exact Match (EM)** and **F1 Score** using Stanford-style dataset.

---

## 🧠 What’s Inside  
- ✅ Built a QA system using **DistilBERT**, **BERT**, **RoBERTa**, and **ALBERT**
- 📈 Evaluated on **custom + SQuAD-format** dataset (title, context, question, answer, answer_start, answer_end)
- 💯 Achieved **90% Exact Match**, **95% Avg F1 Score**
- 🧪 Model comparison with performance metrics
- 🌐 Deployed a **Streamlit interface** for live QA  

---

## 📊 Model Performance Summary  

| 🔍 Model      | ⚡ Exact Match (%) | 🎯 F1 Score (%) |
|--------------|-------------------|----------------|
| 🏆 DistilBERT | 90.00             | 95.00          |
| BERT         | 90.00             | 92.67          |
| RoBERTa      | 80.00             | 93.00          |
| ALBERT       | 70.00             | 80.00          |

➡️ **RoBERTa** impressed with its high F1, though **DistilBERT** took the crown with both **speed** and **precision**  
➡️ **ALBERT** trades performance for lightweight efficiency — great for constrained devices  
➡️ **BERT** delivers a strong balance of accuracy and stability — a reliable baseline

---

## 🧰 Tech Stack  
**Python** 🐍 | **Hugging Face Transformers** 🤗 | **PyTorch** ⚙️ | **pandas**, **NumPy**, **tqdm** 📊  
**Evaluation**: Exact Match, F1 Score 🎯 | **Streamlit** for QA App 🌐  
**Dataset Format**: `title`, `context`, `question`, `answer`, `answer_start`, `answer_end`

---

## 🔄 Workflow  

### ✅ 1. Data Handling  
- Parsed context–question–answer triplets  
- Calculated `answer_end` positions  
- Prepared Stanford-style QA JSON format

### 🧠 2. Model Implementation  
- Loaded **DistilBERT**, **BERT**, **RoBERTa**, and **ALBERT** from 🤗 Transformers  
- Tokenized input for QA (context + question)  
- Extracted start and end logits to get answer spans

### 📈 3. Evaluation  
- Custom function to compute **Exact Match** and **F1 Score**  
- Averaged across all models

### 🌐 4. Streamlit App  
- Built real-time interface to input passage and question  
- Displayed extracted answers using selected models  

---

## 🚀 Bonus Features  

### 🔁 Model Comparison Summary  
Tested all four models under the same dataset and evaluation criteria, revealing trade-offs between size, speed, and accuracy.

### 🧪 Evaluation Insights  
- Normalized answers before matching  
- Highlighted text spans  
- Tracked answer position accuracy  

---

## 💡 Learnings  
- Question Answering is an **extractive NLP task**, best tackled with fine-tuned transformers  
- Different transformer models have strengths: **speed**, **accuracy**, or **compactness**  
- Simple UI via Streamlit brings **hands-on interactivity** to QA pipelines  

---

## 📚 Concepts Covered  
- ✂️ Tokenization, Attention Masks  
- 🔍 Context-Question Encoding  
- 🧠 Transformer Architecture (BERT family)  
- 📏 Evaluation: EM & F1 Metrics  
- 🌐 Streamlit App Deployment  

---
