# Explainable Fake News Detection System (NLP & XAI)

An end-to-end full-stack web application for automated fake news classification with word-level model interpretability powered by **FastAPI**, **PyTorch**, **Hugging Face Transformers**, and **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## 🌟 Key Features
- **BERT Classifier:** Analyzes news text using a fine-tuned BERT sequence classification model.
- **Explainable AI (XAI):** Computes word-level feature importance scores via LIME to highlight key terms driving the prediction.
- **Automated Model Resolution:** Automatically fetches fine-tuned model weights directly from [Hugging Face Hub](https://huggingface.co/Samriddha03/fake-news-bert-xai) on server startup—no manual large file downloads required.

---

## 🚀 Tech Stack
- **Backend:** FastAPI, PyTorch, Transformers, LIME, Uvicorn
- **Frontend:** React, HTML5/CSS3
- **Model Registry:** [Hugging Face Hub (`Samriddha03/fake-news-bert-xai`)](https://huggingface.co/Samriddha03/fake-news-bert-xai)

---

## 🛠️ Local Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 16+ & npm

---

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000


### Step 2: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node modules
npm install

# Start development server
npm run dev
# (or 'npm start' if using Create React App)

'''
📌 Usage
1. Open your browser and navigate to http://localhost:5173 (or http://localhost:3000).

2. Paste any news headline or full article text into the input field.

3. Click Analyze Article.

4. Review the Classification Label (Fake vs Real), Confidence Score, and LIME Word Feature Importance Graph.