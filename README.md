# Explainable Fake News Detection System (NLP & XAI)

A full-stack machine learning application that detects fake news articles using a fine-tuned BERT model and provides transparent, word-level explanations using **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## 🚀 Live Deployments

* **Frontend Web App (Hugging Face Spaces):** [Fake News Detector UI](https://huggingface.co/spaces/Samriddha03/fake-news-backend)
* **Backend REST API (Render):** [FastAPI Swagger Documentation](https://fake-news-xai-app-1.onrender.com/docs)

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Python, Hugging Face Hub (`InferenceClient`), LIME, NumPy, Render
* **Frontend:** React, Vite, JavaScript, HTML/CSS, Hugging Face Static Spaces
* **Machine Learning Model:** Fine-tuned BERT binary classification (`Samriddha03/fake-news-bert-xai`)

---

## 📂 Project Structure

```text
fake_news_xai_app/
│
├── backend/
│   ├── main.py            # FastAPI application & LIME/BERT inference pipeline
│   ├── requirements.txt   # Python dependencies
│   └── Render.yaml        # Render deployment configuration
│
├── frontend/              # React application source code
│   ├── src/               # React components and App.jsx logic
│   ├── package.json       # Node dependencies and build scripts
│   └── dist/              # Production build output
│
└── README.md

⚙️ How to Run Locally
1. Clone the Repository

git clone [https://github.com/Samriddha03/Fake_news_XAI_app.git](https://github.com/Samriddha03/Fake_news_XAI_app.git)
cd Fake_news_XAI_app
2. Run the Backend Locally

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
The FastAPI server will start at http://127.0.0.1:8000.

3. Run the Frontend Locally
Open a new terminal window, navigate to the frontend directory, install dependencies, and start the development server:


cd frontend
npm install
npm run dev
👤 Author
Samriddha Chakraborty — https://github.com/Samriddha03


---

### How to push this final update to GitHub:
1. Replace the contents of your local `README.md` file with the code block above.
2. Open PowerShell in your project folder (`D:\Documents\fake_news_xai_app`) and run:
   ```powershell
   git add README.md
   git commit -m "docs: include live frontend UI link in README"
   git push origin main