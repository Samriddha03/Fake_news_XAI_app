import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer
from huggingface_hub import InferenceClient

app = FastAPI(title="Fake News Detection & XAI API")

# Enable CORS so your React frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Model & Token Setup
MODEL_PATH = "Samriddha03/fake-news-bert-xai"
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# Initialize Hugging Face Inference Client (runs remotely on HF servers)
client = InferenceClient(api_key=HF_TOKEN)

class_names = ["Fake", "Real"]
explainer = LimeTextExplainer(class_names=class_names)

class ArticleRequest(BaseModel):
    text: str


def predictor(texts):
    """Predict function required by LIME.
    Sends text perturbations to Hugging Face Inference API and returns class probabilities.
    """
    if isinstance(texts, str):
        texts = [texts]

    all_probs = []

    for text in texts:
        try:
            # Query Hugging Face Serverless Inference API
            results = client.text_classification(text, model=MODEL_PATH)

            # Map HF output labels to probability array [Fake_prob, Real_prob]
            prob_dict = {"LABEL_0": 0.0, "LABEL_1": 0.0, "Fake": 0.0, "Real": 0.0}
            for item in results:
                prob_dict[item["label"]] = item["score"]

            # Resolve probabilities (supports both LABEL_0/LABEL_1 and Fake/Real output tags)
            fake_prob = prob_dict.get("LABEL_0", prob_dict.get("Fake", 0.0))
            real_prob = prob_dict.get("LABEL_1", prob_dict.get("Real", 0.0))

            # Apply temperature scaling (1.5) to mirror your original setup
            logits = np.log(np.array([fake_prob, real_prob]) + 1e-12) / 1.5
            exp_logits = np.exp(logits - np.max(logits))
            scaled_probs = exp_logits / np.sum(exp_logits)

            all_probs.append(scaled_probs)

        except Exception:
            # Fallback uniform probability if API call fails for a single perturbation
            all_probs.append(np.array([0.5, 0.5]))

    return np.array(all_probs)


@app.get("/")
def home():
    return {"status": "Fake News Detection & XAI API is live"}


@app.post("/analyze")
def analyze_article(request: ArticleRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # 1. Get Prediction & Continuous Probabilities
    probs = predictor([request.text])[0]
    pred_class = int(np.argmax(probs))

    label = class_names[pred_class]
    confidence = round(float(probs[pred_class]) * 100, 2)

    # 2. Get LIME Explanation
    # Keep num_samples low (e.g., 25-50) so LIME runs quickly over API calls
    exp = explainer.explain_instance(
        request.text, predictor, num_features=8, num_samples=30
    )

    # Scale up LIME scores so they display cleanly on the frontend UI
    feature_weights = [
        {
            "word": word,
            "score": round(score * 1000, 4),
            "weight": round(score * 1000, 4),
        }
        for word, score in exp.as_list()
    ]

    return {
        "text": request.text,
        "prediction": label,
        "confidence": confidence,
        "explanation": feature_weights,
    }