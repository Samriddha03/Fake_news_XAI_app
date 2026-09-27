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

MODEL_PATH = "Samriddha03/fake-news-bert-xai"
HF_TOKEN = os.getenv("HF_TOKEN", "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")

client = InferenceClient(api_key=HF_TOKEN)

class_names = ["Fake", "Real"]
explainer = LimeTextExplainer(class_names=class_names)

class ArticleRequest(BaseModel):
    text: str


def predictor(texts):
    """Predict function for LIME.
    Queries Hugging Face Inference API and maps raw outputs to continuous probability scores.
    """
    if isinstance(texts, str):
        texts = [texts]

    all_probs = []

    for text in texts:
        try:
            results = client.text_classification(text, model=MODEL_PATH)
            
            # Map predictions dynamically by index or label key
            fake_score = 0.0
            real_score = 0.0

            for res in results:
                label = str(res.get("label", "")).upper()
                score = res.get("score", 0.0)

                if label in ["LABEL_0", "FAKE", "0"]:
                    fake_score = score
                elif label in ["LABEL_1", "REAL", "1"]:
                    real_score = score

            # Normalize probabilities if sum is off
            total = fake_score + real_score
            if total > 0:
                fake_score /= total
                real_score /= total
            else:
                fake_score, real_score = 0.5, 0.5

            all_probs.append([fake_score, real_score])

        except Exception:
            all_probs.append([0.5, 0.5])

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
    exp = explainer.explain_instance(
        request.text, 
        predictor, 
        num_features=8, 
        num_samples=25
    )

    feature_weights = [
        {
            "word": word,
            "score": round(score * 100, 4),
            "weight": round(score * 100, 4),
        }
        for word, score in exp.as_list()
    ]

    return {
        "text": request.text,
        "prediction": label,
        "confidence": confidence,
        "explanation": feature_weights,
    }