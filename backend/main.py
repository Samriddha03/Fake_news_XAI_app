from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np

app = FastAPI(title="Fake News Detection & XAI API")

# Enable CORS so your React frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved model & tokenizer directly for temperature scaling
MODEL_PATH = "Samriddha03/fake-news-bert-xai"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class_names = ["Fake", "Real"]
explainer = LimeTextExplainer(class_names=class_names)

class ArticleRequest(BaseModel):
    text: str

def predictor(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Lower the temperature slightly to allow LIME perturbations to produce distinct gradients
        temperature = 1.5 
        scaled_logits = outputs.logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
    return probs.cpu().numpy()

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
        num_samples=500  # Increased sample count for higher statistical accuracy
    )
    
    # Scale up LIME scores so they display as clean values on the frontend UI
    feature_weights = [
        {
            "word": word, 
            "score": round(score * 1000, 4),  # e.g., 0.000004 becomes +0.0040
            "weight": round(score * 1000, 4)
        } 
        for word, score in exp.as_list()
    ]

    return {
        "text": request.text,
        "prediction": label,
        "confidence": confidence,
        "explanation": feature_weights
    }