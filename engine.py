import streamlit as st
import re
import os
import logging

# 1. SILENCE TERMINAL NOISE (Stops the "Accessing path" wall of text)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

class SentimentEngine:
    def __init__(self):
        # API Client for Cloud Inference
        from huggingface_hub import InferenceClient
        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        
        try:
            token = st.secrets["HF_TOKEN"]
            self.client = InferenceClient(model=model_id, token=token)
        except Exception as e:
            st.error("HF_TOKEN missing or invalid in secrets.toml")
            self.client = None

        # Local Explainer placeholders (Lazy Loading)
        self.explainer = None

    def _load_explainer(self):
        """Loads heavy libraries and local model ONLY when needed."""
        # Moving heavy imports here prevents the UI from freezing at startup
        from transformers import pipeline
        import shap
        import torch
        
        if self.explainer is None:
            # We use the absolute smallest model (DistilBERT) for 8GB RAM
            local_pipe = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None 
            )
            self.explainer = shap.Explainer(local_pipe)

    def detect_script(self, text):
        """Unicode-based script identification."""
        if re.search(r'[\u0980-\u09FF]', text):
            return "Bengali", "Native Unicode"
        elif re.search(r'[\u0900-\u097F]', text):
            return "Hindi", "Native Unicode"
        else:
            return "English", "Standard ASCII"

    def analyze(self, text):
        """Performs multilingual sentiment analysis via Cloud API."""
        if not self.client:
            return {"label": "Error", "score": 0.0}, "Unknown", "Client Not Initialized"

        lang, script = self.detect_script(text)

        try:
            # Uses Hugging Face router to find the model
            response = self.client.text_classification(text)
            # Find the result with the highest confidence
            result = max(response, key=lambda x: x['score'])
            return result, lang, script
        except Exception as e:
            # Catching Router/API errors gracefully
            return {"label": "Error", "score": 0.0}, lang, str(e)

    def get_explanation(self, text):
        """Generates SHAP values for word-level feature importance."""
        try:
            self._load_explainer()
            if self.explainer is None:
                return None
            
            # max_evals=100 speeds up calculation on 8GB RAM CPU
            shap_values = self.explainer([text], max_evals=100)
            return shap_values
        except Exception as e:
            # This helps debug the SHAP failure in the app.py UI
            raise e