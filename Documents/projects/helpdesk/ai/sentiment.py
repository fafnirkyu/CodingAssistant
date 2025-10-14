from transformers import pipeline
import threading

# Load the sentiment model once
_lock = threading.Lock()
_sentiment_analyzer = None

# Lazy load the sentiment analyzer
def get_sentiment_analyzer():
    global _sentiment_analyzer
    with _lock:
        if _sentiment_analyzer is None:
            _sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment_analyzer

# Detect sentiment of the text
def detect_sentiment(text: str) -> str:
    if not text:
        return "NEUTRAL"
    
    analyzer = get_sentiment_analyzer()
    result = analyzer(text[:512])[0]  # limit to 512 tokens
    
    label = result["label"].upper()
    score = result["score"]
    
    if score < 0.6:
        return "NEUTRAL"
    return label
