import threading
from typing import Optional
import os
import requests
import json
import re
import time
from ollama import chat
from tests.debug_logger import log_ai_request, log_ai_response


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUTF8"] = "1"

_MODEL_NAME = "llama3.2:3b"
_FALLBACK_MODEL = "llama3.1:8b"



# CORE CLIENT (Structured responses via Ollama)

class OptimizedOllamaClientImpl:
    def __init__(self, model_name=_MODEL_NAME):
        self.model_name = self._get_available_model(model_name)
        print(f"🔄 Using Ollama with {self.model_name}...")

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama connection successful")
            else:
                print("Ollama connection issue")
        except Exception as e:
            print(f"Could not connect to Ollama: {e}")

    def _get_available_model(self, preferred_model: str) -> str:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m["name"] for m in response.json().get("models", [])]
                if preferred_model in available_models:
                    print(f"Using preferred model: {preferred_model}")
                    return preferred_model
                elif _FALLBACK_MODEL in available_models:
                    print(f"{preferred_model} not available, using {_FALLBACK_MODEL}")
                    return _FALLBACK_MODEL
                elif available_models:
                    fallback = available_models[0]
                    print(f"Using available model: {fallback}")
                    return fallback
            print("No models available, using preferred anyway")
            return preferred_model
        except Exception:
            print("Could not check available models, using preferred")
            return preferred_model

    def _extract_json(self, text: str) -> dict:
        if not text:
            return {}
        text = text.strip()
        json_patterns = [
            r"\{[^{}]*(\{[^{}]*\}[^{}]*)*\}",
            r"\{[^}]+}",
        ]
        for pattern in json_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    json_str = match.group(0)
                    json_str = re.sub(r",\s*}", "}", json_str)
                    json_str = re.sub(r",\s*]", "]", json_str)
                    result = json.loads(json_str)
                    if "category" in result:
                        return result
                except json.JSONDecodeError:
                    continue
        return self._infer_from_text(text)

    def _infer_from_text(self, text: str) -> dict:
        text_lower = text.lower()
        category_keywords = {
            "ACCOUNT": ["account", "login", "password", "email"],
            "ORDER": ["order", "delivery", "shipping", "package"],
            "BILLING": ["billing", "charge", "payment", "refund"],
            "SUBSCRIPTION": ["subscription", "cancel", "renew"],
            "TECHNICAL": ["technical", "crash", "error", "bug"],
        }
        found = "OTHER"
        for cat, keys in category_keywords.items():
            if any(k in text_lower for k in keys):
                found = cat
                break
        return {
            "category": found,
            "subcategory": "general",
            "summary": text[:100] + "...",
            "response": "Thank you for your message. We'll assist you shortly.",
        }

    def generate_json(self, prompt: str) -> dict:
        start = time.time()
        try:
            system_prompt = """Return JSON with: category, subcategory, summary, response.
Categories: ACCOUNT, ORDER, BILLING, TECHNICAL, SUBSCRIPTION, OTHER"""
            response = chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 150,
                    "timeout": 45000,
                },
            )
            elapsed = time.time() - start
            print(f"Model response: {elapsed:.2f}s")
            raw_output = response["message"]["content"]
            result = self._extract_json(raw_output)
            return self._validate_result(result, prompt)
        except Exception as e:
            print(f"Model generation failed: {e}")
            return self._create_fallback_response(prompt)

    def _validate_result(self, result: dict, prompt: str) -> dict:
        valid = {"ACCOUNT", "ORDER", "BILLING", "TECHNICAL", "SUBSCRIPTION", "OTHER"}
        cat = result.get("category", "").upper()
        if cat not in valid:
            cat = self._infer_category_from_text(prompt)
        sub = result.get("subcategory", "general") or "general"
        sum_ = result.get("summary", f"User reported: {prompt[:80]}...")
        res = result.get("response", "Thank you for your message.")
        return {"category": cat, "subcategory": sub, "summary": sum_, "response": res}

    def _infer_category_from_text(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["login", "password", "account", "email"]):
            return "ACCOUNT"
        if any(w in text for w in ["order", "delivery", "shipping", "package"]):
            return "ORDER"
        if any(w in text for w in ["charge", "payment", "bill", "refund"]):
            return "BILLING"
        if any(w in text for w in ["subscription", "cancel", "renew"]):
            return "SUBSCRIPTION"
        if any(w in text for w in ["crash", "error", "technical", "slow"]):
            return "TECHNICAL"
        return "OTHER"

    def _create_fallback_response(self, prompt: str) -> dict:
        cat = self._infer_category_from_text(prompt)
        responses = {
            "ACCOUNT": "I understand you're having account issues. Let me help you resolve this.",
            "ORDER": "I see you have an order-related concern. Let me look into this for you.",
            "BILLING": "I understand your billing concern. Let me check this for you.",
            "SUBSCRIPTION": "I can help with your subscription question.",
            "TECHNICAL": "I understand you're experiencing technical difficulties.",
            "OTHER": "Thank you for your message. I'll help you with this.",
        }
        return {
            "category": cat,
            "subcategory": "general",
            "summary": f"User reported: {prompt[:80]}...",
            "response": responses.get(cat, "Thank you for your message."),
        }



# Singleton Accessor

_client_lock = threading.Lock()
_client_instance: Optional[OptimizedOllamaClientImpl] = None

def get_hf_client() -> OptimizedOllamaClientImpl:
    global _client_instance
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = OptimizedOllamaClientImpl()
    return _client_instance



# Lightweight generator (for classification only)

def hf_generate(prompt: str, model: str = "llama3.2:3b", max_tokens: int = 200) -> str:
    log_ai_request(prompt, model, max_tokens)
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "max_tokens": max_tokens},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()
        log_ai_response(text, time.time() - start, True)
        return text
    except Exception as e:
        log_ai_response(f"ERROR: {e}", time.time() - start, False)
        return f"ERROR: {e}"


# Context-aware classification (used by rag.py and ai_pipeline)

def classify_with_llm_detailed(message: str) -> dict:
    from ai.rag import retrieve_examples 

    examples = retrieve_examples(message, top_k=3)
    context = "\n".join([f"- {e['instruction']} => {e['response']}" for e in examples])

    prompt = f"""
You are a helpdesk classifier.
Use the following examples as context:
{context}

Now analyze the new ticket and return JSON:
{{
  "primary": "CATEGORY",
  "secondary": ["OTHER_CATEGORY"],
  "confidence": 0.0-1.0,
  "summary": "Short summary",
  "response": "Customer-facing answer"
}}

Ticket: "{message}"
Return only JSON.
"""

    raw = hf_generate(prompt, max_tokens=300)

    try:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            return json.loads(match.group(0))
        else:
            return {
                "primary": "OTHER",
                "confidence": 0.0,
                "summary": message[:120],
                "response": "Thank you for your message.",
            }
    except Exception:
        return {
            "primary": "OTHER",
            "confidence": 0.0,
            "summary": message[:120],
            "response": "Thank you for your message.",
        }
