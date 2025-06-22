from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

NIM_API_KEY = os.environ.get("NIM_API_KEY")
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1"
MODEL_ID = "meta/llama-3-1-nemotron-ultra-253b-v1"

HEADERS = {
    "Authorization": f"Bearer {NIM_API_KEY}",
    "Content-Type": "application/json"
}

# 🔁 Shared function to call NIM API
def call_nemotron(prompt, system_message="You are EduSpark AI. Respond clearly and helpfully."):
    url = f"{NIM_ENDPOINT}/llm/message/{MODEL_ID}"
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    response = requests.post(url, headers=HEADERS, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(response.text)

# ✅ EduSpark Chat
@app.route("/api/ask-nemotron", methods=["POST"])
def ask_nemotron():
    data = request.get_json()
    prompt = data.get("prompt", "")
    try:
        reply = call_nemotron(prompt)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📝 Essay Generator
@app.route("/api/essay-generator", methods=["POST"])
def essay_generator():
    data = request.get_json()
    topic = data.get("topic", "")
    try:
        prompt = f"Write a detailed, well-structured essay on the topic: {topic}. Use paragraphs and clear points."
        reply = call_nemotron(prompt, "You are an academic essay generator.")
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📄 Notes Summarizer
@app.route("/api/notes-summarizer", methods=["POST"])
def notes_summarizer():
    data = request.get_json()
    content = data.get("content", "")
    try:
        prompt = f"Summarize the following text into neat, bullet-pointed study notes:\n\n{content}"
        reply = call_nemotron(prompt, "You are a helpful summarizer.")
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🌐 Translator
@app.route("/api/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("to", "Hindi")
    try:
        prompt = f"Translate this into {lang}:\n{text}"
        reply = call_nemotron(prompt, f"You are a professional translator to {lang}.")
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 💻 Code Explainer
@app.route("/api/explain-code", methods=["POST"])
def explain_code():
    data = request.get_json()
    code = data.get("code", "")
    try:
        prompt = f"Explain this code line by line in simple terms:\n{code}"
        reply = call_nemotron(prompt, "You are a code explainer for beginners.")
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 📄 PDF Assistant (future)
@app.route("/api/ask-pdf", methods=["POST"])
def ask_pdf():
    data = request.get_json()
    question = data.get("question", "")
    pdf_text = data.get("text", "")
    try:
        prompt = f"Based on this PDF content:\n\n{pdf_text}\n\nAnswer this: {question}"
        reply = call_nemotron(prompt, "You are a PDF assistant. Answer based only on the content.")
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Root Ping Test
@app.route("/", methods=["GET"])
def home():
    return "EduSpark AI Backend is running!"

