from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API Key securely
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

app = Flask(__name__)
CORS(app)

# ✨ Model selection per use-case
MODEL_MAP = {
    "chat": "mistral-nemo-12b-instruct",
    "code": "llama-3.1-nemotron-70b-instruct",
    "study": "phi-3-mini-4k-instruct",
    "langchat": "llama-3.3-nemotron-super-49b-v1",
    "research": "llama-3-70b-instruct",
    "vision": "llama3-2-llb-vision-instruct"
}

# 🧠 Common system prompt for consistency
SYSTEM_PROMPT = (
    "You are EduSpark 🌟 — a super-intelligent, helpful AI assistant. You are EduSpark AI, a brilliant yet friendly teacher. Answer clearly, briefly, and helpfully like a human expert. Use bullet points or line breaks if needed, and include emojis to enhance clarity 😊. Avoid using asterisks (*) or markdown. Your tone should be positive, direct, and easy to understand — just like a real classroom!"
    "Respond clearly, accurately, and concisely. Add emojis when relevant. "
    "Avoid using * for formatting. Keep answers engaging and informative."
)

def generate_response(prompt, model_key, max_tokens=4096, temperature=0.7, top_p=0.95,
                      freq_penalty=0, presence_penalty=0):
    try:
        model_name = MODEL_MAP.get(model_key)
        if not model_name:
            return {"response": f"❌ Model key '{model_key}' not found."}, 400

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=presence_penalty,
            stream=False
        )
        return {"response": response.choices[0].message.content}, 200

    except Exception as e:
        print("❌ Error:", e)
        return {"response": f"Server error: {str(e)}"}, 500

# 🧪 Route test
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "✅ EduSpark AI API is running with NVIDIA NIM 🚀"})

# 🤖 AI Chat
@app.route("/api/chat", methods=["POST"])
def chat():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(prompt, "chat"))

# 💻 Code Generation
@app.route("/api/code", methods=["POST"])
def code():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(
        prompt, "code",
        max_tokens=16384,
        temperature=1,
        top_p=1,
        freq_penalty=2,
        presence_penalty=2
    ))

# 📚 Academic Assistant
@app.route("/api/study", methods=["POST"])
def study():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(prompt, "study"))

# 🌍 Multilingual Assistant
@app.route("/api/langchat", methods=["POST"])
def langchat():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(prompt, "langchat"))

# 📊 Research Expert (large context)
@app.route("/api/research", methods=["POST"])
def research():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(prompt, "research", max_tokens=8192))

# 🖼️ Future: Vision/Image QA
@app.route("/api/vision", methods=["POST"])
def vision():
    prompt = request.json.get("prompt", "")
    return jsonify(*generate_response(prompt, "vision"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
