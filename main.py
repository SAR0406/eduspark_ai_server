from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os

# Load .env variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# NVIDIA API Config
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Flask App
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "✅ EduSpark AI Server is running",
        "model": MODEL_NAME,
        "streaming": False
    })


# =========================================
# ✨ /api/send-message — EduSpark AI Chat
# =========================================
@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"response": "⚠️ No prompt provided."}), 400

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are EduSpark, a helpful and concise AI tutor. "
                        "Answer like a knowledgeable but human teacher. Avoid long or overly complex replies."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            top_p=0.9,
            max_tokens=512,
            frequency_penalty=0.2,
            presence_penalty=0,
            stream=False
        )

        message = response.choices[0].message.content.strip()
        logging.info(f"[Chat] Prompt: {prompt}\nReply: {message}")
        return jsonify({"response": message})

    except Exception as e:
        logging.error(f"[Chat Error]: {e}")
        return jsonify({"response": f"⚠️ Server error: {str(e)}"}), 500


# =========================================
# 💻 /api/code — Code Assistant Endpoint
# =========================================
@app.route("/api/code", methods=["POST"])
def code_assistant():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"response": "⚠️ No prompt provided."}), 400

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are EduSpark Code Assistant. Help users write code, debug, explain functions, and improve their code. "
                        "Always give clean, well-commented, and optimized code with explanations."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=16384,
            frequency_penalty=2.0,
            presence_penalty=2.0,
            stream=False
        )

        message = response.choices[0].message.content.strip()
        logging.info(f"[Code] Prompt: {prompt}\nCode Reply: {message}")
        return jsonify({"response": message})

    except Exception as e:
        logging.error(f"[Code Error]: {e}")
        return jsonify({"response": f"⚠️ Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
