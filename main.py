from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Securely load NVIDIA API Key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "✅ EduSpark AI is running with Nemotron Ultra 253B (streaming disabled)"})

@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"response": "No prompt provided"}), 400

        # Always use stream=False
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
        {
            "role": "system",
            "content": "You are EduSpark, a helpful, concise, and accurate AI tutor. Answer user questions clearly and directly, like a professional teacher. Keep it short and useful."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False  # Force stream OFF
        )

        message = response.choices[0].message.content
        return jsonify({"response": message})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"response": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
