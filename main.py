from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ NVIDIA Nemotron API Key
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("❌ NVIDIA_API_KEY not set in .env file")

# ✅ Initialize NVIDIA-compatible OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# ✅ Flask app and CORS
app = Flask(__name__)
CORS(app)

# ✅ Root health check
@app.route("/")
def root():
    return jsonify({"status": "🟢 EduSpark AI backend is live."})

# ✅ AI query endpoint
@app.route("/api/ask-nemotron", methods=["POST"])
def ask_nemotron():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is missing."}), 400

        print(f"[🧠] Prompt: {prompt}")

        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024
        )

        reply = completion.choices[0].message.content
        print(f"[✅] Reply: {reply[:100]}...")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[🔥] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
