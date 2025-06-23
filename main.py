from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
import traceback

# 🔐 Load environment variables from .env
load_dotenv()

# ✅ Read OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# ✅ Set up OpenAI client
openai.api_key = openai_api_key

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend access

# ✅ Health check route
@app.route("/")
def home():
    return jsonify({"status": "🟢 EduSpark AI Backend is live."})

# ✅ AI Query Endpoint
@app.route("/api/ask-chatgpt", methods=["POST"])
def ask_chatgpt():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "⚠️ Prompt is missing."}), 400

        print(f"[🧠] Received prompt: {prompt}")

        # 📡 Send request to ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if your key supports it
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95
        )

        reply = response.choices[0].message["content"].strip()
        print(f"[✅] ChatGPT replied: {reply[:100]}...")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[🔥] Server error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500

# ✅ Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting EduSpark AI on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
