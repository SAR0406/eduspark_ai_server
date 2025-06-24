from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# ✅ Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Health check
@app.route("/")
def home():
    return jsonify({"status": "🟢 EduSpark Gemini AI backend is running."})

# ✅ AI prompt handler
@app.route("/api/ask-gemini", methods=["POST"])
def ask_gemini():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        print(f"[🧠] Prompt: {prompt}")

        response = model.generate_content(prompt)
        reply = response.text

        print(f"[✅] Gemini Reply: {reply[:100]}...")
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[🔥] Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
