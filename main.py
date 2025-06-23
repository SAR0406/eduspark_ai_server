from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ OpenAI API Key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not set in .env file")

openai.api_key = openai_api_key  # Set key globally

# ✅ Flask app setup
app = Flask(__name__)
CORS(app)

# ✅ Root route to check health
@app.route("/")
def root():
    return jsonify({"status": "🟢 EduSpark ChatGPT API is live."})

# ✅ AI endpoint
@app.route("/api/ask-chatgpt", methods=["POST"])
def ask_chatgpt():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is missing."}), 400

        print(f"[🧠] Prompt: {prompt}")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )

        reply = response['choices'][0]['message']['content']
        print(f"[✅] Reply: {reply[:100]}...")

        return jsonify({"reply": reply})

    except Exception as e:
        print(f"[🔥] Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
