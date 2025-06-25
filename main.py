from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in your .env file")

# ✅ Configure Gemini Client
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")  # Use "gemini-1.5-pro" or "gemini-1.0-pro"

# ✅ Flask App Setup
app = Flask(__name__)
CORS(app)

# ✅ In-memory session store (for demo)
chat_sessions = {}


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ Gemini Chat Server is live."})


@app.route("/api/start-session", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = model.start_chat(history=[])
    return jsonify({"session_id": session_id})


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        if session_id not in chat_sessions:
            return jsonify({"error": "Invalid or expired session_id."}), 400

        chat = chat_sessions[session_id]
        response = chat.send_message(prompt)

        reply = response.text.strip() if response.text else "No response"
        history = [
            {"role": msg.role, "text": msg.parts[0].text}
            for msg in chat.history
        ]

        return jsonify({
            "reply": reply,
            "history": history
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/end-session", methods=["POST"])
def end_session():
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if session_id in chat_sessions:
            del chat_sessions[session_id]
            return jsonify({"status": "Session ended."})
        else:
            return jsonify({"error": "Invalid session ID."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
