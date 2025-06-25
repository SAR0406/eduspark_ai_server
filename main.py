from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
import logging

# Import Gemini SDK
import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentConfig,
    ThinkingConfig,
)

# =========================
# 🔐 Load environment
# =========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY is missing in your .env file.")

# =========================
# 🧠 Initialize Gemini
# =========================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

# =========================
# 🚀 Setup Flask
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# 💬 Session Store (In-Memory)
# =========================
chat_sessions: dict[str, genai.ChatSession] = {}

# =========================
# 📌 Routes
# =========================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "✅ Gemini Chat API is running!"})


@app.route("/api/start-session", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    chat = model.start_chat(history=[])
    chat_sessions[session_id] = chat

    return jsonify({
        "message": "🟢 Session started",
        "session_id": session_id
    })


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        prompt = data.get("prompt", "").strip()
        budget = int(data.get("budget", -1))  # -1 for dynamic thinking
        include_thoughts = bool(data.get("thoughts", True))

        if not prompt:
            return jsonify({"error": "❌ Prompt is required."}), 400

        if session_id not in chat_sessions:
            return jsonify({"error": "❌ Invalid session ID."}), 400

        config = GenerateContentConfig(
            thinking_config=ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=include_thoughts
            )
        )

        chat = chat_sessions[session_id]
        response = chat.send_message(prompt, config=config)

        reply = ""
        thoughts = []

        for part in response.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thoughts.append(part.text)
            else:
                reply += part.text

        # Collect full chat history
        history = [
            {"role": msg.role, "text": msg.parts[0].text}
            for msg in chat.get_history()
        ]

        return jsonify({
            "reply": reply.strip(),
            "thoughts": thoughts,
            "history": history
        })

    except Exception as e:
        logging.exception("Error processing message")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/end-session", methods=["POST"])
def end_session():
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if session_id in chat_sessions:
            del chat_sessions[session_id]
            return jsonify({"message": "🛑 Session ended."})
        else:
            return jsonify({"error": "❌ Invalid session ID."}), 400

    except Exception as e:
        logging.exception("Error ending session")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# =========================
# 🚀 Run Server
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 Server running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port)
