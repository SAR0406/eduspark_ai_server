from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import uuid

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("\u274c GEMINI_API_KEY is missing in your .env file")

# Initialize Gemini client
genai_client = genai.Client(api_key=api_key)
model = genai_client.models.get("gemini-2.5-pro")  # Or "gemini-2.5-flash"

# Flask setup
app = Flask(__name__)
CORS(app)

# Store chat histories (in-memory, for demo; use DB for production)
chat_sessions = {}


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "\u2705 Gemini Chat Server is active."})


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
        budget = data.get("budget", -1)  # Dynamic by default
        include_thoughts = data.get("thoughts", True)

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        if session_id not in chat_sessions:
            return jsonify({"error": "Invalid or expired session_id."}), 400

        # Configure thinking
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=include_thoughts
            )
        )

        chat = chat_sessions[session_id]
        response = chat.send_message(prompt, config=config)

        output = ""
        thoughts = []

        for part in response.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thoughts.append(part.text)
            else:
                output += part.text

        return jsonify({
            "reply": output.strip(),
            "thoughts": thoughts,
            "history": [
                {"role": msg.role, "text": msg.parts[0].text}
                for msg in chat.get_history()
            ]
        })

    except Exception as e:
        print(f"\u274c Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/end-session", methods=["POST"])
def end_session():
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if session_id and session_id in chat_sessions:
            del chat_sessions[session_id]
            return jsonify({"status": "Session ended."})
        else:
            return jsonify({"error": "Invalid session ID."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
