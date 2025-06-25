from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

# ✅ Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY not found in environment variables.")

# ✅ Initialize Gemini client
genai_client = genai.Client(api_key=api_key)
model = genai_client.models.get("gemini-2.5-pro")  # or "gemini-2.5-flash"

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ Gemini AI server is running."})


@app.route("/api/ask-gemini", methods=["POST"])
def ask_gemini():
    try:
        # ✅ Parse request body
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        use_thinking = data.get("thinking", True)
        budget = data.get("budget", -1)  # -1 = dynamic, 0 = disable, else int
        include_thoughts = data.get("thoughts", False)

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        print(f"📩 Prompt: {prompt}")
        print(f"🧠 Thinking: {use_thinking} | Budget: {budget} | Thoughts: {include_thoughts}")

        # ✅ Configure thinking
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=include_thoughts
            )
        )

        # ✅ Generate AI content
        response = model.generate_content(
            contents=prompt,
            config=config
        )

        output = ""
        thoughts = []

        # ✅ Extract output and thoughts
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thoughts.append(part.text)
            else:
                output += part.text

        result = {
            "reply": output.strip(),
            "thoughts": thoughts,
            "tokens_used": {
                "output": response.usage_metadata.candidates_token_count,
                "thinking": response.usage_metadata.thoughts_token_count
            }
        }

        print(f"✅ AI Response: {output.strip()[:100]}...")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
