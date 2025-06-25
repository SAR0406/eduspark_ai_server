from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load API key securely
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
    return jsonify({"message": "✅ EduSpark AI is running with Nemotron Ultra 253B"})

@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)

        if not prompt:
            return jsonify({"response": "No prompt provided"}), 400

        def generate_stream():
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        if stream:
            return Response(generate_stream(), mimetype="text/plain")

        # Fallback: stream=False, simple response
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096
        )
        message = response.choices[0].message.content
        return jsonify({"response": message})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"response": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
