from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

NIM_API_KEY = os.environ.get("NIM_API_KEY")
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1"
MODEL_ID = "meta/llama-3-1-nemotron-ultra-253b-v1"

@app.route("/api/ask-nemotron", methods=["POST"])
def ask_nemotron():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        headers = {
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": [
                {"role": "system", "content": "You are EduSpark AI. Respond clearly and helpfully."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

        url = f"{NIM_ENDPOINT}/llm/message/{MODEL_ID}"
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            return jsonify({"reply": data["choices"][0]["message"]["content"]})
        else:
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
