from flask import Blueprint, request, jsonify
import requests

gen_llm_bp = Blueprint('gllm', __name__)

@gen_llm_bp.route('/gllm', methods=['GET','POST'])
def generate_text():
    try:
        data = request.get_json()
    except:
        data = dict()
        data['prompt'] = request.args.get('prompt')
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400

    # Call DeepSeek API (same as your earlier code)
    headers = {"Authorization": f"Bearer {'sk-a1ca5a1afa65433bba1ad7d33381fc6c'}"}
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": data['prompt']}],
            "max_tokens": data.get('max_length', 100),
            "temperature": data.get('temperature', 0.7),
        },
        headers=headers
    )

    return jsonify({"response": response.json()['choices'][0]['message']['content']})
