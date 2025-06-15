from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    text = request.json.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)[0]

    labels = model.config.id2label
    result = [
        {"label": labels[i], "score": round(probs[i].item(), 4)}
        for i in range(len(probs))
    ]
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
