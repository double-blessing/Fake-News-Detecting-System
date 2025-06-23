import os
import sys
from datetime import datetime
from flask import Flask, render_template, request
from typing import Dict
import json # Import json for loading metrics

# Extend sys path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import verifier, analytics

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
LOG_DIR = os.path.join(ROOT_DIR, "log")
MODEL_DIR = os.path.join(ROOT_DIR, "model") # Define MODEL_DIR
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "predictions.log")

# Flask app initialization
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load model accuracy once when the app starts
app_model_accuracy = "N/A" # Default value
try:
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            app_model_accuracy = f"{metrics.get('model_accuracy', 0.0) * 100:.2f}%"
except Exception as e:
    print(f"[ERROR] Failed to load model_metrics.json: {e}")


def log_prediction(text: str, result: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {result} | {text.strip()}\n")

@app.route("/")
def index():
    # Pass app_model_accuracy to the index page directly
    return render_template("index.html", model_accuracy=app_model_accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    result_data = verifier.verify_news(news)

    log_prediction(news, result_data["final_verdict"])

    # Pass both model_accuracy and confidence to the template
    return render_template(
        "index.html",
        news=news,
        result=f"{result_data['final_verdict']} — {result_data['reason']}",
        entity_verification=result_data.get("entity_verification", []),
        fact_check_links=result_data.get("fact_check_links", {}),
        model_accuracy=app_model_accuracy, # Pass model accuracy here
        confidence=result_data.get('ml_confidence', 0.0), # Pass ML confidence here
        prediction_details=result_data
    )

@app.route("/insights")
def insights():
    stats = analytics.parse_logs()
    return render_template("insights.html", stats=stats)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)