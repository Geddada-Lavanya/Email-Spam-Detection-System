from flask import Flask, render_template, request, jsonify
from model import predict_spam
import pandas as pd

app = Flask(__name__)

# Load dataset once
data = pd.read_csv("spam_sms.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    message = data_json.get("message")
    result = predict_spam(message)
    return jsonify({"result": result})

@app.route("/dataset")
def dataset():
    # Send first 50 rows
    preview = data.head(50).to_dict(orient="records")
    return jsonify(preview)

if __name__ == "__main__":
    app.run(debug=True)