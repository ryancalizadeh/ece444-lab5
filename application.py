from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

@application.route("/predict", methods=['POST'])
def predict():
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    try:
        query = request.json.get('query')
        if not query:
            return json.jsonify({"error": "No query provided"}), 400

        prediction = loaded_model.predict(vectorizer.transform([query]))[0]
        return json.jsonify({"prediction": prediction})
    except Exception as e:
        return json.jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    application.run(port=5000, debug=True)