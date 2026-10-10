from flask import Flask, request, jsonify, send_from_directory
import joblib

# Trained pipeline exported from Colab (TF-IDF + Logistic Regression)
# Label mapping from the notebook: 1 = True (verified), 0 = False (misinformation)
MODEL_PATH = "healthguard_model.joblib"
MAX_CHARS = 3000

app = Flask(__name__, static_folder="static", static_url_path="")
model = joblib.load(MODEL_PATH)


def explain(claim, top=5):
    """Break the prediction into the words and phrases in this claim that pushed it toward True or False."""
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    x = vec.transform([claim]).tocoo()
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    items = [(names[j], float(val) * float(coef[j])) for j, val in zip(x.col, x.data) if val > 0]
    toward_true = sorted([i for i in items if i[1] > 0], key=lambda t: -t[1])[:top]
    toward_false = sorted([i for i in items if i[1] < 0], key=lambda t: t[1])[:top]
    return {
        "known_terms": len(items),
        "toward_true": [{"word": w, "weight": round(c, 3)} for w, c in toward_true],
        "toward_false": [{"word": w, "weight": round(abs(c), 3)} for w, c in toward_false],
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    claim = str(data.get("claim", "")).strip()

    if not claim:
        return jsonify(error="Claim is required."), 400
    if len(claim) > MAX_CHARS:
        return jsonify(error="Claim is too long."), 400

    probabilities = model.predict_proba([claim])[0]
    best = probabilities.argmax()
    prediction = int(model.classes_[best])      # 1 = True, 0 = False
    confidence = float(probabilities[best])     # probability of the predicted class

    return jsonify(prediction=prediction, confidence=round(confidence, 3), explanation=explain(claim))


if __name__ == "__main__":
    app.run(port=5000, debug=False)
