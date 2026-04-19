"""House rate (price) prediction web app using a trained decision tree."""
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
ARTIFACT_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__)


class _ArtifactStore:
    _data = None

    @classmethod
    def get(cls):
        if cls._data is None:
            if not ARTIFACT_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found at {ARTIFACT_PATH}. Run: python train.py"
                )
            cls._data = joblib.load(ARTIFACT_PATH)
        return cls._data


def load_artifact():
    return _ArtifactStore.get()


@app.route("/")
def index():
    artifact = load_artifact()
    return render_template(
        "index.html",
        feature_names=artifact["feature_names"],
        target_description=artifact["target_description"],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        artifact = load_artifact()
        model = artifact["model"]
        names = artifact["feature_names"]
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    body = request.get_json(silent=True) or {}
    features = body.get("features")
    if not isinstance(features, dict):
        return jsonify({"error": "Send JSON: { \"features\": { \"MedInc\": 3.5, ... } }"}), 400

    vec = []
    missing = []
    for name in names:
        if name not in features:
            missing.append(name)
            continue
        try:
            vec.append(float(features[name]))
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid number for feature: {name}"}), 400

    if missing:
        return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400

    x = np.array([vec], dtype=float)
    pred = float(model.predict(x)[0])
    return jsonify(
        {
            "prediction_hundred_k": pred,
            "prediction_usd_approx": round(pred * 100_000, 2),
            "unit_note": artifact["target_description"],
        }
    )


if __name__ == "__main__":
    load_artifact()
    app.run(host="127.0.0.1", port=5000, debug=True)
