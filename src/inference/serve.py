import io
import json
import os
import logging

import joblib
import pandas as pd
from flask import Flask, Response, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = "/opt/ml/model/model.joblib"
model = None


def load_model():
    global model
    if model is None:
        logger.info("Loading model from %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/ping", methods=["GET"])
def ping():
    try:
        load_model()
        return Response(response="", status=200)
    except Exception as exc:
        logger.exception("Ping failed")
        return Response(response=str(exc), status=500)


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        reg = load_model()
        content_type = request.content_type or ""

        if "text/csv" in content_type:
            data = request.data.decode("utf-8")
            d = pd.read_csv(io.StringIO(data))

        elif "application/json" in content_type:
            payload = request.get_json()

            if isinstance(payload, dict) and "instances" in payload:
                d = pd.DataFrame(payload["instances"])
            elif isinstance(payload, list):
                d = pd.DataFrame(payload)
            elif isinstance(payload, dict):
                d = pd.DataFrame([payload])
            else:
                return Response(
                    response=json.dumps({"error": "Unsupported JSON payload"}),
                    status=400,
                    mimetype="application/json",
                )
        else:
            return Response(
                response=json.dumps(
                    {"error": f"Unsupported content type: {content_type}"}
                ),
                status=415,
                mimetype="application/json",
            )

        logger.info("Received inference request with shape %s", d.shape)

        preds = reg.predict(d)

        return Response(
            response=json.dumps({"predictions": preds.tolist()}),
            status=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logger.exception("Inference failed")
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    logger.info("Starting inference server on port %s", port)
    app.run(host="0.0.0.0", port=port)
