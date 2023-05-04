import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify

model = tf.keras.models.load_model("nn.h5")

app = Flask(__name__)


def transform_image(pillow_image):
    data = np.array(pillow_image)
    data = data / 255.0
    data = data[np.newaxis, ..., np.newaxis]
    data = tf.image.resize(data, [28, 28])
    return data


def predict(x):
    predictions = model(x)
    pred = np.argmax(predictions)
    return pred


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_image = Image.open(io.BytesIO(image_bytes)).convert("L")
            tensor = transform_image(pillow_image)
            prediction = predict(tensor)
            data = {"prediction": int(prediction)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
