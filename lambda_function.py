import json
import boto3
import base64
import os
from PIL import Image
import io
from flask import Flask, request, render_template
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# AWS SageMaker client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """ Convert image bytes into a Base64-encoded string for SageMaker inference """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))  # Resize to match model input
    image = image.convert("RGB")  # Ensure it's RGB
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    image = request.files["image"]
    if not is_allowed_file(image.filename):
        return {"error": "Invalid file format"}, 400

    image_bytes = image.read()
    processed_image = preprocess_image(image_bytes)

    SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT")
    if not SAGEMAKER_ENDPOINT:
        return {"error": "SAGEMAKER_ENDPOINT not set"}, 500

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"image": processed_image}),
    )

    prediction = json.loads(response["Body"].read().decode())
    return {"prediction": prediction}

if __name__ == "__main__":
    app.run(debug=True)
