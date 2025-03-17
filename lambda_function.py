import json
import boto3
import base64
import os
from PIL import Image
import io

# AWS SageMaker client
sagemaker_runtime = boto3.client("sagemaker-runtime")

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

def lambda_handler(event, context):
    """ AWS Lambda handler for HTML rendering and image prediction """
    try:
        path = event.get("rawPath", "/")

        # Serve the HTML page
        if path == "/":
            with open("templates/index.html", "r") as file:
                html_content = file.read()
            return {
                "statusCode": 200,
                "body": html_content,
                "headers": {"Content-Type": "text/html"}
            }

        # Handle image prediction
        elif path == "/predict":
            body = json.loads(event["body"])
            if "image" not in body:
                return {"statusCode": 400, "body": json.dumps({"error": "No image provided"})}

            image_bytes = base64.b64decode(body["image"])

            if not is_allowed_file("image.jpg"):  # Simulating a filename
                return {"statusCode": 400, "body": json.dumps({"error": "Invalid file format"})}

            processed_image = preprocess_image(image_bytes)

            SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT")
            if not SAGEMAKER_ENDPOINT:
                return {"statusCode": 500, "body": json.dumps({"error": "SAGEMAKER_ENDPOINT not set"})}

            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps({"image": processed_image}),
            )

            prediction = json.loads(response["Body"].read().decode())

            return {
                "statusCode": 200,
                "body": json.dumps({"prediction": prediction}),
                "headers": {"Content-Type": "application/json"}
            }

        else:
            return {"statusCode": 404, "body": "Not Found"}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
