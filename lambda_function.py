import json
import boto3
import base64
import os
import io
from PIL import Image

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
            print("Received event:", event)

            if "body" not in event or not event["body"]:
                return {"statusCode": 400, "body": json.dumps({"error": "No body in request"})}

            # Decode body if Base64 encoded
            body_bytes = base64.b64decode(event["body"]) if event["isBase64Encoded"] else event["body"].encode()

            # Try to extract image bytes manually
            try:
                delimiter = b"\r\n"
                parts = body_bytes.split(delimiter)
                
                # Locate the file content (skipping headers)
                file_start = next(i for i in range(len(parts)) if b"Content-Type" in parts[i]) + 2
                file_content = b"\r\n".join(parts[file_start:-2])  # Ignore last boundary
                
                if not file_content:
                    return {"statusCode": 400, "body": json.dumps({"error": "Empty file received"})}
                
            except Exception as e:
                return {"statusCode": 400, "body": json.dumps({"error": f"Malformed form-data: {str(e)}"})}

            # Process the extracted image bytes
            if not is_allowed_file("image.jpg"):  # Simulated filename check
                return {"statusCode": 400, "body": json.dumps({"error": "Invalid file format"})}

            processed_image = preprocess_image(file_content)

            # Get SageMaker endpoint
            SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT")
            if not SAGEMAKER_ENDPOINT:
                return {"statusCode": 500, "body": json.dumps({"error": "SAGEMAKER_ENDPOINT not set"})}

            print("SAGEMAKER_ENDPOINT:", SAGEMAKER_ENDPOINT)

            # Call SageMaker endpoint
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps({"image": processed_image}),
            )

            # Read response
            response_body = response["Body"].read().decode()
            if not response_body:
                return {"statusCode": 500, "body": json.dumps({"error": "Empty response from SageMaker"})}

            prediction = json.loads(response_body)

            print("Prediction:", prediction)

            return {
                "statusCode": 200,
                "body": json.dumps({"prediction": prediction}),
                "headers": {"Content-Type": "application/json"}
            }

        else:
            return {"statusCode": 404, "body": "Not Found"}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
