# vision.py

from openai import AzureOpenAI
from utils import describe_local_image, create_openai_client
import os
from PIL import Image, ImageDraw, ImageFont
import json

# Function to describe the generated image and annotate issues
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_VISION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_VISION")
GPT_DEPLOYMENT = "gpt-4-turbo"
GPT_VERSION = "2024-08-01-preview"

def describe_image(client, image_path, original_complaint, deployment_name):
    """
    Describes an image and identifies key visual elements related to the customer complaint.
    Returns:
        dict: A description of the image, including the annotated details in JSON format.
    """
    prompt = f"""
    The following is a transcription of a customer's complaint: "{original_complaint}". 
    Please describe the image and identify any issues related to the customer complaint.
    Ensure the response is in JSON format with the following key names:

    1. "GeneralDescription": Provide a high-level overview of the scene depicted in the image.
    2. "KeyElements": List and describe the main objects or people in the image. Each element should have:
       - "ElementName": The name of the element.
       - "Description": A detailed description of the element.
       - "Position": The position of the element in the image.
    3. "Issues": List and describe any specific problems or complaints observed in the image that relate to the original complaint. Each issue should have:
       - "IssueDescription": A detailed description of the issue.
       - "IssueContext": Context or additional information about the issue.
    4. "BoundingBoxes": List coordinates for any issues or important elements to visually reference them in the image. Each bounding box should have:
       - "ElementName": The name of the element.
       - "Coordinates": The coordinates in the format "x_min, y_min, x_max, y_max".
    5. "Recommendations": Suggest potential solutions or actions based on the issues identified. Each recommendation should have:
       - "RecommendationDescription": A detailed description of the recommended action.

    Ensure the JSON keys are consistent and structured as specified above.
    """
    response = describe_local_image(client, image_path, deployment_name, prompt)
    if not response:
        print("No response received from describe_local_image.")
        return None

    print(f"Response from describe_local_image: {response}")

    # Remove the triple backticks and ensure valid JSON format
    response = response.replace("```json", "").replace("```", "").strip()

    # Find the position of the last valid closing curly brace and truncate any extra data
    last_brace_pos = response.rfind("}")
    if last_brace_pos != -1:
        response = response[:last_brace_pos + 1]

    # Save the description to a file
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    description_file_path = os.path.join(output_dir, "description.txt")
    
    with open(description_file_path, "w") as file:
        file.write(response)

    # Parse the JSON response
    try:
        description_json = json.loads(response)
        return description_json
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None

def create_annotated_image(image_path, description, output_path):
    """
    Creates an annotated image with bounding boxes highlighting key elements.

    Args:
        image_path (str): Path to the original image file.
        description (dict): The description containing the annotations in JSON format.
        output_path (str): Path to save the annotated image.

    Returns:
        None
    """
    try:
        # Open the original image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Load a font
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # Extract bounding box coordinates from the JSON description
        bounding_boxes = description.get("BoundingBoxes", [])
        
        if bounding_boxes:
            for bbox in bounding_boxes:
                element = bbox.get("ElementName")
                coordinates = bbox.get("Coordinates")
                if coordinates:
                    coords = [int(value.strip()) for value in coordinates.split(",")]
                    draw.rectangle(coords, outline="red", width=2)
                    text_position = (coords[0], max(0, coords[1] - 10))
                    draw.text(text_position, element, fill="red", font=font)

            # Save the annotated image
            image.save(output_path)
            print(f"Annotated image saved at: {output_path}")
        else:
            print("No bounding boxes found in the description.")

    except Exception as e:
        print(f"An error occurred while creating the annotated image: {e}")
