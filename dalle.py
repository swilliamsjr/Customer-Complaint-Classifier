# dalle.py

import openai
import requests
import os
from PIL import Image
from utils import create_azure_openai_client_2

def create_prompt(transcription):
    prompt = f"""
    The following is a transcription of a customer's complaint. Please generate an image that visually represents the scenario described in the complaint. Ensure the image captures the key elements of the scene, including any dissatisfaction expressed by the customer, interactions between people, and any notable objects or settings mentioned. Here are the details of the complaint:

    Customer Complaint: "{transcription}"

    Key Details:
    1. Include any interactions between individuals mentioned in the complaint.
    2. Depict any specific objects, actions, or settings referenced by the customer.
    3. Capture the emotions and body language that reflect the sentiments expressed in the complaint.

    Context:
    - The setting should match the one described in the complaint (e.g., restaurant, store, office, etc.).
    - Ensure the scene includes all key elements mentioned by the customer.

    Hints:
    - Use visual cues to highlight important aspects of the complaint (e.g., facial expressions, body language, condition of objects).
    - The arrangement and interactions of characters should convey the overall sentiment of the complaint.
    """
    return prompt

def generate_image(client, prompt):
    """
    Generates an image based on a prompt using OpenAI's DALL-E model.
    Returns:
        str: The URL of the generated image.
    """
    # Call the DALL-E model to generate an image based on the prompt
    result = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    # Extract and return the image URL
    image_url = result.data[0].url
    return image_url

def generate_image_from_complaint(transcription):
    prompt = create_prompt(transcription)
    client = create_azure_openai_client_2()
    image_url = generate_image(client, prompt)
    return image_url

# Example Usage (for testing purposes, remove/comment when deploying):
# if __name__ == "__main__":
#     transcription = "A busy restaurant where a server ignores customers. The server did not greet them, served cold food, and never checked on them. The customers look disappointed and the table is messy with empty glasses."
#     image_url = generate_image_from_complaint(transcription)
#     print(f"Generated image URL: {image_url}")
#
#     # Download the generated image and save it locally
#     image_data = requests.get(image_url).content
#     image_path = os.path.join("images", "generated_image.png")
#     os.makedirs(os.path.dirname(image_path), exist_ok=True)
#     with open(image_path, "wb") as image_file:
#         image_file.write(image_data)
#
#     # Display the generated image
#     image = Image.open(image_path)
#     image.show()
#     print(f"Generated image saved at: {image_path}")
