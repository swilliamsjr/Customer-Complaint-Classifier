# main.py

import os
from whisper import transcribe_audio
from dalle import generate_image_from_complaint
from vision import describe_image, create_annotated_image
from gpt import classify_with_gpt, load_categories
from utils import create_azure_openai_client_1, create_azure_openai_client_2, create_openai_client
import json

def main():
    """
    Orchestrates the workflow for handling customer complaints.
    
    Steps include:
    1. Transcribe the audio complaint.
    2. Generate an image representing the issue.
    3. Describe the generated image using the original complaint.
    4. Classify the complaint based on the image description.
    Returns:
    None
    """
    try:
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Transcribe the audio complaint
        print("Transcribing the audio complaint...")
        client1 = create_azure_openai_client_1()
        audio_test_file = os.path.join(os.path.dirname(__file__), "audio", "Recording.m4a")
        transcription = transcribe_audio(client1, audio_test_file)
        print(f"Transcription: {transcription}")

        # Save the transcription to a file
        transcription_path = os.path.join(output_dir, "transcription.txt")
        with open(transcription_path, "w") as file:
            file.write(transcription)

        # Step 2: Generate an image based on the prompt created from the transcription
        print("Generating the image based on the prompt...")
        image_url = generate_image_from_complaint(transcription)
        print(f"Generated image URL: {image_url}")

        image_path = os.path.join(output_dir, "generated_image.png")
        if image_url:
            # Download the image and save it locally
            import requests
            image_data = requests.get(image_url).content
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            print(f"Image saved at: {image_path}")

            # Step 3: Describe the generated image using the original complaint
            print("Describing the generated image...")
            api_version = "2024-08-01-preview"
            api_key = os.getenv("AZURE_OPENAI_API_KEY_VISION")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_VISION")
            client3 = create_openai_client(api_version, api_key, azure_endpoint)

            # Pass the transcription as the original complaint to describe_image
            description = describe_image(client3, image_path, transcription, "gpt-4-turbo")
            if description:
                print(f"Image Description: {description}")

                # Save the description to a file
                description_path = os.path.join(output_dir, "description.txt")
                with open(description_path, "w") as file:
                    file.write(json.dumps(description, indent=4))

                # Step 4: Create the annotated image
                create_annotated_image(image_path, description, os.path.join(output_dir, "annotated_image.png"))

                # Step 5: Classify the complaint based on the image description
                print("Classifying the complaint based on the image description...")
                categories = load_categories()
                description_json_str = json.dumps(description)
                classification = classify_with_gpt(description_json_str, categories)
                print(f"Classification: {classification}")

                # Save the classification result to a file
                classification_path = os.path.join(output_dir, "classification.txt")
                with open(classification_path, "w") as file:
                    file.write(classification)
            else:
                print("Failed to describe the image.")
        else:
            print("Failed to generate the image.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
