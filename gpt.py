# gpt.py

import openai
import json
from utils import chat, create_openai_client
import os

# Load categories from categories.json
def load_categories():
    file_path = os.path.join(os.path.dirname(__file__), "categories.json")
    with open(file_path, "r") as file:
        categories = json.load(file)
    return categories

# Function to classify the customer complaint based on the image description
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_VISION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_VISION")
GPT_DEPLOYMENT = "gpt-4-turbo"
GPT_VERSION = "2024-08-01-preview"

def classify_with_gpt(description, categories):
    """
    Classifies the customer complaint into a category/subcategory based on the image description.
    Returns:
        str: The category and subcategory of the complaint.
    """
    # Create a prompt that includes the image description and the categories
    prompt = f"""
    Classify the following customer complaint description into one of the categories and subcategories provided below:
    Description: {description}
    
    Categories and Subcategories:
    {json.dumps(categories, indent=4)}
    
    Provide the most appropriate category and subcategory.
    """

    gpt4v = create_openai_client(GPT_VERSION, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
    response = chat(gpt4v, GPT_DEPLOYMENT, prompt)
    print(response)
    return response

# # Example Usage (for testing purposes, remove/comment when deploying):
# if __name__ == "__main__":
#     categories = load_categories()
#     description = {
#         "GeneralDescription": "The image depicts a restaurant scene with several patrons and a waiter. The environment is lively with many guests having meals or conversations.",
#         "KeyElements": [
#             {
#                 "ElementName": "Waiter",
#                 "Description": "A male waiter in a formal attire of vest and bow tie, handling a stack of plates and a napkin.",
#                 "Position": "Center, facing forward"
#             },
#             {
#                 "ElementName": "Plate of food",
#                 "Description": "A white plate with what looks like a fish dish accompanied by vegetables and a wedge of lemon.",
#                 "Position": "On the table in the foreground"
#             }
#         ],
#         "Issues": [
#             {
#                 "IssueDescription": "One of the patrons depicted toward the left part of the image looks visibly stressed and concerned, covering his face.",
#                 "IssueContext": "This might indicate dissatisfaction or discomfort possibly due to the busy environment or service issues."
#             }
#         ],
#         "BoundingBoxes": [
#             {
#                 "ElementName": "Stressed Patron",
#                 "Coordinates": "180, 406, 251, 630"
#             }
#         ],
#         "Recommendations": [
#             {
#                 "RecommendationDescription": "Management should consider checking on the patron who appears stressed to resolve any immediate issues or discomforts."
#             }
#         ]
#     }
#     classification = classify_with_gpt(description, categories)
#     print(classification)
