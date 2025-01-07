# utils.py

from mimetypes import guess_type
import os
from openai import AzureOpenAI
import base64
import json

# Function to create the first Azure OpenAI client
def create_azure_openai_client_1():
    api_key = os.getenv("AZURE_OPENAI_API_KEY_1")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION_1")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_1")
    if not api_key or not api_version or not azure_endpoint:
        raise ValueError("Azure OpenAI API key, version, and endpoint for client 1 must be set in the environment variables.")
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client

# Function to create the second Azure OpenAI client
def create_azure_openai_client_2():
    api_key = os.getenv("AZURE_OPENAI_API_KEY_2")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION_2")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_2")
    if not api_key or not api_version or not azure_endpoint:
        raise ValueError("Azure OpenAI API key, version, and endpoint for client 2 must be set in the environment variables.")
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client

# Function to create a third Azure OpenAI client
def create_azure_openai_client_3():
    api_key = os.getenv("AZURE_OPENAI_API_KEY_3")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION_3")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_3")
    if not api_key or not api_version or not azure_endpoint:
        raise ValueError("Azure OpenAI API key, version, and endpoint for client 3 must be set in the environment variables.")
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )
    return client

# Function to create an Azure OpenAI client
def create_openai_client(api_version, api_key, azure_endpoint):
    if not api_version:
        raise ValueError("Must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable.")
    client = AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )
    return client

# Function to generate an image using the DALL-E model
def generate_image(client, prompt, model, size, quality, style):
    result = client.images.generate(
        model=model, 
        prompt=prompt, 
        size=size, 
        quality=quality, 
        style=style
    )
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]
    return image_url

# Function to convert a local image to a data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Function to describe an image using the Azure OpenAI Chat API
def describe_local_image(client, image_path, deployment_name, prompt):
    data_url = local_image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content

# Function to describe an online image using the Azure OpenAI Chat API
def describe_online_image(client, image_url, deployment_name, prompt):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content

# Function to chat with the Azure OpenAI Chat API
def chat(gpt_client, deployment_name, prompt):
    response = gpt_client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    result = response.choices[0].message.content
    return result
