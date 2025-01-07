# whisper.py

import openai
from utils import create_azure_openai_client_1

# Function to transcribe customer audio complaints using the Whisper model
def transcribe_audio(client, audio_file_path):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.
    Returns:
        str: The transcribed text of the audio file.
    """
    # The deployment ID of the Whisper model
    deployment_id = "whisper"

    # Load the audio file
    with open(audio_file_path, "rb") as audio_file:
        # Call the Whisper model to transcribe the audio file
        result = client.audio.transcriptions.create(
            file=audio_file,
            model=deployment_id
        )

    # Extract the transcription and return it
    transcription = result.text
    return transcription

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    client = create_azure_openai_client_1()
    audio_test_file = r"C:\Users\sammy\OneDrive\Documents\SoundRecordings\Recording.m4a"
    transcription = transcribe_audio(client, audio_test_file)
    print("Transcription:", transcription)
