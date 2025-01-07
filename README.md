Customer-Complaint-Classifier
The Customer-Complaint-Classifier is an AI-driven application designed to enhance customer service by efficiently handling and categorizing customer complaints. This innovative app leverages advanced language and vision models to transcribe audio complaints, generate visual representations, and accurately classify issues based on their descriptions.

Key Features
Audio Transcription: Utilizes OpenAI's Whisper model to convert customer audio complaints into text for further processing.

Visual Representation: Uses DALL-E to generate images based on the transcribed complaints, providing a visual context.

Image Description and Annotation: Employs GPT-4 to describe the generated images and highlight key elements and issues using bounding boxes.

Complaint Classification: Classifies complaints into relevant categories and subcategories, helping organizations understand and address customer concerns effectively.

Technology Stack
OpenAI Whisper: For accurate audio transcription.

DALL-E: To create detailed visual representations of complaints.

GPT-4: For image description, annotation, and complaint classification.

Installation
Clone the repository:

bash
git clone https://github.com/<your-username>/Customer-Complaint-Classifier.git
Navigate to the project directory:

bash
cd Customer-Complaint-Classifier
Install the required dependencies:

bash
pip install -r requirements.txt
Usage
Ensure you have the necessary API keys and endpoints set as environment variables.

Run the main script:

bash
python main.py
Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.