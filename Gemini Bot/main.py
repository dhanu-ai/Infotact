#Use the API key of your own
#Create a .env file and add the API key to it
#Keep the name of variable in the .env file as GOOGLE_API

import base64, dotenv
import os
from google import genai
from google.genai import types

# Load environment variables
dotenv.load_dotenv()

# Memory to store conversation history
conversation_history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="hii")],
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="Hello! How can I help you today?")],
    ),
]

def generate(user_input):
    global conversation_history

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-pro-exp-02-05"

    # Add new user input to the conversation history
    conversation_history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        )
    )

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    # Generate response with the updated history
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=conversation_history,  # Pass full history
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        # Append model response to the conversation history
        conversation_history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=chunk.text)],
            )
        )

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        generate(user_input)
