# This is a sample program to demonstrate few-shot prompting for entity extraction
# Before you run this code
#   pip install openai python-dotenv
#   create .env file in the project root and add OPENAI_API_KEY=<your_api_key>

import os
import openai
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file in the parent directory
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
llm_name = 'gpt-4-turbo'

def read_api_key():
    temp = os.getenv("OPENAI_API_KEY")
    if not temp:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    return temp

def extract_entities(message: str, api_key: str = None) -> Dict[str, Any]:
    """
    Extract entities from a given message using OpenAI's API.

    Args:
        message (str): The text message to analyze
        api_key (str, optional): OpenAI API key.

    Returns:
        dict: A dictionary containing the entities extracted from the message
    """
    client = openai.OpenAI(api_key=api_key)

    system_message = """
    You are an entity extraction assistant. Analyze the entire customer review and extract all product-related entities.
    
    Important rules:
    1. Process the ENTIRE review as a single unit, even if it contains multiple sentences or paragraphs
    2. Extract all product names, brands, models, features, and components mentioned
    3. Combine related mentions (e.g., 'Logitech mouse' should be one entity, not two)
    4. Include both the specific product being reviewed and any comparison products mentioned
    
    Respond with a JSON object in this exact format:
    {
        "entities": ["entity1", "entity2", "entity3"]
    }
    
    Only include the JSON object in your response, with no other text or explanation.
    """
    user_message_template = "```{review}```"
    shot_1 = get_shot_1()
    shot_2 = get_shot_2()

    few_shot_examples = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message_template.format(review=shot_1[0])},
        {'role': 'assistant', 'content': f"{shot_1[1]}"},
        {'role': 'user', 'content': user_message_template.format(review=shot_2[0])},
        {'role': 'assistant', 'content': f"{shot_2[1]}"}
    ]

    few_shot_prompt = few_shot_examples + [
        {'role': 'user', 'content': user_message_template.format(review=message)}
    ]

    try:
        response = client.chat.completions.create(
            model=llm_name,
            messages=few_shot_prompt,
            response_format={"type": "json_object"}
        )

        # Parse the response
        result = response.choices[0].message.content
        return json.loads(result)

    except Exception as e:
        return {"error": str(e), "sentiment": "error", "confidence": 0.0}

def get_shot_1():
    return [
        """
        Ordered grey which advertises green lighting, when you're going for a cheap aesthetic, it's upsetting. Mouse works fine.
        """,
        """
        Entities: [Mouse]
        """
    ]

def get_shot_2():
    return [
         """
        I bought one of these for PC gaming. Loved it, then bought another for work.This mouse is not on par with high end mouses from like the Logitech MX Master series, but at 1/5-/8th the price, I didn't expect that level of quality.
        It does perform well, mouse wheel feels weighty, side buttons are well place with different textures so you can tell them apart.
        DPI buttons are handy for adjusting between games, work jobs, etc.
        The mouse does feel rather plasticky and cheap, but for the money, it about what I expected.I like a wired mouse to avoid the pointer/game jumping around due to latency.Long wire too, so snagging issues are minimized. Great value overall.
        """,
        """
        Entities: [Mouse, Logitech MX Master, DPI Buttons, Mouse Wheel, Wire]
        """
    ]


def main():
    """Main function to run the sentiment analysis from the command line."""
    print("\n=== Sentiment Analysis Tool ===")
    print("Enter a message to analyze (or 'quit' to exit):")

    while True:
        message = input("\nYour message: ").strip()

        if message.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not message:
            print("Please enter a message to analyze.")
            continue

        try:
            result = extract_entities(message, read_api_key())

            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                entities = result.get('entities', 'unknown')
                print(f"\nEntities: {entities}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()