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

def api_key():
    temp = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
    return temp

def analyze_sentiment(message: str, api_key: str = None) -> Dict[str, Any]:
    """
    Analyze the sentiment of a given message using OpenAI's API.

    Args:
        message (str): The text message to analyze
        api_key (str, optional): OpenAI API key.

    Returns:
        dict: A dictionary containing the sentiment analysis result
    """
    client = openai.OpenAI(api_key=api_key)

    system_message = """
        You are a sentiment analysis assistant. 
        Classify the sentiment of the provided customer review as either positive or negative.
        
        Respond with a JSON object in the following format:
        {
            "sentiment": "positive" or "negative"
        }
        
        Do not include any other text or explanation in your response. Only return the JSON object.
        The review will be delimited by triple backticks (```).
    """

    user_message = f"""```
    {message}
    ```"""
    
    zero_shot_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model=llm_name,
            messages= zero_shot_prompt,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = response.choices[0].message.content
        return json.loads(result)
        
    except Exception as e:
        return {"error": str(e), "sentiment": "error", "confidence": 0.0}

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
            result = analyze_sentiment(message, api_key())
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                sentiment = result.get('sentiment', 'unknown').capitalize()
                confidence = result.get('confidence', 0)
                print(f"\nSentiment: {sentiment}")
                print(f"Confidence: {confidence:.2f}")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
