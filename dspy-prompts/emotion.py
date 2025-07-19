import dspy
from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the LM
gpt_4o_mino = dspy.LM(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.settings.configure(lm=gpt_4o_mino)

class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField(desc="The sentence to classify")
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField(desc="The sentiment of the sentence")

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

def main():
    sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion
    
    # Make prediction
    classify = dspy.Predict(Emotion)
    result = classify(sentence=sentence)
    
    # Print the compiled prompt for Emotion classification
    print("\n--- Emotion Classification Prompt ---")
    print(classify.demos)
    
    # Print the result
    print("\nEmotion Classification Result:")
    print(f"Input: {sentence}")
    print(f"Predicted Emotion: {result.sentiment}")

    # CoT
    context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

    text = "Lee scored 3 goals for Colchester United."

    faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    result = faithfulness(context=context, text=text)

    # Print the compiled prompt for CheckCitationFaithfulness
    print("\n--- Check Citation Faithfulness ---")
    print(f"Context: {context}")
    print(f"Text to verify: {text}")
    
    # Print the result
    print("\nCheck Citation Faithfulness Result:")
    print(f"Input: {context}")
    print(f"Text to verify: {text}")
    print(f"Faithfulness: {result.faithfulness}")
    if not result.faithfulness:
        print(f"Evidence: {result.evidence}")

if __name__ == "__main__":
    main()