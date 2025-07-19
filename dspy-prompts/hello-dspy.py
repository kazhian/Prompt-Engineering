import dspy
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')
print(f"loaded api_key. Key length: {len(api_key)}")

# Initialize the language model with the API key
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.9, max_tokens=3000, stop=None, cache=False)
dspy.configure(lm = gpt_4o_mini)

# Define a simple question-answering prompt
qa = dspy.Predict('question -> answer')

response = qa(question='What is the capital of France?')
print(response.answer)

# Define a simple Chain of Thought (CoT) prompt
cot = dspy.ChainOfThought('question -> answer')
response = cot(question="Two dice are tossed. What is the probability that the sum of the numbers on the dice is 7?")
print("gpt-4o-mini: " + response.answer)

# with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
#     response = cot(question="Two dice are tossed. What is the probability that the sum of the numbers on the dice is 7?")
#     print('GPT-3.5-turbo:', response.answer)

sentence = "It's a charming and often affection journey through the world of AI, with a touch of humor and a lot of heart."
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence=sentence).sentiment
print(f"Sentiment of the sentence: {response}")

# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)