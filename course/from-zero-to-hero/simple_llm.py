from langchain.llms import OpenAI
from dotenv import load_dotenv
import os


# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, openai_api_key=openai_api_key)

text = "Suggerisci una routine di allenamento personalizzata per chi desidera migliorare la resistenza cardiovascolare e preferisce le attività all'aperto."
print(llm(text))
