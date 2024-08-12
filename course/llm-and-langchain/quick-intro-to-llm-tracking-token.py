from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Raccontami una barzelletta")
    print(cb)
    
