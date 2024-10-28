from langchain.llms import OpenAI

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)

text = "Quale sarebbe un buon nome per un'azienda che produce calzini colorati?"

print(llm(text))