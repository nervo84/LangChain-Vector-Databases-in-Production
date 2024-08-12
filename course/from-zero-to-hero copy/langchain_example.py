from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os


# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, openai_api_key=openai_api_key)
prompt = PromptTemplate(
    input_variables=["product"],
    template="Qual è un buon nome per un'azienda che produce {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("bottiglie d'acqua ecologiche"))