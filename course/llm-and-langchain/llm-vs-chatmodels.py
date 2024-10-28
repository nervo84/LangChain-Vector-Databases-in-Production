from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(temperature=0)

prompt = PromptTemplate(
  input_variables=["product"],
  template="Qual è un buon nome per un'azienda che produce {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

print( chain.run("wireless headphones") )