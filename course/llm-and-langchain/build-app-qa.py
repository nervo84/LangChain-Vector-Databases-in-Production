from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)
resp = chain.run("Qual è il senso della vita?")
print(resp)