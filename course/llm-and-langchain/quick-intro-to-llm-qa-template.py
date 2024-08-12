from langchain import PromptTemplate
from langchain import LLMChain
from langchain import HuggingFaceHub, LLMChain

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "Qual'è la città capitale della Francia?"


# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))