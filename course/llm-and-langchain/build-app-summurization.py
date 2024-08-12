# Import necessary modules
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize language model
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="path/to/your/pdf/file.pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])