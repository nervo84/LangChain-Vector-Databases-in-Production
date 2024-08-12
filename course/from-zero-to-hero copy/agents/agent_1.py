from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os


# Load environment variables from the .env file
load_dotenv()


# Now you can access the variables using os.getenv or os.environ
openai_api_key      = os.getenv('OPENAI_API_KEY')
activeloop_token    = os.getenv('ACTIVELOOP_TOKEN')
activaloop_org_id   = os.getenv('ACTIVELOOP_ORG_ID')

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{activaloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

response = agent.run("Quando è nato Michael Jordan?")
print(response)