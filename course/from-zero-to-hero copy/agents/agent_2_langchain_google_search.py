from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key  = os.getenv('OPENAI_API_KEY')
google_cs_id    = os.getenv('GOOGLE_CSE_ID')

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name = "gcse-search",
        func=search.run,
        description="utile quando hai bisogno di cercare su Google per rispondere a domande sugli eventi attuali"
    )
]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

response = agent("Quali sono le ultime novità sul rover su Marte?")
print(response['output'])