from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

llm = ChatOpenAI(model_name="gpt-4")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)
# add to messages
messages.append(prompt)

# add latest AI response to messages
response = llm(messages)