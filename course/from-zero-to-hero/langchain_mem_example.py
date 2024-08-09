from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os


# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Parlami di te.")

# Continue the conversation
conversation.predict(input="Cosa sai fare?")
conversation.predict(input="Come puoi aiutarmi con l'analisi dei dati?")

# Display the conversation
print(conversation)