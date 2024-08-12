from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

summarization_template = "Riassumi il testo seguente in una frase: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

text = "LangChain fornisce molti moduli che possono essere utilizzati per creare applicazioni di modelli linguistici. I moduli possono essere combinati per creare applicazioni più complesse o utilizzati singolarmente per applicazioni semplici. Il componente di base più elementare di LangChain è la chiamata di un LLM su un input. Facciamo un semplice esempio di come farlo. A questo scopo, immaginiamo di creare un servizio che genera un nome aziendale in base a ciò che l'azienda produce."
summarized_text = summarization_chain.predict(text=text)

print(summarized_text)