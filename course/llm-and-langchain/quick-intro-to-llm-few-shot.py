from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Now you can access the variables using os.getenv or os.environ
openai_api_key = os.getenv('OPENAI_API_KEY')
# create our examples
examples = [
    {
        "query": "Che tempo fa?",
        "answer": "Piove a dirotto, meglio portare l'ombrello!"
    }, {
        "query": "Quanti anni hai?",
        "answer": "L'età è solo un numero, ma io sono senza tempo."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """Di seguito sono riportati alcuni estratti di conversazioni con un assistente AI. 
L'assistente è noto per il suo umorismo e la sua arguzia, 
fornendo risposte divertenti e divertenti alle domande degli utenti. 
Ecco alcuni esempi:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)



# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
res = chain.run("Qual è il senso della vita?")
print(res)