from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

local_path = './models/gpt4all-lora-quantized-ggml.bin'  # replace with your desired local file path

import requests
from pathlib import Path
from tqdm import tqdm

local_path = './models/gpt4all-lora-quantized-ggml.bin'
Path(local_path).parent.mkdir(parents=True, exist_ok=True)

# Example model. Check https://github.com/nomic-ai/pyllamacpp for the latest models.
url = 'https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized-ggml.bin'
# url = 'http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin'

# send a GET request to the URL to download the file. Stream since it's large
response = requests.get(url, stream=True)

# open the file in binary mode and write the contents of the response to it in chunks
# This is a large file, so be prepared to wait.
with open(local_path, 'wb') as f:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            f.write(chunk)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Verbose is required to pass to the callback manager
llm = GPT4All(model="./models/ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm_chain.run(question)

question = "Write a poem about friendship that rhymes."

llm_chain.run(question)

question = "Write a social media post to celebrate mother's day."

llm_chain.run(question)

question = "What happens when it rains somewhere?"

llm_chain.run(question)

template = """Question: {question}

Answer: Let's answer in two sentence while being funny."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

llm_chain.run(question)