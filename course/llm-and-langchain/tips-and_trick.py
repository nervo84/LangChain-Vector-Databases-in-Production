from dotenv import load_dotenv

load_dotenv()

import openai

prompt_system = "You are a helpful assistant whose goal is to help write stories."

prompt = """Continue the following story. Write no more than 50 words.

Once upon a time, in a world where animals could speak, a courageous \
mouse named Benjamin decided to"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0]['message']['content'])

import openai

prompt_system = """You are a helpful assistant whose goal is to help write \
product descriptions."""

prompt = """Write a captivating product description for a luxurious, handcrafted\
, limited-edition fountain pen made from rosewood and gold.
Write no more than 50 words."""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0]['message']['content'])

import openai

prompt_system = "You are a helpful assistant whose goal is to write short poems."

prompt = """Write a short poem about {topic}."""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0]['message']['content'])

import openai

prompt_system = "You are a helpful assistant whose goal is to write short poems."

prompt = """Write a short poem about {topic}."""

examples = {
    "nature": """Birdsong fills the air,\nMountains high and valleys \
deep,\nNature's music sweet.""",
    "winter": """Snow blankets the ground,\nSilence is the only \
sound,\nWinter's beauty found."""
}

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="nature")},
        {"role": "assistant", "content": examples["nature"]},
        {"role": "user", "content": prompt.format(topic="winter")},
        {"role": "assistant", "content": examples["winter"]},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0]['message']['content'])