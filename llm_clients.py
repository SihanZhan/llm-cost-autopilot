from dataclasses import dataclass
from urllib import response
from models import ModelConfig
from openai import OpenAI
from anthropic import Anthropic
from ollama import Ollama
@dataclass
class Response:
    output_text: str
    input_text: int
    output_tokens: int
    latency: float
    cost: float
    model_id: str

"""Open AI, Anthropic, and Ollama clients are initialized with API keys.
The send_request function takes a prompt and a model configuration, sends the 
request to the appropriate model provider, and returns a Response object containing 
the output text, token counts, latency, cost, and model ID."""

def send_request(prompt, model_config):
    if model_config.provider == "openai":
        client = OpenAI(api_key ="key")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens



    elif model_config.provider == "anthropic":
        client = Anthropic(api_key="key")
        response = client.completions.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.completion
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens


    elif model_config.provider == "ollama":
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": "your prompt here"}]
    )

        text = response["message"]["content"]