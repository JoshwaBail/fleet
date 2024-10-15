from typing import Optional, Any, List, Dict

from fleet.response.response import ResponseObject

import openai
import anthropic
import groq


class Agent:
    def __init__(self, client: Optional[Any], system_prompt: Dict[str, str], name: str, description: str):
        self.client = client if client else None
        self.name = name if name else "Agent"
        self.description = description if description else ""
        self.messages = []
        self.system_prompt = system_prompt if system_prompt else "You are a helpful assistant."

    def __str__(self):
        return self.name
    
    def add_message(self, message: Dict[str, str]):
        self.messages.append(message)
    
    def set_system_prompt(self):
        self.messages.append({"role": "system", "content": self.system_prompt})

    def get_messages(self):
        return self.messages
    
    def clear_messages(self):
        self.messages = []      

    def send_message(self, model, message: Dict[str, str]):
        self.set_system_prompt()
        self.add_message(message)
        
        if not model:
            raise ValueError("Model not specified")

        if isinstance(self.client, openai.Client):
            if model not in [model.id for model in self.client.models.list().data]:
                raise ValueError(f"Model {model} not available for OpenAI")
    
            response = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
            )
            content = response.choices[0].message.content
            self.add_message({"role": "assistant", "content": content})
            self.content = ResponseObject(
                content=content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return self.content
        
        elif isinstance(self.client, anthropic.Anthropic):
            available_models = self.client.models.list().data
            if not any(m.id == model for m in available_models):
                raise ValueError(f"Model {model} not available for Anthropic")
            response = self.client.messages.create(
                model=model,
                messages=self.messages,
            )
            content = response.content[0].text
            self.add_message({"role": "assistant", "content": content})
            self.content = ResponseObject(
                content=content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            return self.content
        
        elif isinstance(self.client, groq.Groq):
            # TODO: need to update this list
            groq_models = ["mixtral-8x7b-32768", "llama2-70b-4096"]
            if model not in groq_models:
                raise ValueError(f"Model {model} not available for Groq")
            response = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
            )
            content = response.choices[0].message.content
            self.add_message({"role": "assistant", "content": content})
            self.content = ResponseObject(
                content=content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return self.content
                
        else:
            raise ValueError("Unsupported client type")
