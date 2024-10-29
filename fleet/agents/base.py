import logging
from typing import Optional, Any, Dict, List, Union, Callable
from pydantic import BaseModel
from fleet.response.response import ResponseObject
import openai
import anthropic
import groq
import json
import re


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, 
                 client: Any, 
                 system_prompt: Dict[str, str] = "You are a helpful assistant.", 
                 name: Optional[str] = "Agent",
                 description: Optional[str] = "",
                 json_mode: Optional[bool] = False, 
                 json_schema: Optional[Union[Dict[str, Any], BaseModel]] = None,
                 functions: Optional[Dict[str, Callable]] = None,
                 function_schemas: Optional[List[Dict[str, Any]]] = None,
                 ):
        logger.info(f"Initializing Agent: {name}")
        self.client = client
        self.name = name
        self.description = description
        self.messages = []
        self.system_prompt = system_prompt
        self.json_mode = json_mode  
        self.json_schema = json_schema
        self.color = 'white'
        self.functions = functions
        self.function_schemas = function_schemas
        self.set_system_prompt()

    def __str__(self):
        return self.name
    
    def add_message(self, message: Dict[str, str]):
        logger.debug(f"Adding message to {self.name}: {message}")
        self.messages.append(message)
    
    def set_system_prompt(self):
        logger.debug(f"Setting system prompt for {self.name}")
        self.messages.append({"role": "system", "content": self.system_prompt})

    def get_messages(self):
        return self.messages
    
    def clear_messages(self):
        logger.debug(f"Clearing messages for {self.name}")
        self.messages = []      

    def send_message(self, model, message: Dict[str, str], function_schemas, temperature: float = 0.0, max_tokens: int = 1024):
        logger.info(f"Sending message with {self.name} using model: {model}")
        if function_schemas:
            self.function_schemas = function_schemas
        
        self._prepare_message(message)
        self._validate_model(model)
        
        if isinstance(self.client, openai.Client):
            return self._handle_openai_chat(model, temperature, max_tokens)
        elif isinstance(self.client, anthropic.Anthropic):
            return self._handle_anthropic_chat(model, temperature, max_tokens)
        elif isinstance(self.client, groq.Groq):
            return self._handle_groq_chat(model, temperature, max_tokens)
        else:
            logger.error("Unsupported client type")
            raise ValueError("Unsupported client type")

    def _prepare_message(self, message: Dict[str, str]):
        """Prepare the message by adding JSON instructions if needed"""
        if self.json_mode and "json" not in self.system_prompt.lower():
            json_instruction = f"\n\nPlease respond in valid JSON format according to the following schema: {json.dumps(self.json_schema)}"
            if isinstance(message['content'], str):
                message['content'] += json_instruction
            elif isinstance(message['content'], dict):
                message['content']['text'] += json_instruction
        
        self.add_message(message)

    def _validate_model(self, model):
        """Validate that the model is specified and available"""
        if not model:
            logger.error("Model not specified")
            raise ValueError("Model not specified")

    def _prepare_openai_kwargs(self, model: str, temperature: float, max_tokens: int) -> dict:
        """Prepare kwargs for OpenAI API call"""
        kwargs = {
            "model": model,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        if self.function_schemas:
            tools = [{"type": "function", "function": schema} for schema in self.function_schemas]
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        return kwargs

    def _handle_function_calls(self, tool_calls, model: str, temperature: float, max_tokens: int):
        """Handle function calls in the OpenAI response"""
        self.add_message({"role": "assistant", "content": None, "tool_calls": tool_calls})
        
        for tool_call in tool_calls:
            if tool_call.type == "function":
                self._process_single_function_call(tool_call)
        
        final_response = self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return final_response.choices[0].message.content

    def _process_single_function_call(self, tool_call):
        """Process a single function call"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name in self.functions:
            logger.info(f"Executing function: {function_name}")
            function_result = self.functions[function_name](**function_args)
            content = json.dumps(function_result)
        else:
            logger.error(f"Function '{function_name}' not found")
            content = json.dumps({"error": f"Function '{function_name}' not found"})
        
        self.add_message({"role": "tool", "content": content, "tool_call_id": tool_call.id})

    def _handle_openai_chat(self, model: str, temperature: float, max_tokens: int):
        """Handle chat completion for OpenAI"""
        logger.debug("Using OpenAI client")
        if model not in [model.id for model in self.client.models.list().data]:
            logger.error(f"Model {model} not available for OpenAI")
            raise ValueError(f"Model {model} not available for OpenAI")

        kwargs = self._prepare_openai_kwargs(model, temperature, max_tokens)
        logger.debug(f"OpenAI API call parameters: {kwargs}")
        
        response = self.client.chat.completions.create(**kwargs)
        logger.debug(f"OpenAI API response: {response}")

        if response.choices[0].message.tool_calls:
            logger.info("Function call detected in response")
            content = self._handle_function_calls(
                response.choices[0].message.tool_calls,
                model,
                temperature,
                max_tokens
            )
        else:
            content = response.choices[0].message.content

        self.add_message({"role": "assistant", "content": content})
        self.content = ResponseObject(
            content=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        return self.content

    def _handle_anthropic_chat(self, model: str, temperature: float, max_tokens: int):
        """Handle chat completion for Anthropic"""
        logger.debug("Using Anthropic client")
        available_models = self.client.models.list().data
        if not any(m.id == model for m in available_models):
            logger.error(f"Model {model} not available for Anthropic")
            raise ValueError(f"Model {model} not available for Anthropic")
        
        response = self.client.messages.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.content[0].text
        
        if self.json_mode:
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logger.error("Response is not in valid JSON format")
                raise ValueError("Response is not in valid JSON format")
        
        self.add_message({"role": "assistant", "content": content})
        self.content = ResponseObject(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        return self.content

    def _handle_groq_chat(self, model: str, temperature: float, max_tokens: int):
        """Handle chat completion for Groq"""
        logger.debug("Using Groq client")
        groq_models = ["mixtral-8x7b-32768", "llama2-70b-4096", "llama3-8b-8192"]
        if model not in groq_models:
            logger.error(f"Model {model} not available for Groq")
            raise ValueError(f"Model {model} not available for Groq")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if self.json_mode else None
        )
        content = response.choices[0].message.content
        self.add_message({"role": "assistant", "content": content})
        self.content = ResponseObject(
            content=json.loads(content) if self.json_mode else content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        return self.content
