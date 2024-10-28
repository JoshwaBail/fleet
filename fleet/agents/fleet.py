import asyncio
from typing import List, Union, Dict, Any
from termcolor import colored
from fleet.agents.base import Agent
from fleet.response.response import ResponseObject

class Fleet:
    def __init__(self, agents_or_composers, name="Fleet", description="", synthesize=True):
        self.agents_or_composers = agents_or_composers
        self.name = name
        self.description = description
        self.synthesize = synthesize
        self.colors = ['magenta', 'cyan', 'yellow', 'green', 'blue', 'red']
        self._assign_colors()

    def _assign_colors(self):
        for i, agent in enumerate(self.agents_or_composers):
            agent.color = self.colors[i % len(self.colors)]

    def _print_agent_action(self, agent: Union[Agent, 'Fleet'], action: str, message: str):
        color = getattr(agent, 'color', 'white')  # Default to white if color is not set
        print(colored(f"[{self.name}] {agent.name} - {action}: {message}", color))

    def compose_synchronously(self, initial_message: Dict[str, str], model: str, temperature: float = 0.0, max_tokens: int = 1024) -> ResponseObject:
        """
        Compose agents synchronously, passing the output of one agent to the next.
        """
        current_message = initial_message
        final_response = None
        print(current_message['content'])

        for item in self.agents_or_composers:
            self._print_agent_action(item, "Processing", current_message['content'])
            if isinstance(item, Fleet):
                response = item.compose_synchronously(current_message, model, temperature, max_tokens)
            else:  # It's an individual agent
                response = item.send_message(model=model, message=current_message, temperature=temperature, max_tokens=max_tokens, function_schemas=item.function_schemas)
            response_content = response.content if isinstance(response, ResponseObject) else response['content']
            current_message = {"role": "user", "content": current_message["content"] + f"\n\n{item.name}: {response_content}"}
            self._print_agent_action(item, "Responded", response_content)
            final_response = response

        return final_response

    async def compose_asynchronously(self, initial_message: Dict[str, str], model: str, temperature: float = 0.0, max_tokens: int = 1024) -> Union[ResponseObject, List[ResponseObject]]:
        """
        Compose agents asynchronously, running all agents in parallel and optionally synthesizing the results.
        """
        async def agent_task(agent: Union[Agent, 'Fleet'], initial_message: Dict[str, str], model: str, temperature: float, max_tokens: int) -> ResponseObject:
            self._print_agent_action(agent, "Processing", initial_message['content'][:50] + "...")
            if isinstance(agent, Fleet):
                response = agent.compose_synchronously(initial_message=initial_message, model=model, temperature=temperature, max_tokens=max_tokens)
            else:  # It's an individual agent
                response = agent.send_message(model=model, message=initial_message, temperature=temperature, max_tokens=max_tokens, function_schemas=agent.function_schemas )
            
            # Fix for the slicing error
            response_content = response.content if hasattr(response, 'content') else str(response)
            truncated_content = response_content[:50] + "..." if len(response_content) > 50 else response_content
            self._print_agent_action(agent, "Responded", truncated_content)
            return response

        tasks = [agent_task(agent=agent, initial_message=initial_message, model=model, temperature=temperature, max_tokens=max_tokens) for agent in self.agents_or_composers]
        responses = await asyncio.gather(*tasks)
        
        if self.synthesize:
            # Synthesize the results
            synthesized_response = self.synthesize_responses(responses=responses, model=model, temperature=temperature, max_tokens=max_tokens)
            return synthesized_response
        else:
            return responses

    def synthesize_responses(self, responses: List[ResponseObject], model: str, temperature: float, max_tokens: int) -> ResponseObject:
        """
        Synthesize the responses from multiple agents into a single, cohesive output.
        """
        synthesis_prompt = f"""
        As an expert synthesizer, your task is to combine the outputs of multiple agents into a single, coherent response. 
        The team's overall goal is: {self.description}

        Here are the individual agent responses:

        {self._format_agent_responses(responses)}

        Please synthesize these responses into a single, cohesive output that aligns with the team's goal. 
        Ensure that you address all key points raised by the individual agents while avoiding redundancy. 
        The final output should be well-structured and provide clear, actionable insights or recommendations.
        """
        if all(isinstance(agent, Fleet) for agent in self.agents_or_composers):
            client = self.agents_or_composers[0].agents_or_composers[0].client
            synthesis_agent = Agent(client, synthesis_prompt, "Synthesis Agent", "Synthesizes multiple agent responses")
            synthesis_agent.color = 'white'  # Assign a color to the synthesis agent
            synthesized_response = synthesis_agent.send_message(model=model, message={"role": "user", "content": synthesis_prompt}, temperature=temperature, max_tokens=max_tokens, function_schemas=None)
        
        elif isinstance(self.agents_or_composers[0], Agent):
            synthesis_agent = Agent(self.agents_or_composers[0].client, synthesis_prompt, "Synthesis Agent", "Synthesizes multiple agent responses")
            synthesis_agent.color = 'white'  # Assign a color to the synthesis agent
            synthesized_response = synthesis_agent.send_message(model=model, message={"role": "user", "content": synthesis_prompt}, temperature=temperature, max_tokens=max_tokens, function_schemas=None)
        
        elif isinstance(self.agents_or_composers[0], Fleet):
            client = self.agents_or_composers[0].agents_or_composers[0].client
            synthesis_agent = Agent(client, synthesis_prompt, "Synthesis Agent", "Synthesizes multiple agent responses")
            synthesis_agent.color = 'white'  # Assign a color to the synthesis agent
            synthesized_response = synthesis_agent.send_message(model=model, message={"role": "user", "content": synthesis_prompt}, temperature=temperature, max_tokens=max_tokens, function_schemas=None)

        self._print_agent_action(synthesis_agent, "Synthesized", synthesized_response.content[:50] + "...")
        return synthesized_response

    def _format_agent_responses(self, responses: List[ResponseObject]) -> str:
        formatted_responses = ""
        for i, response in enumerate(responses):
            agent = self.agents_or_composers[i]
            if isinstance(agent, Agent):
                formatted_responses += f"Agent: {agent.name}\n"
                formatted_responses += f"Description: {agent.description}\n"
                formatted_responses += f"System Prompt: {agent.system_prompt}\n"
                formatted_responses += f"Response: {response.content}\n\n"
            elif isinstance(agent, Fleet):
                formatted_responses += f"Fleet: {agent.name}\n"
                formatted_responses += f"Description: {agent.description}\n"
                formatted_responses += f"Response: {response.content}\n\n"
        return formatted_responses

    def compose(self, initial_message: Dict[str, str], model: str, mode: str = "synchronously", temperature = 0, max_tokens = 1024) -> Union[ResponseObject, List[ResponseObject]]:
        """
        Compose agents based on the specified mode.
        """
        # In the Fleet's compose method
        print(colored(f"[{self.name}] Starting composition in {mode} mode", 'white', 'on_blue'))
        if mode == "synchronously":
            return self.compose_synchronously(initial_message=initial_message, model=model, temperature=temperature, max_tokens=max_tokens)
        elif mode == "asynchronously":
            print(temperature)
            return asyncio.run(self.compose_asynchronously(initial_message=initial_message, model=model, temperature=temperature, max_tokens=max_tokens))
        else:
            raise ValueError("Invalid mode. Choose 'synchronously' or 'asynchrounsly'.")
