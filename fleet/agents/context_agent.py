from typing import List, Dict, Optional, Any

from fleet.agents.base import Agent


class ContextAgent(Agent):
    def __init__(self, client: Optional[Any], system_prompt: Dict[str, str], name: str, description: str,  agents: List[Agent]):
        super().__init__(client, system_prompt, name, description)
        self.agents = agents

    def set_system_prompt(self):
        self.messages.append({"role": "system", "content": self.system_prompt})

    def analyse_agents(self, model):
        agents_info = {}
        for i, agent in enumerate(self.agents):
            other_agents_info = []
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    other_agents_info.append({
                        "name": other_agent.name,
                        "content": other_agent.content
                    })
        prompt = (
            f"Analyse {agent.name} which has written the following content: {agent.content}. "
            f"Are there any overlaps between the content of {agent.name} and the other agents? Are there any contradictions? You should look at this in terms of the content they have written."
            f"The other agents are:\n\n"
            f"{self._format_other_agents(other_agents_info)}"
        )
        print(prompt)
        response = self.send_message(model=model, message={"role": "user", "content": prompt})
        return response

    def _format_other_agents(self, other_agents_info):
        formatted = ""
        for agent_info in other_agents_info:
            formatted += (
                f"Name: {agent_info['name']}\n"
                f"Content: {agent_info['content'].content}\n\n"
            )
        return formatted.strip()
   
        