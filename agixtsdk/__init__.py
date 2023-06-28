import requests
from typing import Dict, List, Any


class AGiXTSDK:
    def __init__(self, base_uri: str = None):
        if not base_uri:
            base_uri = "http://localhost:7437"
        self.base_uri = base_uri
        if self.base_uri[-1] == "/":
            self.base_uri = self.base_uri[:-1]

    def get_providers(self) -> List[str]:
        response = requests.get(f"{self.base_uri}/api/provider")
        return response.json()["providers"]

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/provider/{provider_name}")
        return response.json()["settings"]

    def get_embed_providers(self) -> List[str]:
        response = requests.get(f"{self.base_uri}/api/embedding_providers")
        return response.json()["providers"]

    def add_agent(
        self, agent_name: str, settings: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_uri}/api/agent",
            json={"agent_name": agent_name, "settings": settings},
        )
        return response.json()

    def import_agent(
        self,
        agent_name: str,
        settings: Dict[str, Any] = {},
        commands: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_uri}/api/agent/import",
            json={"agent_name": agent_name, "settings": settings, "commands": commands},
        )
        return response.json()

    def rename_agent(self, agent_name: str, new_name: str) -> str:
        response = requests.patch(
            f"{self.base_uri}/api/agent/{agent_name}",
            json={"new_name": new_name},
        )
        return response.json()

    def update_agent_settings(self, agent_name: str, settings: Dict[str, Any]) -> str:
        response = requests.put(
            f"{self.base_uri}/api/agent/{agent_name}",
            json={"settings": settings, "agent_name": agent_name},
        )
        return response.json()["message"]

    def update_agent_commands(self, agent_name: str, commands: Dict[str, Any]) -> str:
        response = requests.put(
            f"{self.base_uri}/api/agent/{agent_name}/commands",
            json={"commands": commands, "agent_name": agent_name},
        )
        return response.json()["message"]

    def delete_agent(self, agent_name: str) -> str:
        response = requests.delete(f"{self.base_uri}/api/agent/{agent_name}")
        return response.json()["message"]

    def get_agents(
        self,
    ) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_uri}/api/agent")
        return response.json()["agents"]

    def get_agentconfig(self, agent_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/agent/{agent_name}")
        return response.json()["agent"]

    def get_chat_history(self, agent_name: str) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.base_uri}/api/{agent_name}/chat")
        return response.json()["chat_history"]

    def delete_agent_history(self, agent_name: str) -> str:
        response = requests.delete(f"{self.base_uri}/api/agent/{agent_name}/history")
        return response.json()["message"]

    def delete_history_message(self, agent_name: str, message: str) -> str:
        response = requests.delete(
            f"{self.base_uri}/api/agent/{agent_name}/history/message",
            json={"message": message},
        )
        return response.json()["message"]

    def wipe_agent_memories(self, agent_name: str) -> str:
        response = requests.delete(f"{self.base_uri}/api/agent/{agent_name}/memory")
        return response.json()["message"]

    def prompt_agent(
        self,
        agent_name: str,
        prompt_name: str,
        prompt_args: dict,
        user_input: str = "",
        websearch: bool = False,
        websearch_depth: int = 3,
        context_results: int = 5,
        shots: int = 1,
    ) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/prompt",
            json={
                "user_input": user_input,
                "prompt_name": prompt_name,
                "prompt_args": prompt_args,
                "websearch": websearch,
                "websearch_depth": websearch_depth,
                "context_results": context_results,
                "shots": shots,
            },
        )
        return response.json()["response"]

    def instruct(self, agent_name: str, prompt: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/instruct",
            json={"prompt": prompt},
        )
        return response.json()["response"]

    def smartinstruct(self, agent_name: str, shots: int, prompt: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/smartinstruct/{shots}",
            json={"prompt": prompt},
        )
        return response.json()["response"]

    def chat(self, agent_name: str, prompt: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/chat",
            json={"prompt": prompt},
        )
        return response.json()["response"]

    def smartchat(self, agent_name: str, shots: int, prompt: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/smartchat/{shots}",
            json={"prompt": prompt},
        )
        return response.json()["response"]

    def get_commands(self, agent_name: str) -> Dict[str, Dict[str, bool]]:
        response = requests.get(f"{self.base_uri}/api/agent/{agent_name}/command")
        return response.json()["commands"]

    def toggle_command(self, agent_name: str, command_name: str, enable: bool) -> str:
        response = requests.patch(
            f"{self.base_uri}/api/agent/{agent_name}/command",
            json={"command_name": command_name, "enable": enable},
        )
        return response.json()["message"]

    def get_chains(self) -> List[str]:
        response = requests.get(f"{self.base_uri}/api/chain")
        return response.json()

    def get_chain(self, chain_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/chain/{chain_name}")
        return response.json()["chain"]

    def get_chain_responses(self, chain_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/chain/{chain_name}/responses")
        return response.json()["chain"]

    def run_chain(
        self,
        chain_name: str,
        user_input: str,
        agent_name: str = "",
        all_responses: bool = False,
        from_step: int = 1,
    ) -> str:
        response = requests.post(
            f"{self.base_uri}/api/chain/{chain_name}/run",
            json={
                "prompt": user_input,
                "agent_override": agent_name,
                "all_responses": all_responses,
                "from_step": int(from_step),
            },
        )
        return response.json()

    def add_chain(self, chain_name: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/chain",
            json={"chain_name": chain_name},
        )
        return response.json()["message"]

    def import_chain(self, chain_name: str, steps: dict) -> str:
        response = requests.post(
            f"{self.base_uri}/api/chain/import",
            json={"chain_name": chain_name, "steps": steps},
        )
        return response.json()["message"]

    def rename_chain(self, chain_name: str, new_name: str) -> str:
        response = requests.put(
            f"{self.base_uri}/api/chain/{chain_name}",
            json={"new_name": new_name},
        )
        return response.json()["message"]

    def delete_chain(self, chain_name: str) -> str:
        response = requests.delete(f"{self.base_uri}/api/chain/{chain_name}")
        return response.json()["message"]

    def add_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        response = requests.post(
            f"{self.base_uri}/api/chain/{chain_name}/step",
            json={
                "step_number": step_number,
                "agent_name": agent_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
            },
        )
        return response.json()["message"]

    def update_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        response = requests.put(
            f"{self.base_uri}/api/chain/{chain_name}/step/{step_number}",
            json={
                "step_number": step_number,
                "agent_name": agent_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
            },
        )
        return response.json()["message"]

    def move_step(
        self,
        chain_name: str,
        old_step_number: int,
        new_step_number: int,
    ) -> str:
        response = requests.patch(
            f"{self.base_uri}/api/chain/{chain_name}/step/move",
            json={
                "old_step_number": old_step_number,
                "new_step_number": new_step_number,
            },
        )
        return response.json()["message"]

    def delete_step(self, chain_name: str, step_number: int) -> str:
        response = requests.delete(
            f"{self.base_uri}/api/chain/{chain_name}/step/{step_number}"
        )
        return response.json()["message"]

    def add_prompt(self, prompt_name: str, prompt: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/prompt",
            json={"prompt_name": prompt_name, "prompt": prompt},
        )
        return response.json()["message"]

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/prompt/{prompt_name}")
        return response.json()["prompt"]

    def get_prompts(self) -> List[str]:
        response = requests.get(f"{self.base_uri}/api/prompt")
        return response.json()["prompts"]

    def get_prompt_args(self, prompt_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/prompt/{prompt_name}/args")
        return response.json()["prompt_args"]

    def delete_prompt(self, prompt_name: str) -> str:
        response = requests.delete(f"{self.base_uri}/api/prompt/{prompt_name}")
        return response.json()["message"]

    def update_prompt(self, prompt_name: str, prompt: str) -> str:
        response = requests.put(
            f"{self.base_uri}/api/prompt/{prompt_name}",
            json={"prompt": prompt, "prompt_name": prompt_name},
        )
        return response.json()["message"]

    def get_extension_settings(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/extensions/settings")
        return response.json()["extension_settings"]

    def get_extensions(self) -> List[tuple]:
        response = requests.get(f"{self.base_uri}/api/extensions")
        return response.json()["extensions"]

    def get_command_args(self, command_name: str) -> Dict[str, Any]:
        response = requests.get(f"{self.base_uri}/api/extensions/{command_name}/args")
        return response.json()["command_args"]

    def learn_url(self, agent_name: str, url: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/learn/url",
            json={"url": url},
        )
        return response.json()["message"]

    def learn_file(self, agent_name: str, file_name: str, file_content: str) -> str:
        response = requests.post(
            f"{self.base_uri}/api/agent/{agent_name}/learn/file",
            json={"file_name": file_name, "file_content": file_content},
        )
        return response.json()["message"]
