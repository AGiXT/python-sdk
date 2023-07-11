import requests
from typing import Dict, List, Any


class AGiXTSDK:
    def __init__(self, base_uri: str = None, api_key: str = None):
        if not base_uri:
            base_uri = "http://localhost:7437"
        self.base_uri = base_uri
        if not api_key:
            self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        if self.base_uri[-1] == "/":
            self.base_uri = self.base_uri[:-1]

    def handle_error(self, error) -> str:
        print(f"Error: {error}")
        return f"Unable to retrieve data."

    def get_providers(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/provider"
            )
            return response.json()["providers"]
        except Exception as e:
            return self.handle_error(e)

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/provider/{provider_name}",
            )
            return response.json()["settings"]
        except Exception as e:
            return self.handle_error(e)

    def get_embed_providers(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedding_providers"
            )
            return response.json()["providers"]
        except Exception as e:
            return self.handle_error(e)

    def add_agent(
        self, agent_name: str, settings: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent",
                json={"agent_name": agent_name, "settings": settings},
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def import_agent(
        self,
        agent_name: str,
        settings: Dict[str, Any] = {},
        commands: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/import",
                json={
                    "agent_name": agent_name,
                    "settings": settings,
                    "commands": commands,
                },
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def rename_agent(self, agent_name: str, new_name: str) -> str:
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}",
                json={"new_name": new_name},
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def update_agent_settings(self, agent_name: str, settings: Dict[str, Any]) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/agent/{agent_name}",
                json={"settings": settings, "agent_name": agent_name},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_agent_commands(self, agent_name: str, commands: Dict[str, Any]) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/agent/{agent_name}/commands",
                json={"commands": commands, "agent_name": agent_name},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_agent(self, agent_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/api/agent/{agent_name}"
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_agents(
        self,
    ) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/agent"
            )
            return response.json()["agents"]
        except Exception as e:
            return self.handle_error(e)

    def get_agentconfig(self, agent_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/agent/{agent_name}"
            )
            return response.json()["agent"]
        except Exception as e:
            return self.handle_error(e)

    def get_conversations(self, agent_name: str) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/{agent_name}/conversations",
            )
            return response.json()["conversations"]
        except Exception as e:
            return self.handle_error(e)

    def get_conversation(
        self, agent_name: str, conversation_name: str, limit: int = 100, page: int = 1
    ) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation",
                json={
                    "conversation_name": conversation_name,
                    "agent_name": agent_name,
                    "limit": limit,
                    "page": page,
                },
            )
            return response.json()["conversation_history"]
        except Exception as e:
            return self.handle_error(e)

    def new_conversation(self, agent_name: str, conversation_name: str):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation",
                json={
                    "conversation_name": conversation_name,
                    "agent_name": agent_name,
                },
            )
            return response.json()["conversation_history"]
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation(self, agent_name: str, conversation_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation",
                json={
                    "conversation_name": conversation_name,
                    "agent_name": agent_name,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation_message(
        self, agent_name: str, conversation_name: str, message: str
    ) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/message",
                json={
                    "message": message,
                    "agent_name": agent_name,
                    "conversation_name": conversation_name,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def wipe_agent_memories(self, agent_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory",
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def prompt_agent(
        self,
        agent_name: str,
        prompt_name: str,
        prompt_args: dict,
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/prompt",
                json={
                    "prompt_name": prompt_name,
                    "prompt_args": prompt_args,
                },
            )
            return response.json()["response"]
        except Exception as e:
            return self.handle_error(e)

    def instruct(self, agent_name: str, prompt: str) -> str:
        return self.prompt_agent(
            agent_name=agent_name,
            prompt_name="instruct",
            prompt_args={"user_input": prompt},
        )

    def chat(self, agent_name: str, prompt: str) -> str:
        return self.prompt_agent(
            agent_name=agent_name,
            prompt_name="Chat",
            prompt_args={"user_input": prompt, "context_reslts": 4},
        )

    def smartinstruct(self, agent_name: str, prompt: str) -> str:
        return self.run_chain(
            chain_name="Smart Instruct",
            user_input=prompt,
            agent_name=agent_name,
            all_responses=False,
            from_step=1,
        )

    def smartchat(self, agent_name: str, prompt: str) -> str:
        return self.run_chain(
            chain_name="Smart Chat",
            user_input=prompt,
            agent_name=agent_name,
            all_responses=False,
            from_step=1,
        )

    def get_commands(self, agent_name: str) -> Dict[str, Dict[str, bool]]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/command",
            )
            return response.json()["commands"]
        except Exception as e:
            return self.handle_error(e)

    def toggle_command(self, agent_name: str, command_name: str, enable: bool) -> str:
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/command",
                json={"command_name": command_name, "enable": enable},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_chains(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/chain"
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_chain(self, chain_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/chain/{chain_name}"
            )
            return response.json()["chain"]
        except Exception as e:
            return self.handle_error(e)

    def get_chain_responses(self, chain_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/responses",
            )
            return response.json()["chain"]
        except Exception as e:
            return self.handle_error(e)

    def run_chain(
        self,
        chain_name: str,
        user_input: str,
        agent_name: str = "",
        all_responses: bool = False,
        from_step: int = 1,
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/run",
                json={
                    "prompt": user_input,
                    "agent_override": agent_name,
                    "all_responses": all_responses,
                    "from_step": int(from_step),
                },
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def add_chain(self, chain_name: str) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain",
                json={"chain_name": chain_name},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def import_chain(self, chain_name: str, steps: dict) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/import",
                json={"chain_name": chain_name, "steps": steps},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def rename_chain(self, chain_name: str, new_name: str) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/chain/{chain_name}",
                json={"new_name": new_name},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_chain(self, chain_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/api/chain/{chain_name}"
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def add_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/step",
                json={
                    "step_number": step_number,
                    "agent_name": agent_name,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        try:
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
        except Exception as e:
            return self.handle_error(e)

    def move_step(
        self,
        chain_name: str,
        old_step_number: int,
        new_step_number: int,
    ) -> str:
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/step/move",
                json={
                    "old_step_number": old_step_number,
                    "new_step_number": new_step_number,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_step(self, chain_name: str, step_number: int) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/step/{step_number}",
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def add_prompt(self, prompt_name: str, prompt: str) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt",
                json={"prompt_name": prompt_name, "prompt": prompt},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/prompt/{prompt_name}"
            )
            return response.json()["prompt"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompts(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/prompt"
            )
            return response.json()["prompts"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_args(self, prompt_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_name}/args",
            )
            return response.json()["prompt_args"]
        except Exception as e:
            return self.handle_error(e)

    def delete_prompt(self, prompt_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/api/prompt/{prompt_name}"
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_prompt(self, prompt_name: str, prompt: str) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/prompt/{prompt_name}",
                json={"prompt": prompt, "prompt_name": prompt_name},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_extension_settings(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/extensions/settings"
            )
            return response.json()["extension_settings"]
        except Exception as e:
            return self.handle_error(e)

    def get_extensions(self):
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/extensions"
            )
            return response.json()["extensions"]
        except Exception as e:
            return self.handle_error(e)

    def get_command_args(self, command_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/extensions/{command_name}/args",
            )
            return response.json()["command_args"]
        except Exception as e:
            return self.handle_error(e)

    def learn_url(self, agent_name: str, url: str) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/url",
                json={"url": url},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_file(self, agent_name: str, file_name: str, file_content: str) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/file",
                json={"file_name": file_name, "file_content": file_content},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)
