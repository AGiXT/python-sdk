import requests
from typing import Dict, List, Any


class AGiXTSDK:
    def __init__(self, base_uri: str = None, api_key: str = None):
        if not base_uri:
            base_uri = "http://localhost:7437"
        self.base_uri = base_uri
        if api_key:
            self.headers = {"Authorization": f"Bearer {api_key}"}
        else:
            self.headers = {}
        if self.base_uri[-1] == "/":
            self.base_uri = self.base_uri[:-1]

    def handle_error(self, response: requests.Response) -> str:
        return "Unable to retrieve data."

    def agixt_request(self, endpoint: str, method: str, data: Dict[str, Any] = {}):
        try:
            if method == "GET":
                response = requests.get(
                    f"{self.base_uri}/api/{endpoint}", headers=self.headers
                )
            elif method == "POST":
                response = requests.post(
                    f"{self.base_uri}/api/{endpoint}", json=data, headers=self.headers
                )
            elif method == "PUT":
                response = requests.put(
                    f"{self.base_uri}/api/{endpoint}", json=data, headers=self.headers
                )
            elif method == "PATCH":
                response = requests.patch(
                    f"{self.base_uri}/api/{endpoint}", json=data, headers=self.headers
                )
            elif method == "DELETE":
                response = requests.delete(
                    f"{self.base_uri}/api/{endpoint}", headers=self.headers
                )
            return response.json()
        except requests.RequestException:
            return self.handle_error(response)

    def get_providers(self) -> List[str]:
        return self.agixt_request(endpoint="provider", method="GET")["providers"]

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        return self.agixt_request(endpoint=f"provider/{provider_name}", method="GET")[
            "settings"
        ]

    def get_embed_providers(self) -> List[str]:
        return self.agixt_request(endpoint="embedding_providers", method="GET")[
            "providers"
        ]

    def add_agent(
        self, agent_name: str, settings: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        return self.agixt_request(
            endpoint="agent",
            method="POST",
            data={"agent_name": agent_name, "settings": settings},
        )

    def import_agent(
        self,
        agent_name: str,
        settings: Dict[str, Any] = {},
        commands: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        return self.agixt_request(
            endpoint="agent/import",
            method="POST",
            data={
                "agent_name": agent_name,
                "settings": settings,
                "commands": commands,
            },
        )

    def rename_agent(self, agent_name: str, new_name: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}",
            method="PATCH",
            data={"new_name": new_name},
        )

    def update_agent_settings(self, agent_name: str, settings: Dict[str, Any]) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}",
            method="PUT",
            data={"settings": settings, "agent_name": agent_name},
        )["message"]

    def update_agent_commands(self, agent_name: str, commands: Dict[str, Any]) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/commands",
            method="PUT",
            data={"commands": commands, "agent_name": agent_name},
        )["message"]

    def delete_agent(self, agent_name: str) -> str:
        return self.agixt_request(endpoint=f"agent/{agent_name}", method="DELETE")[
            "message"
        ]

    def get_agents(self) -> List[Dict[str, Any]]:
        return self.agixt_request(endpoint="agent", method="GET")["agents"]

    def get_agentconfig(self, agent_name: str) -> Dict[str, Any]:
        return self.agixt_request(endpoint=f"agent/{agent_name}", method="GET")["agent"]

    def get_chat_history(self, agent_name: str) -> List[Dict[str, Any]]:
        return self.agixt_request(endpoint=f"{agent_name}/chat", method="GET")[
            "chat_history"
        ]

    def delete_agent_history(self, agent_name: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/history",
            method="DELETE",
        )["message"]

    def delete_history_message(self, agent_name: str, message: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/history/message",
            method="DELETE",
            data={"message": message},
        )["message"]

    def wipe_agent_memories(self, agent_name: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/memory",
            method="DELETE",
        )["message"]

    def prompt_agent(self, agent_name: str, prompt_name: str, prompt_args: dict) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/prompt",
            method="POST",
            data={
                "prompt_name": prompt_name,
                "prompt_args": prompt_args,
            },
        )["response"]

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
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/command",
            method="GET",
        )["commands"]

    def toggle_command(self, agent_name: str, command_name: str, enable: bool) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/command",
            method="PATCH",
            data={"command_name": command_name, "enable": enable},
        )["message"]

    def get_chains(self) -> List[str]:
        return self.agixt_request(endpoint="chain", method="GET")

    def get_chain(self, chain_name: str) -> Dict[str, Any]:
        return self.agixt_request(endpoint=f"chain/{chain_name}", method="GET")["chain"]

    def get_chain_responses(self, chain_name: str) -> Dict[str, Any]:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/responses",
            method="GET",
        )["chain"]

    def run_chain(
        self,
        chain_name: str,
        user_input: str,
        agent_name: str = "",
        all_responses: bool = False,
        from_step: int = 1,
    ) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/run",
            method="POST",
            data={
                "prompt": user_input,
                "agent_override": agent_name,
                "all_responses": all_responses,
                "from_step": int(from_step),
            },
        )

    def add_chain(self, chain_name: str) -> str:
        return self.agixt_request(
            endpoint="chain",
            method="POST",
            data={"chain_name": chain_name},
        )["message"]

    def import_chain(self, chain_name: str, steps: dict) -> str:
        return self.agixt_request(
            endpoint="chain/import",
            method="POST",
            data={"chain_name": chain_name, "steps": steps},
        )["message"]

    def rename_chain(self, chain_name: str, new_name: str) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}",
            method="PUT",
            data={"new_name": new_name},
        )["message"]

    def delete_chain(self, chain_name: str) -> str:
        return self.agixt_request(endpoint=f"chain/{chain_name}", method="DELETE")[
            "message"
        ]

    def add_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/step",
            method="POST",
            data={
                "step_number": step_number,
                "agent_name": agent_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
            },
        )["message"]

    def update_step(
        self,
        chain_name: str,
        step_number: int,
        agent_name: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/step/{step_number}",
            method="PUT",
            data={
                "step_number": step_number,
                "agent_name": agent_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
            },
        )["message"]

    def move_step(
        self,
        chain_name: str,
        old_step_number: int,
        new_step_number: int,
    ) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/step/move",
            method="PATCH",
            data={
                "old_step_number": old_step_number,
                "new_step_number": new_step_number,
            },
        )["message"]

    def delete_step(self, chain_name: str, step_number: int) -> str:
        return self.agixt_request(
            endpoint=f"chain/{chain_name}/step/{step_number}",
            method="DELETE",
        )["message"]

    def add_prompt(self, prompt_name: str, prompt: str) -> str:
        return self.agixt_request(
            endpoint="prompt",
            method="POST",
            data={"prompt_name": prompt_name, "prompt": prompt},
        )["message"]

    def get_prompt(self, prompt_name: str) -> Dict[str, Any]:
        return self.agixt_request(endpoint=f"prompt/{prompt_name}", method="GET")[
            "prompt"
        ]

    def get_prompts(self) -> List[str]:
        return self.agixt_request(endpoint="prompt", method="GET")["prompts"]

    def get_prompt_args(self, prompt_name: str) -> Dict[str, Any]:
        return self.agixt_request(
            endpoint=f"prompt/{prompt_name}/args",
            method="GET",
        )["prompt_args"]

    def delete_prompt(self, prompt_name: str) -> str:
        return self.agixt_request(endpoint=f"prompt/{prompt_name}", method="DELETE")[
            "message"
        ]

    def update_prompt(self, prompt_name: str, prompt: str) -> str:
        return self.agixt_request(
            endpoint=f"prompt/{prompt_name}",
            method="PUT",
            data={"prompt": prompt, "prompt_name": prompt_name},
        )["message"]

    def get_extension_settings(self) -> Dict[str, Any]:
        return self.agixt_request(endpoint="extensions/settings", method="GET")[
            "extension_settings"
        ]

    def get_extensions(self):
        return self.agixt_request(endpoint="extensions", method="GET")["extensions"]

    def get_command_args(self, command_name: str) -> Dict[str, Any]:
        return self.agixt_request(
            endpoint=f"extensions/{command_name}/args", method="GET"
        )["command_args"]

    def learn_url(self, agent_name: str, url: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/learn/url",
            method="POST",
            data={"url": url},
        )["message"]

    def learn_file(self, agent_name: str, file_name: str, file_content: str) -> str:
        return self.agixt_request(
            endpoint=f"agent/{agent_name}/learn/file",
            method="POST",
            data={"file_name": file_name, "file_content": file_content},
        )["message"]
