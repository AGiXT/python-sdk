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
            api_key = api_key.replace("Bearer ", "").replace("bearer ", "")
            self.headers = {
                "Authorization": f"{api_key}",
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

    def get_providers_by_service(self, service: str) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/providers/service/{service}",
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

    def get_embedders(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedders"
            )
            return response.json()["embedders"]
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
                headers=self.headers,
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_agent_commands(self, agent_name: str, commands: Dict[str, Any]) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/agent/{agent_name}/commands",
                json={"commands": commands, "agent_name": agent_name},
                headers=self.headers,
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

    def get_conversations(self, agent_name: str = "") -> List[str]:
        if agent_name == "":
            url = f"{self.base_uri}/api/conversations"
        else:
            url = f"{self.base_uri}/api/{agent_name}/conversations"
        try:
            response = requests.get(
                headers=self.headers,
                url=url,
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

    def new_conversation(
        self,
        agent_name: str,
        conversation_name: str,
        conversation_content: List[Dict[str, Any]] = [],
    ) -> List[Dict[str, Any]]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation",
                json={
                    "conversation_name": conversation_name,
                    "agent_name": agent_name,
                    "conversation_content": conversation_content,
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

    def update_conversation_message(
        self, agent_name: str, conversation_name: str, message: str, new_message: str
    ) -> str:
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/message",
                json={
                    "message": message,
                    "new_message": new_message,
                    "agent_name": agent_name,
                    "conversation_name": conversation_name,
                },
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

    def instruct(self, agent_name: str, user_input: str, conversation: str) -> str:
        return self.prompt_agent(
            agent_name=agent_name,
            prompt_name="instruct",
            prompt_args={
                "user_input": user_input,
                "disable_memory": True,
                "conversation_name": conversation,
            },
        )

    def chat(
        self,
        agent_name: str,
        user_input: str,
        conversation: str,
        context_results: int = 4,
    ) -> str:
        return self.prompt_agent(
            agent_name=agent_name,
            prompt_name="Chat",
            prompt_args={
                "user_input": user_input,
                "context_results": context_results,
                "conversation_name": conversation,
                "disable_memory": True,
            },
        )

    def smartinstruct(self, agent_name: str, user_input: str, conversation: str) -> str:
        return self.run_chain(
            chain_name="Smart Instruct",
            user_input=user_input,
            agent_name=agent_name,
            all_responses=False,
            from_step=1,
            chain_args={
                "conversation_name": conversation,
                "disable_memory": True,
            },
        )

    def smartchat(self, agent_name: str, user_input: str, conversation: str) -> str:
        return self.run_chain(
            chain_name="Smart Chat",
            user_input=user_input,
            agent_name=agent_name,
            all_responses=False,
            from_step=1,
            chain_args={
                "conversation_name": conversation,
                "disable_memory": True,
            },
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

    def execute_command(
        self,
        agent_name: str,
        command_name: str,
        command_args: dict,
        conversation_name: str = "AGiXT Terminal Command Execution",
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/command",
                json={
                    "command_name": command_name,
                    "command_args": command_args,
                    "conversation_name": conversation_name,
                },
            )
            return response.json()["response"]
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

    def get_chain_args(self, chain_name: str) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/args",
            )
            return response.json()["chain_args"]
        except Exception as e:
            return self.handle_error(e)

    def run_chain(
        self,
        chain_name: str,
        user_input: str,
        agent_name: str = "",
        all_responses: bool = False,
        from_step: int = 1,
        chain_args={},
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
                    "chain_args": chain_args,
                },
            )
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def run_chain_step(
        self,
        chain_name: str,
        step_number: int,
        user_input: str,
        agent_name=None,
        chain_args={},
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/run/step/{step_number}",
                json={
                    "prompt": user_input,
                    "agent_override": agent_name,
                    "chain_args": chain_args,
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

    def add_prompt(
        self, prompt_name: str, prompt: str, prompt_category: str = "Default"
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}",
                json={
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt(
        self, prompt_name: str, prompt_category: str = "Default"
    ) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}",
            )
            return response.json()["prompt"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompts(self, prompt_category: str = "Default") -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}",
            )
            return response.json()["prompts"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_categories(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/categories",
            )
            return response.json()["prompt_categories"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_args(
        self, prompt_name: str, prompt_category: str = "Default"
    ) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}/args",
            )
            return response.json()["prompt_args"]
        except Exception as e:
            return self.handle_error(e)

    def delete_prompt(self, prompt_name: str, prompt_category: str = "Default") -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}",
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_prompt(
        self, prompt_name: str, prompt: str, prompt_category: str = "Default"
    ) -> str:
        try:
            response = requests.put(
                f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}",
                json={
                    "prompt": prompt,
                    "prompt_name": prompt_name,
                    "prompt_category": prompt_category,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def rename_prompt(
        self, prompt_name: str, new_name: str, prompt_category: str = "Default"
    ) -> str:
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}",
                json={"prompt_name": new_name},
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

    def get_embedders_details(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedders"
            )
            return response.json()["embedders"]
        except Exception as e:
            return self.handle_error(e)

    def learn_text(
        self, agent_name, user_input: str, text: str, collection_number: int = 0
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/text",
                json={
                    "user_input": user_input,
                    "text": text,
                    "collection_number": collection_number,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_url(self, agent_name: str, url: str, collection_number: int = 0) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/url",
                json={
                    "url": url,
                    "collection_number": collection_number,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_file(
        self,
        agent_name: str,
        file_name: str,
        file_content: str,
        collection_number: int = 0,
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/file",
                json={
                    "file_name": file_name,
                    "file_content": file_content,
                    "collection_number": collection_number,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_github_repo(
        self,
        agent_name: str,
        github_repo: str,
        github_user: str = None,
        github_token: str = None,
        github_branch: str = "main",
        use_agent_settings: bool = False,
        collection_number: int = 0,
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/github",
                json={
                    "github_repo": github_repo,
                    "github_user": github_user,
                    "github_token": github_token,
                    "github_branch": github_branch,
                    "collection_number": collection_number,
                    "use_agent_settings": use_agent_settings,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_arxiv(
        self,
        agent_name: str,
        query: str = None,
        arxiv_ids: str = None,
        max_results: int = 5,
        collection_number: int = 0,
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/learn/arxiv",
                json={
                    "query": query,
                    "arxiv_ids": arxiv_ids,
                    "max_results": max_results,
                    "collection_number": collection_number,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def agent_reader(
        self,
        agent_name: str,
        reader_name: str,
        data: dict,
        collection_number: int = 0,
    ):
        if "collection_number" not in data:
            data["collection_number"] = collection_number
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/reader/{reader_name}",
                json=data,
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def wipe_agent_memories(self, agent_name: str, collection_number: int = 0) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=(
                    f"{self.base_uri}/api/agent/{agent_name}/memory"
                    if collection_number == 0
                    else f"{self.base_uri}/api/agent/{agent_name}/memory/{collection_number}"
                ),
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_agent_memory(
        self, agent_name: str, memory_id: str, collection_number: int = 0
    ) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/{collection_number}/{memory_id}",
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_agent_memories(
        self,
        agent_name: str,
        user_input: str,
        limit: int = 5,
        min_relevance_score: float = 0.0,
        collection_number: int = 0,
    ) -> List[Dict[str, Any]]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/{collection_number}/query",
                json={
                    "user_input": user_input,
                    "limit": limit,
                    "min_relevance_score": min_relevance_score,
                },
            )
            return response.json()["memories"]
        except Exception as e:
            return self.handle_error(e)

    def export_agent_memories(self, agent_name: str) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/export",
            )
            return response.json()["memories"]
        except Exception as e:
            return self.handle_error(e)

    def import_agent_memories(
        self, agent_name: str, memories: List[Dict[str, Any]]
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/import",
                json={"memories": memories},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def create_dataset(self, agent_name: str, dataset_name: str, batch_size: int = 4):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/dataset",
                json={"dataset_name": dataset_name, "batch_size": batch_size},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def train(
        self,
        agent_name: str = "AGiXT",
        dataset_name: str = "dataset",
        model: str = "unsloth/mistral-7b-v0.2",
        max_seq_length: int = 16384,
        huggingface_output_path: str = "JoshXT/finetuned-mistral-7b-v0.2",
        private_repo: bool = True,
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/dataset/{dataset_name}/finetune",
                json={
                    "model": model,
                    "max_seq_length": max_seq_length,
                    "huggingface_output_path": huggingface_output_path,
                    "private_repo": private_repo,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def text_to_speech(self, agent_name: str, text: str, conversation_name: str):
        response = self.execute_command(
            agent_name=agent_name,
            command_name="Text to Speech",
            command_args={"text": text},
            conversation_name=conversation_name,
        )
        return response
