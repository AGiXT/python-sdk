import tiktoken
import uuid
import requests
import base64
import time
import openai
import requests
from datetime import datetime
from pydub import AudioSegment
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Callable


class ChatCompletions(BaseModel):
    model: str = "gpt4free"  # This is the agent name
    messages: List[dict] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 1.0
    tools: Optional[List[dict]] = None
    tools_choice: Optional[str] = "auto"
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 4096
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = "Chat"  # This is the conversation name


def get_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


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
        raise Exception(f"Unable to retrieve data. {error}")

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
        self,
        agent_name: str,
        settings: Dict[str, Any] = {},
        commands: Dict[str, Any] = {},
        training_urls: List[str] = [],
    ) -> Dict[str, Any]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent",
                json={
                    "agent_name": agent_name,
                    "settings": settings,
                    "commands": commands,
                    "training_urls": training_urls,
                },
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
        url = f"{self.base_uri}/api/conversations"
        try:
            response = requests.get(
                headers=self.headers,
                url=url,
            )
            return response.json()["conversations"]
        except Exception as e:
            return self.handle_error(e)

    def get_conversations_with_ids(self) -> List[str]:
        url = f"{self.base_uri}/api/conversations"
        try:
            response = requests.get(
                headers=self.headers,
                url=url,
            )
            return response.json()["conversations_with_ids"]
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

    def rename_conversation(
        self,
        agent_name: str,
        conversation_name: str,
        new_name: str = "-",
    ):
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation",
                json={
                    "conversation_name": conversation_name,
                    "new_conversation_name": new_name,
                    "agent_name": agent_name,
                },
            )
            return response.json()["conversation_name"]
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

    def new_conversation_message(
        self,
        role: str = "user",
        message: str = "",
        conversation_name: str = "",
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/message",
                json={
                    "role": role,
                    "message": message,
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
                headers=self.headers,
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
                headers=self.headers,
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

    def positive_feedback(
        self,
        agent_name,
        message: str,
        user_input: str,
        feedback: str,
        conversation_name: str = "",
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/feedback",
                json={
                    "user_input": user_input,
                    "message": message,
                    "feedback": feedback,
                    "positive": True,
                    "conversation_name": conversation_name,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def negative_feedback(
        self,
        agent_name,
        message: str,
        user_input: str,
        feedback: str,
        conversation_name: str = "",
    ) -> str:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/feedback",
                json={
                    "user_input": user_input,
                    "message": message,
                    "feedback": feedback,
                    "positive": False,
                    "conversation_name": conversation_name,
                },
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_text(
        self, agent_name, user_input: str, text: str, collection_number: str = "0"
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

    def learn_url(self, agent_name: str, url: str, collection_number: str = "0") -> str:
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
        collection_number: str = "0",
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
        collection_number: str = "0",
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
        collection_number: str = "0",
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
        collection_number: str = "0",
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

    def wipe_agent_memories(self, agent_name: str, collection_number: str = "0") -> str:
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
        self,
        agent_name: str,
        memory_id: str,
        collection_number: str = "0",
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
        collection_number: str = "0",
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

    def get_browsed_links(
        self, agent_name: str, collection_number: str = "0"
    ) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/browsed_links/{collection_number}",
            )
            return response.json()["links"]
        except Exception as e:
            return self.handle_error(e)

    def delete_browsed_link(
        self, agent_name: str, link: str, collection_number: str = "0"
    ) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/browsed_links",
                json={"link": link, "collection_number": collection_number},
            )
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_memories_external_sources(self, agent_name: str, collection_number: str):
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/external_sources/{collection_number}",
            )
            return response.json()["external_sources"]
        except Exception as e:
            return self.handle_error(e)

    def delete_memory_external_source(
        self, agent_name: str, source: str, collection_number: str
    ) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/external_source",
                json={
                    "external_source": source,
                    "collection_number": collection_number,
                },
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

    def text_to_speech(self, agent_name: str, text: str):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/text_to_speech",
                json={"text": text},
            )
            return response.json()["url"]
        except Exception as e:
            return self.handle_error(e)

    # Chat Completion wrapper
    async def chat_completions(
        self,
        prompt: ChatCompletions,
        func: Callable = None,
        async_func: Callable = None,
    ):
        agent_name = prompt.model  # prompt.model is the agent name
        conversation_name = (
            prompt.user if prompt.user else "-"
        )  # prompt.user is the conversation name
        agent_config = self.get_agentconfig(agent_name=agent_name)
        agent_settings = agent_config["settings"] if "settings" in agent_config else {}
        images = []
        tts = False
        if "tts_provider" in agent_settings:
            tts_provider = str(agent_settings["tts_provider"]).lower()
            if tts_provider != "none" and tts_provider != "":
                tts = True
        new_prompt = ""
        for message in prompt.messages:
            if "tts" in message:
                tts = message["tts"].lower() == "true"
            if "content" not in message:
                continue
            if isinstance(message["content"], str):
                role = message["role"] if "role" in message else "User"
                if role.lower() == "system":
                    if "/" in message["content"]:
                        new_prompt += f"{message['content']}\n\n"
                if role.lower() == "user":
                    new_prompt += f"{message['content']}\n\n"
            if isinstance(message["content"], list):
                for msg in message["content"]:
                    if "text" in msg:
                        role = message["role"] if "role" in message else "User"
                        if role.lower() == "user":
                            new_prompt += f"{msg['text']}\n\n"
                    if "image_url" in msg:
                        url = (
                            msg["image_url"]["url"]
                            if "url" in msg["image_url"]
                            else msg["image_url"]
                        )
                        image_path = f"./WORKSPACE/{uuid.uuid4().hex}.jpg"
                        if url.startswith("http"):
                            image = requests.get(url).content
                        else:
                            file_type = url.split(",")[0].split("/")[1].split(";")[0]
                            if file_type == "jpeg":
                                file_type = "jpg"
                            file_name = f"{uuid.uuid4().hex}.{file_type}"
                            image_path = f"./WORKSPACE/{file_name}"
                            image = base64.b64decode(url.split(",")[1])
                        with open(image_path, "wb") as f:
                            f.write(image)
                        images.append(image_path)
                    if "audio_url" in msg:
                        audio_url = (
                            msg["audio_url"]["url"]
                            if "url" in msg["audio_url"]
                            else msg["audio_url"]
                        )
                        # If it is not a url, we need to find the file type and convert with pydub
                        if not audio_url.startswith("http"):
                            file_type = (
                                audio_url.split(",")[0].split("/")[1].split(";")[0]
                            )
                            audio_data = base64.b64decode(audio_url.split(",")[1])
                            audio_path = f"./WORKSPACE/{uuid.uuid4().hex}.{file_type}"
                            with open(audio_path, "wb") as f:
                                f.write(audio_data)
                            audio_url = audio_path
                        else:
                            # Download the audio file from the url, get the file type and convert to wav
                            audio_type = audio_url.split(".")[-1]
                            audio_url = f"./WORKSPACE/{uuid.uuid4().hex}.{audio_type}"
                            audio_data = requests.get(audio_url).content
                            with open(audio_url, "wb") as f:
                                f.write(audio_data)
                        wav_file = f"./WORKSPACE/{uuid.uuid4().hex}.wav"
                        AudioSegment.from_file(audio_url).set_frame_rate(16000).export(
                            wav_file, format="wav"
                        )
                        # Switch this to use the endpoint
                        openai.api_key = (
                            self.headers["Authorization"]
                            .replace("Bearer ", "")
                            .replace("bearer ", "")
                        )
                        openai.base_url = f"{self.base_uri}/v1/"
                        self.new_conversation_message(
                            role=agent_name,
                            message=f"[ACTIVITY] Transcribing audio to text.",
                            conversation_name=conversation_name,
                        )
                        with open(wav_file, "rb") as audio_file:
                            transcription = openai.audio.transcriptions.create(
                                model=agent_name, file=audio_file
                            )
                        new_prompt += transcription.text
                    if "video_url" in msg:
                        video_url = str(
                            msg["video_url"]["url"]
                            if "url" in msg["video_url"]
                            else msg["video_url"]
                        )
                        if "collection_number" in msg:
                            collection_number = str(msg["collection_number"])
                        else:
                            collection_number = "0"
                        if video_url.startswith("https://www.youtube.com/watch?v="):
                            self.new_conversation_message(
                                role=agent_name,
                                message=f"[ACTIVITY] Learning video from YouTube.",
                                conversation_name=conversation_name,
                            )
                            self.learn_url(
                                agent_name=agent_name,
                                url=video_url,
                                collection_number=collection_number,
                            )
                    if (
                        "file_url" in msg
                        or "application_url" in msg
                        or "text_url" in msg
                        or "url" in msg
                    ):
                        file_url = str(
                            msg["file_url"]["url"]
                            if "url" in msg["file_url"]
                            else msg["file_url"]
                        )
                        if "collection_number" in message or "collection_number" in msg:
                            collection_number = str(
                                message["collection_number"]
                                if "collection_number" in message
                                else msg["collection_number"]
                            )
                        else:
                            collection_number = "0"
                        if file_url.startswith("http"):
                            if file_url.startswith("https://www.youtube.com/watch?v="):
                                self.new_conversation_message(
                                    role=agent_name,
                                    message=f"[ACTIVITY] Learning video from YouTube.",
                                    conversation_name=conversation_name,
                                )
                                self.learn_url(
                                    agent_name=agent_name,
                                    url=file_url,
                                    collection_number=collection_number,
                                )
                            elif file_url.startswith("https://github.com"):
                                self.new_conversation_message(
                                    role=agent_name,
                                    message=f"[ACTIVITY] Learning from GitHub.",
                                    conversation_name=conversation_name,
                                )
                                self.learn_github_repo(
                                    agent_name=agent_name,
                                    github_repo=file_url,
                                    github_user=(
                                        agent_settings["GITHUB_USER"]
                                        if "GITHUB_USER" in agent_settings
                                        else None
                                    ),
                                    github_token=(
                                        agent_settings["GITHUB_TOKEN"]
                                        if "GITHUB_TOKEN" in agent_settings
                                        else None
                                    ),
                                    github_branch=(
                                        "main"
                                        if "branch" not in message
                                        else message["branch"]
                                    ),
                                    collection_number=collection_number,
                                )
                            else:
                                self.new_conversation_message(
                                    role=agent_name,
                                    message=f"[ACTIVITY] Browsing {file_url} .",
                                    conversation_name=conversation_name,
                                )
                                self.learn_url(
                                    agent_name=agent_name,
                                    url=file_url,
                                    collection_number=collection_number,
                                )
                        else:
                            file_type = (
                                file_url.split(",")[0].split("/")[1].split(";")[0]
                            )
                            file_data = base64.b64decode(file_url.split(",")[1])
                            file_path = f"./WORKSPACE/{uuid.uuid4().hex}.{file_type}"
                            with open(file_path, "wb") as f:
                                f.write(file_data)
                            # file name should be a safe timestamp
                            file_name = f"Uploaded File {datetime.now().strftime('%Y%m%d%H%M%S')}.{file_type}"
                            self.new_conversation_message(
                                role=agent_name,
                                message=f"[ACTIVITY] Learning from uploaded file.",
                                conversation_name=conversation_name,
                            )
                            self.learn_file(
                                agent_name=agent_name,
                                file_name=f"Uploaded File {uuid.uuid4().hex}.{file_type}",
                                file_content=file_data,
                                collection_number=collection_number,
                            )
        self.new_conversation_message(
            role="user",
            message=new_prompt,
            conversation_name=conversation_name,
        )
        if async_func:
            response = await async_func(new_prompt)
        else:
            response = func(new_prompt)
        self.new_conversation_message(
            role=agent_name,
            message=response,
            conversation_name=conversation_name,
        )
        if tts:
            self.new_conversation_message(
                role=agent_name,
                message=f"[ACTIVITY] Generating audio response.",
                conversation_name=conversation_name,
            )
            tts_response = self.text_to_speech(agent_name=agent_name, text=response)
            self.new_conversation_message(
                role=agent_name,
                message=f'<audio controls><source src="{tts_response}" type="audio/wav"></audio>',
                conversation_name=conversation_name,
            )
        prompt_tokens = get_tokens(str(new_prompt))
        completion_tokens = get_tokens(str(response))
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        res_model = {
            "id": conversation_name,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response),
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return res_model

    def plan_task(
        self,
        agent_name: str,
        user_input: str,
        websearch: bool = False,
        websearch_depth: int = 3,
        conversation_name: str = "",
        log_user_input: bool = True,
        log_output: bool = True,
        enable_new_command: bool = True,
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/plan/task",
                json={
                    "user_input": user_input,
                    "websearch": websearch,
                    "websearch_depth": websearch_depth,
                    "conversation_name": conversation_name,
                    "log_user_input": log_user_input,
                    "log_output": log_output,
                    "enable_new_command": enable_new_command,
                },
            )
            return response.json()["response"]
        except Exception as e:
            return self.handle_error(e)
