from typing import (
    Dict,
    List,
    Any,
    Optional,
    Callable,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from pydantic import BaseModel
from pydub import AudioSegment
from datetime import datetime
from enum import Enum
import tiktoken
import requests
import inspect
import base64
import openai
import pyotp
import uuid
import time
import json
import os


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


def parse_response(response: requests.Response):
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    if response.status_code == 200:
        print(response.json())
    else:
        print(response.text)
        assert False
    print("\n")


class AGiXTSDK:
    def __init__(
        self, base_uri: str = None, api_key: str = None, verbose: bool = False
    ):
        if not base_uri:
            base_uri = "http://localhost:7437"
        self.base_uri = base_uri
        self.verbose = verbose
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
        self.failures = 0

    def handle_error(self, error) -> str:
        print(f"Error: {error}")
        raise Exception(f"Unable to retrieve data. {error}")

    def login(self, email, otp):
        response = requests.post(
            f"{self.base_uri}/v1/login",
            json={"email": email, "token": otp},
        )
        if self.verbose:
            parse_response(response)
        response = response.json()
        if "detail" in response:
            detail = response["detail"]
            if "?token=" in detail:
                token = detail.split("token=")[1]
                self.headers = {"Authorization": token}
                print(f"Log in at {detail}")
                return token

    def register_user(self, email, first_name, last_name):
        login_response = requests.post(
            f"{self.base_uri}/v1/user",
            json={
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
            },
        )
        if self.verbose:
            parse_response(login_response)
        response = login_response.json()
        if "otp_uri" in response:
            mfa_token = str(response["otp_uri"]).split("secret=")[1].split("&")[0]
            totp = pyotp.TOTP(mfa_token)
            self.login(email=email, otp=totp.now())
            return response["otp_uri"]
        else:
            return response

    def user_exists(self, email):
        response = requests.get(f"{self.base_uri}/v1/user/exists?email={email}")
        if self.verbose:
            parse_response(response)
        return response.json()

    def update_user(self, **kwargs):
        response = requests.put(
            f"{self.base_uri}/v1/user",
            headers=self.headers,
            json={**kwargs},
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def get_user(self):
        response = requests.get(f"{self.base_uri}/v1/user", headers=self.headers)
        if self.verbose:
            parse_response(response)
        return response.json()

    def get_providers(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/provider"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["providers"]
        except Exception as e:
            return self.handle_error(e)

    def get_providers_by_service(self, service: str) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/providers/service/{service}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["providers"]
        except Exception as e:
            return self.handle_error(e)

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/provider/{provider_name}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["settings"]
        except Exception as e:
            return self.handle_error(e)

    def get_embed_providers(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedding_providers"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["providers"]
        except Exception as e:
            return self.handle_error(e)

    def get_embedders(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedders"
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_agent(self, agent_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/api/agent/{agent_name}"
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["agents"]
        except Exception as e:
            return self.handle_error(e)

    def get_agentconfig(self, agent_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/agent/{agent_name}"
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["conversation_history"]
        except Exception as e:
            return self.handle_error(e)

    def fork_conversation(
        self,
        conversation_name: str,
        message_id: str,
    ):
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/fork",
                json={"conversation_name": conversation_name, "message_id": message_id},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["response"]
        except Exception as e:
            return self.handle_error(e)

    def get_chains(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/chain"
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_chain(self, chain_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/chain/{chain_name}"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["chain"]
        except Exception as e:
            return self.handle_error(e)

    def get_chain_responses(self, chain_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/responses",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["chain"]
        except Exception as e:
            return self.handle_error(e)

    def get_chain_args(self, chain_name: str) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/args",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_chain(self, chain_name: str) -> str:
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/api/chain/{chain_name}"
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
                headers=self.headers,
                json={
                    "step_number": step_number,
                    "agent_name": agent_name,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                },
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_step(self, chain_name: str, step_number: int) -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/chain/{chain_name}/step/{step_number}",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["prompt"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompts(self, prompt_category: str = "Default") -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["prompts"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_categories(self) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/categories",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["prompt_args"]
        except Exception as e:
            return self.handle_error(e)

    def delete_prompt(self, prompt_name: str, prompt_category: str = "Default") -> str:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/prompt/{prompt_category}/{prompt_name}",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_extension_settings(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/extensions/settings"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["extension_settings"]
        except Exception as e:
            return self.handle_error(e)

    def get_extensions(self):
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/extensions"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["extensions"]
        except Exception as e:
            return self.handle_error(e)

    def get_agent_extensions(self, agent_name: str = "AGiXT"):
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/extensions",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["extensions"]
        except Exception as e:
            return self.handle_error(e)

    def get_command_args(self, command_name: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/extensions/{command_name}/args",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["command_args"]
        except Exception as e:
            return self.handle_error(e)

    def get_embedders_details(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/api/embedders"
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["memories"]
        except Exception as e:
            return self.handle_error(e)

    def export_agent_memories(self, agent_name: str) -> List[Dict[str, Any]]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/export",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_memories_external_sources(self, agent_name: str, collection_number: str):
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/external_sources/{collection_number}",
            )
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
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
            if self.verbose:
                parse_response(response)
            return response.json()["response"]
        except Exception as e:
            return self.handle_error(e)

    def _generate_detailed_schema(self, model: Type[BaseModel], depth: int = 0) -> str:
        """
        Recursively generates a detailed schema representation of a Pydantic model,
        including nested models and complex types.
        """
        fields = get_type_hints(model)
        field_descriptions = []
        indent = "  " * depth
        for field, field_type in fields.items():
            description = f"{indent}{field}: "
            origin_type = get_origin(field_type)
            if origin_type is None:
                origin_type = field_type
            if inspect.isclass(origin_type) and issubclass(origin_type, BaseModel):
                description += f"Nested Model:\n{self._generate_detailed_schema(origin_type, depth + 1)}"
            elif origin_type == list:
                list_type = get_args(field_type)[0]
                if inspect.isclass(list_type) and issubclass(list_type, BaseModel):
                    description += f"List of Nested Model:\n{self._generate_detailed_schema(list_type, depth + 1)}"
                elif get_origin(list_type) == Union:
                    union_types = get_args(list_type)
                    description += f"List of Union:\n"
                    for union_type in union_types:
                        if inspect.isclass(union_type) and issubclass(
                            union_type, BaseModel
                        ):
                            description += f"{indent}  - Nested Model:\n{self._generate_detailed_schema(union_type, depth + 2)}"
                        else:
                            description += (
                                f"{indent}  - {self._get_type_name(union_type)}\n"
                            )
                else:
                    description += f"List[{self._get_type_name(list_type)}]"
            elif origin_type == dict:
                key_type, value_type = get_args(field_type)
                description += f"Dict[{self._get_type_name(key_type)}, {self._get_type_name(value_type)}]"
            elif origin_type == Union:
                union_types = get_args(field_type)

                for union_type in union_types:
                    if inspect.isclass(union_type) and issubclass(
                        union_type, BaseModel
                    ):
                        description += f"{indent}  - Nested Model:\n{self._generate_detailed_schema(union_type, depth + 2)}"
                    else:
                        type_name = self._get_type_name(union_type)
                        if type_name != "NoneType":
                            description += f"{self._get_type_name(union_type)}\n"
            elif inspect.isclass(origin_type) and issubclass(origin_type, Enum):
                enum_values = ", ".join([f"{e.name} = {e.value}" for e in origin_type])
                description += f"{origin_type.__name__} (Enum values: {enum_values})"
            else:
                description += self._get_type_name(origin_type)
            field_descriptions.append(description)
        return "\n".join(field_descriptions)

    def _get_type_name(self, type_):
        """Helper method to get the name of a type, handling some special cases."""
        if hasattr(type_, "__name__"):
            return type_.__name__
        return str(type_).replace("typing.", "")

    def convert_to_model(
        self,
        input_string: str,
        model: Type[BaseModel],
        agent_name: str = "gpt4free",
        max_failures: int = 3,
        response_type: str = None,
        **kwargs,
    ):
        """
        Converts a string to a Pydantic model using an AGiXT agent.

        Args:
        input_string (str): The string to convert to a model.
        model (Type[BaseModel]): The Pydantic model to convert the string to.
        agent_name (str): The name of the AGiXT agent to use for the conversion.
        max_failures (int): The maximum number of times to retry the conversion if it fails.
        response_type (str): The type of response to return. Either 'json' or None. None will return the model.
        **kwargs: Additional arguments to pass to the AGiXT agent as prompt arguments.
        """
        input_string = str(input_string)
        schema = self._generate_detailed_schema(model)
        if "user_input" in kwargs:
            del kwargs["user_input"]
        if "schema" in kwargs:
            del kwargs["schema"]
        response = self.prompt_agent(
            agent_name=agent_name,
            prompt_name="Convert to Model",
            prompt_args={
                "schema": schema,
                "user_input": input_string,
                **kwargs,
            },
        )
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        try:
            response = json.loads(response)
            if response_type == "json":
                return response
            else:
                return model(**response)
        except Exception as e:
            self.failures += 1
            if self.failures > max_failures:
                print(
                    f"Error: {e} . Failed to convert the response to the model after {max_failures} attempts. Response: {response}"
                )
                self.failures = 0
                return (
                    response
                    if response
                    else "Failed to convert the response to the model."
                )
            else:
                self.failures = 1
            print(
                f"Error: {e} . Failed to convert the response to the model, trying again. {self.failures}/3 failures. Response: {response}"
            )
            return self.convert_to_model(
                input_string=input_string,
                model=model,
                agent_name=agent_name,
                max_failures=max_failures,
                **kwargs,
            )

    def convert_list_of_dicts(
        self,
        data: List[dict],
        model: Type[BaseModel],
        agent_name: str = "gpt4free",
    ):
        converted_data = self.convert_to_model(
            input_string=json.dumps(data[0], indent=4),
            model=model,
            agent_name=agent_name,
        )
        mapped_list = []
        for info in data:
            new_data = {}
            for key, value in converted_data.items():
                item = [k for k, v in data[0].items() if v == value]
                if item:
                    new_data[key] = info[item[0]]
            mapped_list.append(new_data)
        return mapped_list

    def create_extension(
        self,
        agent_name: str,
        extension_name: str,
        openapi_json_url: str,
    ):
        """
        Create an AGiXT extension for an OpenAPI specification from a JSON URL.

        Parameters:
        - extension_name (str): The name of the extension to create.
        - openapi_json_url (str): The URL of the OpenAPI specification in JSON format.
        """
        print(
            f"Creating AGiXT extension for {extension_name}, this will take some time!"
        )
        chain_name = self.execute_command(
            agent_name=agent_name,
            command_name="Generate Extension from OpenAPI",
            command_args={
                "openapi_json_url": openapi_json_url,
                "extension_name": extension_name,
            },
            conversation_name=f"{extension_name} Extension Generation",
        )
        extension_download = self.run_chain(
            chain_name=chain_name,
            agent_name=agent_name,
            user_input=f"Create an AGiXT extension for {extension_name}.",
        )
        file_name = extension_download.split("/")[-1]
        extension_file = requests.get(extension_download)
        extension_dir = os.path.join(os.getcwd(), "extensions")
        extension_file_path = os.path.join(extension_dir, file_name)
        os.makedirs(extension_dir, exist_ok=True)
        with open(extension_file_path, "wb") as f:
            f.write(extension_file.content)
        return f"{extension_name} extension created and downloaded to {extension_file_path}"

    def get_dpo_response(
        self,
        agent_name: str,
        user_input: str,
        injected_memories: int = 10,
        conversation_name: str = "",
    ) -> Dict[str, Any]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/dpo",
                json={
                    "user_input": user_input,
                    "injected_memories": injected_memories,
                    "conversation_name": conversation_name,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def transcribe_audio(
        self,
        file: str,
        model: str,
        language: str = None,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        try:
            with open(file, "rb") as audio_file:
                files = {"file": (file, audio_file)}
                data = {
                    "model": model,
                    "language": language,
                    "prompt": prompt,
                    "response_format": response_format,
                    "temperature": temperature,
                }
                response = requests.post(
                    headers=self.headers,
                    url=f"{self.base_uri}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def translate_audio(
        self,
        file: str,
        model: str,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        try:
            with open(file, "rb") as audio_file:
                files = {"file": (file, audio_file)}
                data = {
                    "model": model,
                    "prompt": prompt,
                    "response_format": response_format,
                    "temperature": temperature,
                }
                response = requests.post(
                    headers=self.headers,
                    url=f"{self.base_uri}/v1/audio/translations",
                    files=files,
                    data=data,
                )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> Dict[str, Any]:
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/images/generations",
                json={
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "response_format": response_format,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def oauth2_login(
        self, provider: str, code: str, referrer: str = None
    ) -> Dict[str, Any]:
        try:
            data = {"code": code}
            if referrer:
                data["referrer"] = referrer
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/oauth2/{provider}",
                json=data,
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def update_conversation_message_by_id(
        self,
        message_id: str,
        new_message: str,
        conversation_name: str,
    ) -> Dict[str, Any]:
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/message/{message_id}",
                json={
                    "new_message": new_message,
                    "conversation_name": conversation_name,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation_message_by_id(
        self,
        message_id: str,
        conversation_name: str,
    ) -> Dict[str, Any]:
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/api/conversation/message/{message_id}",
                json={"conversation_name": conversation_name},
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_unique_external_sources(
        self,
        agent_name: str,
        collection_number: str = "0",
    ) -> List[str]:
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/api/agent/{agent_name}/memory/external_sources/{collection_number}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["external_sources"]
        except Exception as e:
            return self.handle_error(e)


def snake_case(old_str: str = ""):
    if not old_str:
        return ""
    if " " in old_str:
        old_str = old_str.replace(" ", "")
    if "@" in old_str:
        old_str = old_str.replace("@", "_")
    if "." in old_str:
        old_str = old_str.replace(".", "_")
    if "-" in old_str:
        old_str = old_str.replace("-", "_")
    if "&" in old_str:
        old_str = old_str.replace("&", "and")
    if ":" in old_str:
        old_str = old_str.replace(":", "_")
    snake_str = ""
    for i, char in enumerate(old_str):
        if char.isupper():
            if i != 0 and old_str[i - 1].islower():
                snake_str += "_"
            if i != len(old_str) - 1 and old_str[i + 1].islower():
                snake_str += "_"
        snake_str += char.lower()
    snake_str = snake_str.strip("_")
    return snake_str
