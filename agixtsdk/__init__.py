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

try:

    from pydub import AudioSegment
except:
    pass


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

    def login(self, username: str, password: str, mfa_token: str = None):
        """
        Login with username/password authentication.

        Args:
            username: Username or email address
            password: User's password
            mfa_token: Optional TOTP code if MFA is enabled

        Returns:
            JWT token on success, or response dict on failure
        """
        payload = {
            "username": username,
            "password": password,
        }
        if mfa_token:
            payload["mfa_token"] = mfa_token

        response = requests.post(
            f"{self.base_uri}/v1/login",
            json=payload,
        )
        if self.verbose:
            parse_response(response)

        result = response.json()
        if response.status_code == 200:
            token = result.get("token")
            if token:
                self.headers = {"Authorization": token}
                if self.verbose:
                    print(f"Logged in successfully")
                return token
        return result

    def login_magic_link(self, email: str, otp: str):
        """
        Legacy login with magic link (email + OTP token).
        Maintained for backward compatibility.

        Args:
            email: User's email address
            otp: TOTP code from authenticator app

        Returns:
            JWT token on success, or response dict on failure
        """
        response = requests.post(
            f"{self.base_uri}/v1/login/magic-link",
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
                if self.verbose:
                    print(f"Log in at {detail}")
                return token
        return response

    def register_user(
        self,
        email: str,
        password: str,
        confirm_password: str,
        first_name: str = "",
        last_name: str = "",
        username: str = None,
        organization_name: str = "",
    ):
        """
        Register a new user with username/password authentication.

        Args:
            email: User's email address
            password: User's password
            confirm_password: Password confirmation
            first_name: User's first name (optional)
            last_name: User's last name (optional)
            username: Desired username (optional, auto-generated from email if not provided)
            organization_name: Company/organization name (optional)

        Returns:
            Response dict with user_id, username, token on success
        """
        payload = {
            "email": email,
            "password": password,
            "confirm_password": confirm_password,
            "first_name": first_name,
            "last_name": last_name,
        }
        if username:
            payload["username"] = username
        if organization_name:
            payload["organization_name"] = organization_name

        response = requests.post(
            f"{self.base_uri}/v1/user",
            json=payload,
        )
        if self.verbose:
            parse_response(response)

        result = response.json()
        if response.status_code == 200:
            # Automatically set the token for subsequent requests
            token = result.get("token")
            if token:
                self.headers = {"Authorization": token}
                if self.verbose:
                    print(f"Registered and logged in as {result.get('username')}")
        return result

    def get_mfa_setup(self):
        """
        Get MFA setup information including QR code URI.

        Returns:
            Dict with provisioning_uri, secret, and mfa_enabled status
        """
        response = requests.get(
            f"{self.base_uri}/v1/user/mfa/setup",
            headers=self.headers,
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def enable_mfa(self, mfa_token: str):
        """
        Enable MFA for the current user.

        Args:
            mfa_token: TOTP code from authenticator app to verify setup

        Returns:
            Response dict with success message
        """
        response = requests.post(
            f"{self.base_uri}/v1/user/mfa/enable",
            headers=self.headers,
            json={"mfa_token": mfa_token},
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def disable_mfa(self, password: str = None, mfa_token: str = None):
        """
        Disable MFA for the current user.

        Args:
            password: User's password (optional)
            mfa_token: Current TOTP code (optional)

        Returns:
            Response dict with success message
        """
        payload = {}
        if password:
            payload["password"] = password
        if mfa_token:
            payload["mfa_token"] = mfa_token

        response = requests.post(
            f"{self.base_uri}/v1/user/mfa/disable",
            headers=self.headers,
            json=payload,
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def change_password(
        self, current_password: str, new_password: str, confirm_password: str
    ):
        """
        Change the current user's password.

        Args:
            current_password: Current password
            new_password: New password
            confirm_password: New password confirmation

        Returns:
            Response dict with success message
        """
        response = requests.post(
            f"{self.base_uri}/v1/user/password/change",
            headers=self.headers,
            json={
                "current_password": current_password,
                "new_password": new_password,
                "confirm_password": confirm_password,
            },
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def set_password(self, new_password: str, confirm_password: str):
        """
        Set a password for users who don't have one (migrating from magic link).

        Args:
            new_password: New password
            confirm_password: New password confirmation

        Returns:
            Response dict with success message and username
        """
        response = requests.post(
            f"{self.base_uri}/v1/user/password/set",
            headers=self.headers,
            json={
                "new_password": new_password,
                "confirm_password": confirm_password,
            },
        )
        if self.verbose:
            parse_response(response)
        return response.json()

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

    def get_providers(self) -> List[Dict[str, Any]]:
        """Get all available providers with their details."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/providers"
            )
            if self.verbose:
                parse_response(response)
            data = response.json()
            # Handle both list (v1) and dict (legacy) responses
            if isinstance(data, list):
                return data
            return data.get("providers", data)
        except Exception as e:
            return self.handle_error(e)

    def get_providers_by_service(self, service: str) -> List[str]:
        """Get providers filtered by service type."""
        try:
            providers = self.get_providers()
            # Filter providers by service if they have service info
            filtered = []
            for provider in providers:
                if isinstance(provider, dict):
                    if provider.get("service") == service:
                        filtered.append(provider.get("name", provider))
                else:
                    filtered.append(provider)
            return (
                filtered
                if filtered
                else [p.get("name", p) if isinstance(p, dict) else p for p in providers]
            )
        except Exception as e:
            return self.handle_error(e)

    def get_provider_settings(self, provider_name: str) -> Dict[str, Any]:
        """Get settings for a specific provider."""
        try:
            providers = self.get_providers()
            for provider in providers:
                if isinstance(provider, dict) and provider.get("name") == provider_name:
                    return provider.get("settings", provider)
            return {}
        except Exception as e:
            return self.handle_error(e)

    def get_embed_providers(self) -> List[str]:
        """Get embedding providers."""
        try:
            providers = self.get_providers()
            # Filter for embedding providers
            embed_providers = []
            for provider in providers:
                if isinstance(provider, dict):
                    if provider.get("supports_embeddings", False):
                        embed_providers.append(provider.get("name", provider))
                else:
                    embed_providers.append(provider)
            return embed_providers
        except Exception as e:
            return self.handle_error(e)

    def get_embedders(self) -> Dict[str, Any]:
        """Get all embedders."""
        try:
            providers = self.get_providers()
            embedders = {}
            for provider in providers:
                if isinstance(provider, dict) and provider.get(
                    "supports_embeddings", False
                ):
                    embedders[provider.get("name")] = provider
            return embedders
        except Exception as e:
            return self.handle_error(e)

    def add_agent(
        self,
        agent_name: str,
        settings: Dict[str, Any] = {},
        commands: Dict[str, Any] = {},
        training_urls: List[str] = [],
    ) -> Dict[str, Any]:
        """Create a new agent. Returns agent info including agent_id."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent",
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
        """Import an agent configuration."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/import",
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

    def rename_agent(self, agent_id: str, new_name: str) -> str:
        """Rename an agent by ID."""
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}",
                json={"new_name": new_name},
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def update_agent_settings(
        self, agent_id: str, settings: Dict[str, Any], agent_name: str = ""
    ) -> str:
        """Update agent settings by ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/agent/{agent_id}",
                json={
                    "agent_name": agent_name,
                    "settings": settings,
                    "commands": {},
                    "training_urls": [],
                },
                headers=self.headers,
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_agent_commands(self, agent_id: str, commands: Dict[str, Any]) -> str:
        """Update agent commands by ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/agent/{agent_id}/commands",
                json={"commands": commands},
                headers=self.headers,
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_agent(self, agent_id: str) -> str:
        """Delete an agent by ID."""
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/v1/agent/{agent_id}"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_agents(
        self,
    ) -> List[Dict[str, Any]]:
        """Get all agents. Returns list of agents with their IDs."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/agent"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["agents"]
        except Exception as e:
            return self.handle_error(e)

    def get_agentconfig(self, agent_id: str) -> Dict[str, Any]:
        """Get agent configuration by ID."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/agent/{agent_id}"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["agent"]
        except Exception as e:
            return self.handle_error(e)

    def get_agent_id_by_name(self, agent_name: str) -> Optional[str]:
        """Get agent ID by name. Returns None if not found."""
        try:
            agents = self.get_agents()
            for agent in agents:
                if isinstance(agent, dict) and agent.get("name") == agent_name:
                    return agent.get("id")
            return None
        except Exception:
            return None

    def get_chain_id_by_name(self, chain_name: str) -> Optional[str]:
        """Get chain ID by name. Returns None if not found."""
        try:
            chains = self.get_chains()
            for chain in chains:
                if isinstance(chain, dict) and chain.get("name") == chain_name:
                    return chain.get("id")
            return None
        except Exception:
            return None

    def get_conversation_id_by_name(self, conversation_name: str) -> Optional[str]:
        """Get conversation ID by name. Returns None if not found."""
        try:
            conversations = self.get_conversations_with_ids()
            for conv in conversations:
                if isinstance(conv, dict) and conv.get("name") == conversation_name:
                    return conv.get("id")
            return None
        except Exception:
            return None

    def get_prompt_id_by_name(
        self, prompt_name: str, category: str = "Default"
    ) -> Optional[str]:
        """Get prompt ID by name. Returns None if not found."""
        try:
            prompts = self.get_prompts(prompt_category=category)
            for prompt in prompts:
                if isinstance(prompt, dict) and prompt.get("name") == prompt_name:
                    return prompt.get("id")
            return None
        except Exception:
            return None

    def get_conversations(self, agent_id: str = "") -> List[Dict[str, Any]]:
        """Get all conversations. Returns list with conversation IDs."""
        url = f"{self.base_uri}/v1/conversations"
        try:
            response = requests.get(
                headers=self.headers,
                url=url,
            )
            if self.verbose:
                parse_response(response)
            data = response.json()
            # Handle both list (v1) and dict (legacy) responses
            if isinstance(data, list):
                return data
            return data.get("conversations", data)
        except Exception as e:
            return self.handle_error(e)

    def get_conversations_with_ids(self) -> List[Dict[str, Any]]:
        """Get all conversations with their IDs."""
        url = f"{self.base_uri}/v1/conversations"
        try:
            response = requests.get(
                headers=self.headers,
                url=url,
            )
            if self.verbose:
                parse_response(response)
            data = response.json()
            # Handle both list (v1) and dict (legacy) responses
            if isinstance(data, list):
                return data
            return data.get("conversations_with_ids", data.get("conversations", data))
        except Exception as e:
            return self.handle_error(e)

    def get_conversation(
        self, conversation_id: str, limit: int = 100, page: int = 1
    ) -> List[Dict[str, Any]]:
        """Get conversation history by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}",
                params={
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
        conversation_id: str,
        message_id: str,
    ):
        """Fork a conversation from a specific message."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/fork/{conversation_id}/{message_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def new_conversation(
        self,
        agent_id: str,
        conversation_name: str,
        conversation_content: List[Dict[str, Any]] = [],
    ) -> Dict[str, Any]:
        """Create a new conversation. Returns conversation with ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation",
                json={
                    "conversation_name": conversation_name,
                    "agent_id": agent_id,
                    "conversation_content": conversation_content,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def rename_conversation(
        self,
        conversation_id: str,
        new_name: str = "-",
    ):
        """Rename a conversation by ID."""
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}",
                json={
                    "new_conversation_name": new_name,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation(self, conversation_id: str) -> str:
        """Delete a conversation by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation_message(self, conversation_id: str, message_id: str) -> str:
        """Delete a message from a conversation by IDs."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}/message/{message_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_conversation_message(
        self, conversation_id: str, message_id: str, new_message: str
    ) -> str:
        """Update a message in a conversation by IDs."""
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}/message/{message_id}",
                json={
                    "new_message": new_message,
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
        conversation_id: str = "",
    ) -> str:
        """Add a new message to a conversation."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}/message",
                json={
                    "role": role,
                    "message": message,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def prompt_agent(
        self,
        agent_id: str,
        prompt_name: str,
        prompt_args: dict,
    ) -> str:
        """Send a prompt to an agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/prompt",
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

    def instruct(self, agent_id: str, user_input: str, conversation_id: str) -> str:
        """Send an instruction to an agent."""
        return self.prompt_agent(
            agent_id=agent_id,
            prompt_name="instruct",
            prompt_args={
                "user_input": user_input,
                "disable_memory": True,
                "conversation_name": conversation_id,
            },
        )

    def chat(
        self,
        agent_id: str,
        user_input: str,
        conversation_id: str,
        context_results: int = 4,
    ) -> str:
        """Chat with an agent."""
        return self.prompt_agent(
            agent_id=agent_id,
            prompt_name="Chat",
            prompt_args={
                "user_input": user_input,
                "context_results": context_results,
                "conversation_name": conversation_id,
                "disable_memory": True,
            },
        )

    def smartinstruct(
        self, agent_id: str, user_input: str, conversation_id: str
    ) -> str:
        """Send a smart instruction to an agent using a chain."""
        return self.run_chain(
            chain_name="Smart Instruct",
            user_input=user_input,
            agent_id=agent_id,
            all_responses=False,
            from_step=1,
            chain_args={
                "conversation_name": conversation_id,
                "disable_memory": True,
            },
        )

    def smartchat(self, agent_id: str, user_input: str, conversation_id: str) -> str:
        """Smart chat with an agent using a chain."""
        return self.run_chain(
            chain_name="Smart Chat",
            user_input=user_input,
            agent_id=agent_id,
            all_responses=False,
            from_step=1,
            chain_args={
                "conversation_name": conversation_id,
                "disable_memory": True,
            },
        )

    def get_commands(self, agent_id: str) -> Dict[str, Dict[str, bool]]:
        """Get available commands for an agent by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/command",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["commands"]
        except Exception as e:
            return self.handle_error(e)

    def toggle_command(self, agent_id: str, command_name: str, enable: bool) -> str:
        """Toggle a command for an agent by ID."""
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/command",
                json={"command_name": command_name, "enable": enable},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def execute_command(
        self,
        agent_id: str,
        command_name: str,
        command_args: dict,
        conversation_id: str = "",
    ):
        """Execute a command on an agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/command",
                json={
                    "command_name": command_name,
                    "command_args": command_args,
                    "conversation_name": conversation_id,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()["response"]
        except Exception as e:
            return self.handle_error(e)

    def get_chains(self) -> List[Dict[str, Any]]:
        """Get all chains. Returns list with chain IDs."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/chains"
            )
            if self.verbose:
                parse_response(response)
            return response.json()  # Returns list directly
        except Exception as e:
            return self.handle_error(e)

    def get_chain(self, chain_id: str) -> Dict[str, Any]:
        """Get a chain by ID."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/chain/{chain_id}"
            )
            if self.verbose:
                parse_response(response)
            data = response.json()
            # Response is {chain_name: {chain_data}} - extract the chain data
            if isinstance(data, dict) and len(data) == 1:
                return list(data.values())[0]
            return data
        except Exception as e:
            return self.handle_error(e)

    def get_chain_responses(self, chain_id: str) -> Dict[str, Any]:
        """Get chain responses by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/responses",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["chain"]
        except Exception as e:
            return self.handle_error(e)

    def get_chain_args(self, chain_id: str) -> List[str]:
        """Get chain arguments by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/args",
            )
            if self.verbose:
                parse_response(response)
            # Response is a list directly
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def run_chain(
        self,
        chain_id: str = "",
        chain_name: str = "",
        user_input: str = "",
        agent_id: str = "",
        all_responses: bool = False,
        from_step: int = 1,
        chain_args={},
    ) -> str:
        """Run a chain by ID or name."""
        try:
            # Use chain_id if provided, otherwise use chain_name for lookup
            endpoint = chain_id if chain_id else chain_name
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{endpoint}/run",
                json={
                    "prompt": user_input,
                    "agent_override": agent_id,
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
        chain_id: str,
        step_number: int,
        user_input: str,
        agent_id: str = None,
        chain_args={},
    ) -> str:
        """Run a specific chain step by chain ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/run/step/{step_number}",
                json={
                    "prompt": user_input,
                    "agent_override": agent_id,
                    "chain_args": chain_args,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def add_chain(self, chain_name: str) -> Dict[str, Any]:
        """Create a new chain. Returns chain info with ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain",
                json={"chain_name": chain_name},
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def import_chain(self, chain_name: str, steps: dict) -> str:
        """Import a chain with steps."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/import",
                json={"chain_name": chain_name, "steps": steps},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def rename_chain(self, chain_id: str, new_name: str) -> str:
        """Rename a chain by ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/chain/{chain_id}",
                json={"new_name": new_name},
                headers=self.headers,
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_chain(self, chain_id: str) -> str:
        """Delete a chain by ID."""
        try:
            response = requests.delete(
                headers=self.headers, url=f"{self.base_uri}/v1/chain/{chain_id}"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def add_step(
        self,
        chain_id: str,
        step_number: int,
        agent_id: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        """Add a step to a chain by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/step",
                json={
                    "step_number": step_number,
                    "agent_id": agent_id,
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
        chain_id: str,
        step_number: int,
        agent_id: str,
        prompt_type: str,
        prompt: dict,
    ) -> str:
        """Update a chain step by chain ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/chain/{chain_id}/step/{step_number}",
                headers=self.headers,
                json={
                    "step_number": step_number,
                    "agent_id": agent_id,
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
        chain_id: str,
        old_step_number: int,
        new_step_number: int,
    ) -> str:
        """Move a chain step by chain ID."""
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/step/move",
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

    def delete_step(self, chain_id: str, step_number: int) -> str:
        """Delete a chain step by chain ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/chain/{chain_id}/step/{step_number}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def add_prompt(
        self, prompt_name: str, prompt: str, prompt_category: str = "Default"
    ) -> Dict[str, Any]:
        """Create a new prompt. Returns prompt info with ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt",
                json={
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "prompt_category": prompt_category,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Get a prompt by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/{prompt_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_prompts(self, prompt_category: str = "Default") -> List[Dict[str, Any]]:
        """Get all prompts in a category."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompts",
                params={"prompt_category": prompt_category},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["prompts"]
        except Exception as e:
            return self.handle_error(e)

    def get_all_prompts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all global and user prompts with full details including IDs."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/all",
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_categories(self) -> List[Dict[str, Any]]:
        """Get all prompt categories with IDs."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/categories",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["categories"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompts_by_category_id(self, category_id: str) -> List[Dict[str, Any]]:
        """Get prompts by category ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/category/{category_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["prompts"]
        except Exception as e:
            return self.handle_error(e)

    def get_prompt_args(self, prompt_id: str) -> Dict[str, Any]:
        """Get prompt arguments by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/{prompt_id}/args",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["prompt_args"]
        except Exception as e:
            return self.handle_error(e)

    def delete_prompt(self, prompt_id: str) -> str:
        """Delete a prompt by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/{prompt_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_prompt(self, prompt_id: str, prompt: str) -> str:
        """Update a prompt by ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/prompt/{prompt_id}",
                headers=self.headers,
                json={
                    "prompt": prompt,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def rename_prompt(self, prompt_id: str, new_name: str) -> str:
        """Rename a prompt by ID."""
        try:
            response = requests.patch(
                headers=self.headers,
                url=f"{self.base_uri}/v1/prompt/{prompt_id}",
                json={"prompt_name": new_name},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_persona(self, agent_id: str) -> Dict[str, Any]:
        """Get agent persona by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/persona",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def update_persona(self, agent_id: str, persona: str) -> str:
        """Update agent persona by ID."""
        try:
            response = requests.put(
                f"{self.base_uri}/v1/agent/{agent_id}/persona",
                headers=self.headers,
                json={"persona": persona},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_extension_settings(self) -> Dict[str, Any]:
        """Get extension settings."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/extensions/settings"
            )
            if self.verbose:
                parse_response(response)
            return response.json()["extension_settings"]
        except Exception as e:
            return self.handle_error(e)

    def get_extensions(self):
        """Get all available extensions."""
        try:
            response = requests.get(
                headers=self.headers, url=f"{self.base_uri}/v1/extensions"
            )
            if self.verbose:
                parse_response(response)
            data = response.json()
            # Handle both list (v1) and dict (legacy) responses
            if isinstance(data, list):
                return data
            return data.get("extensions", data)
        except Exception as e:
            return self.handle_error(e)

    def get_agent_extensions(self, agent_id: str):
        """Get extensions for an agent by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/extensions",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["extensions"]
        except Exception as e:
            return self.handle_error(e)

    def get_command_args(self, command_name: str) -> Dict[str, Any]:
        """Get arguments for a command."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/extensions/{command_name}/args",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["command_args"]
        except Exception as e:
            return self.handle_error(e)

    def get_embedders_details(self) -> Dict[str, Any]:
        """Get embedder details."""
        try:
            providers = self.get_providers()
            embedders = {}
            for provider in providers:
                if isinstance(provider, dict) and provider.get(
                    "supports_embeddings", False
                ):
                    embedders[provider.get("name")] = provider
            return embedders
        except Exception as e:
            return self.handle_error(e)

    def positive_feedback(
        self,
        agent_id: str,
        message: str,
        user_input: str,
        feedback: str,
        conversation_id: str = "",
    ) -> str:
        """Submit positive feedback for an agent response."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/feedback",
                json={
                    "user_input": user_input,
                    "message": message,
                    "feedback": feedback,
                    "positive": True,
                    "conversation_name": conversation_id,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def negative_feedback(
        self,
        agent_id: str,
        message: str,
        user_input: str,
        feedback: str,
        conversation_id: str = "",
    ) -> str:
        """Submit negative feedback for an agent response."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/feedback",
                json={
                    "user_input": user_input,
                    "message": message,
                    "feedback": feedback,
                    "positive": False,
                    "conversation_name": conversation_id,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def learn_text(
        self, agent_id: str, user_input: str, text: str, collection_number: str = "0"
    ) -> str:
        """Teach agent text content by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/learn/text",
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

    def learn_url(self, agent_id: str, url: str, collection_number: str = "0") -> str:
        """Teach agent content from a URL by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/learn/url",
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
        agent_id: str,
        file_name: str,
        file_content: str,
        collection_number: str = "0",
    ) -> str:
        """Teach agent content from a file by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/learn/file",
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
        agent_id: str,
        github_repo: str,
        github_user: str = None,
        github_token: str = None,
        github_branch: str = "main",
        use_agent_settings: bool = False,
        collection_number: str = "0",
    ):
        """Teach agent content from a GitHub repo by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/learn/github",
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
        agent_id: str,
        query: str = None,
        arxiv_ids: str = None,
        max_results: int = 5,
        collection_number: str = "0",
    ):
        """Teach agent content from arXiv by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/learn/arxiv",
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
        agent_id: str,
        reader_name: str,
        data: dict,
        collection_number: str = "0",
    ):
        """Use agent reader by ID."""
        if "collection_number" not in data:
            data["collection_number"] = collection_number
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/reader/{reader_name}",
                json=data,
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def wipe_agent_memories(self, agent_id: str, collection_number: str = "0") -> str:
        """Wipe agent memories by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=(
                    f"{self.base_uri}/v1/agent/{agent_id}/memory"
                    if collection_number == "0"
                    else f"{self.base_uri}/v1/agent/{agent_id}/memory/{collection_number}"
                ),
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def delete_agent_memory(
        self,
        agent_id: str,
        memory_id: str,
        collection_number: str = "0",
    ) -> str:
        """Delete a specific agent memory by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/{collection_number}/{memory_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_agent_memories(
        self,
        agent_id: str,
        user_input: str,
        limit: int = 5,
        min_relevance_score: float = 0.0,
        collection_number: str = "0",
    ) -> List[Dict[str, Any]]:
        """Query agent memories by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/{collection_number}/query",
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

    def export_agent_memories(self, agent_id: str) -> List[Dict[str, Any]]:
        """Export agent memories by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/export",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["memories"]
        except Exception as e:
            return self.handle_error(e)

    def import_agent_memories(
        self, agent_id: str, memories: List[Dict[str, Any]]
    ) -> str:
        """Import memories to an agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/import",
                json={"memories": memories},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def create_dataset(self, agent_id: str, dataset_name: str, batch_size: int = 4):
        """Create a dataset from agent memories by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/dataset",
                json={"dataset_name": dataset_name, "batch_size": batch_size},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_browsed_links(
        self, agent_id: str, collection_number: str = "0"
    ) -> List[str]:
        """Get browsed links for an agent by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/browsed_links/{collection_number}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["links"]
        except Exception as e:
            return self.handle_error(e)

    def delete_browsed_link(
        self, agent_id: str, link: str, collection_number: str = "0"
    ) -> str:
        """Delete a browsed link for an agent by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/browsed_links",
                json={"link": link, "collection_number": collection_number},
            )
            if self.verbose:
                parse_response(response)
            return response.json()["message"]
        except Exception as e:
            return self.handle_error(e)

    def get_memories_external_sources(self, agent_id: str, collection_number: str):
        """Get external memory sources for an agent by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/external_sources/{collection_number}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()["external_sources"]
        except Exception as e:
            return self.handle_error(e)

    def delete_memory_external_source(
        self, agent_id: str, source: str, collection_number: str
    ) -> str:
        """Delete an external memory source for an agent by ID."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memories/external_source",
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
        agent_id: str,
        dataset_name: str = "dataset",
        model: str = "unsloth/mistral-7b-v0.2",
        max_seq_length: int = 16384,
        huggingface_output_path: str = "JoshXT/finetuned-mistral-7b-v0.2",
        private_repo: bool = True,
    ):
        """Train/finetune a model using agent memories by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/dataset/{dataset_name}/finetune",
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

    def text_to_speech(self, agent_id: str, text: str):
        """Convert text to speech using agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/text_to_speech",
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
        """OpenAI-compatible chat completions wrapper.

        Note: This method uses agent_id (passed via prompt.model) and
        conversation_id (passed via prompt.user) for compatibility with
        the OpenAI API format.
        """
        agent_id = prompt.model  # prompt.model is the agent ID
        conversation_id = (
            prompt.user if prompt.user else "-"
        )  # prompt.user is the conversation ID
        agent_config = self.get_agentconfig(agent_id=agent_id)
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
                        try:
                            AudioSegment.from_file(audio_url).set_frame_rate(
                                16000
                            ).export(wav_file, format="wav")
                        except:
                            pass
                        # Switch this to use the endpoint
                        openai.api_key = (
                            self.headers["Authorization"]
                            .replace("Bearer ", "")
                            .replace("bearer ", "")
                        )
                        openai.base_uri = f"{self.base_uri}/v1/"
                        self.new_conversation_message(
                            role=agent_id,
                            message=f"[ACTIVITY] Transcribing audio to text.",
                            conversation_id=conversation_id,
                        )
                        try:
                            with open(wav_file, "rb") as audio_file:
                                transcription = openai.audio.transcriptions.create(
                                    model=agent_id, file=audio_file
                                )
                        except:
                            pass
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
                                role=agent_id,
                                message=f"[ACTIVITY] Learning video from YouTube.",
                                conversation_id=conversation_id,
                            )
                            self.learn_url(
                                agent_id=agent_id,
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
                                    role=agent_id,
                                    message=f"[ACTIVITY] Learning video from YouTube.",
                                    conversation_id=conversation_id,
                                )
                                self.learn_url(
                                    agent_id=agent_id,
                                    url=file_url,
                                    collection_number=collection_number,
                                )
                            elif file_url.startswith("https://github.com"):
                                self.new_conversation_message(
                                    role=agent_id,
                                    message=f"[ACTIVITY] Learning from GitHub.",
                                    conversation_id=conversation_id,
                                )
                                self.learn_github_repo(
                                    agent_id=agent_id,
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
                                    role=agent_id,
                                    message=f"[ACTIVITY] Browsing {file_url} .",
                                    conversation_id=conversation_id,
                                )
                                self.learn_url(
                                    agent_id=agent_id,
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
                                role=agent_id,
                                message=f"[ACTIVITY] Learning from uploaded file.",
                                conversation_id=conversation_id,
                            )
                            self.learn_file(
                                agent_id=agent_id,
                                file_name=f"Uploaded File {uuid.uuid4().hex}.{file_type}",
                                file_content=file_data,
                                collection_number=collection_number,
                            )
        self.new_conversation_message(
            role="user",
            message=new_prompt,
            conversation_id=conversation_id,
        )
        if async_func:
            response = await async_func(new_prompt)
        else:
            response = func(new_prompt)
        self.new_conversation_message(
            role=agent_id,
            message=response,
            conversation_id=conversation_id,
        )
        if tts:
            self.new_conversation_message(
                role=agent_id,
                message=f"[ACTIVITY] Generating audio response.",
                conversation_id=conversation_id,
            )
            tts_response = self.text_to_speech(agent_id=agent_id, text=response)
            self.new_conversation_message(
                role=agent_id,
                message=f'<audio controls><source src="{tts_response}" type="audio/wav"></audio>',
                conversation_id=conversation_id,
            )
        prompt_tokens = get_tokens(str(new_prompt))
        completion_tokens = get_tokens(str(response))
        total_tokens = int(prompt_tokens) + int(completion_tokens)
        res_model = {
            "id": conversation_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": agent_id,
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
        agent_id: str,
        user_input: str,
        websearch: bool = False,
        websearch_depth: int = 3,
        conversation_id: str = "",
        log_user_input: bool = True,
        log_output: bool = True,
        enable_new_command: bool = True,
    ):
        """Plan a task using an agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/plan/task",
                json={
                    "user_input": user_input,
                    "websearch": websearch,
                    "websearch_depth": websearch_depth,
                    "conversation_name": conversation_id,
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
        agent_id: str = "gpt4free",
        max_failures: int = 3,
        response_type: str = None,
        **kwargs,
    ):
        """
        Converts a string to a Pydantic model using an AGiXT agent.

        Args:
        input_string (str): The string to convert to a model.
        model (Type[BaseModel]): The Pydantic model to convert the string to.
        agent_id (str): The ID of the AGiXT agent to use for the conversion.
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
            agent_id=agent_id,
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
                agent_id=agent_id,
                max_failures=max_failures,
                **kwargs,
            )

    def convert_list_of_dicts(
        self,
        data: List[dict],
        model: Type[BaseModel],
        agent_id: str = "gpt4free",
    ):
        """Convert a list of dicts to models using an agent by ID."""
        converted_data = self.convert_to_model(
            input_string=json.dumps(data[0], indent=4),
            model=model,
            agent_id=agent_id,
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
        agent_id: str,
        extension_name: str,
        openapi_json_url: str,
    ):
        """
        Create an AGiXT extension for an OpenAPI specification from a JSON URL.

        Parameters:
        - agent_id (str): The ID of the agent to use for extension creation.
        - extension_name (str): The name of the extension to create.
        - openapi_json_url (str): The URL of the OpenAPI specification in JSON format.
        """
        print(
            f"Creating AGiXT extension for {extension_name}, this will take some time!"
        )
        chain_name = self.execute_command(
            agent_id=agent_id,
            command_name="Generate Extension from OpenAPI",
            command_args={
                "openapi_json_url": openapi_json_url,
                "extension_name": extension_name,
            },
            conversation_id=f"{extension_name} Extension Generation",
        )
        extension_download = self.run_chain(
            chain_name=chain_name,
            agent_id=agent_id,
            user_input=f"Create an AGiXT extension for {extension_name}.",
        )
        file_name = extension_download.split("/")[-1]
        extension_file = requests.get(extension_download)
        extension_dir = os.path.join(os.getcwd(), "extensions")
        extension_file_path = os.path.join(extension_dir, file_name)
        extension_content = extension_file.content
        os.makedirs(extension_dir, exist_ok=True)
        with open(extension_file_path, "wb") as f:
            f.write(f"# Generated from OpenAPI JSON URL: {openapi_json_url}\n".encode())
            f.write(
                f"# Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n".encode()
            )
            f.write(extension_content)
        return f"{extension_name} extension created and downloaded to {extension_file_path}"

    def get_dpo_response(
        self,
        agent_id: str,
        user_input: str,
        injected_memories: int = 10,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        """Get DPO response from agent by ID."""
        try:
            response = requests.post(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/dpo",
                json={
                    "user_input": user_input,
                    "injected_memories": injected_memories,
                    "conversation_name": conversation_id,
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

    def get_oauth2_providers(self):
        response = requests.get(f"{self.base_uri}/v1/oauth")
        if self.verbose:
            parse_response(response)
        data = response.json()
        if "providers" in data:
            return data["providers"]
        return data

    def get_user_oauth2_connections(self) -> List[str]:
        response = requests.get(f"{self.base_uri}/v1/oauth2", headers=self.headers)
        if self.verbose:
            parse_response(response)
        return response.json()

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

    def get_companies(self):
        response = requests.get(f"{self.base_uri}/v1/companies", headers=self.headers)
        if self.verbose:
            parse_response(response)
        return response.json()

    def create_company(
        self,
        name: str,
        agent_name: str,
        parent_company_id: Optional[str] = None,
    ):
        data = {"name": name, "agent_name": agent_name}
        if parent_company_id:
            data["parent_company_id"] = parent_company_id
        response = requests.post(
            f"{self.base_uri}/v1/companies",
            headers=self.headers,
            json=data,
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def update_company(self, company_id: str, name: str):
        response = requests.put(
            f"{self.base_uri}/v1/companies/{company_id}",
            headers=self.headers,
            json={"name": name},
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def delete_company(self, company_id: str):
        response = requests.delete(
            f"{self.base_uri}/v1/companies/{company_id}", headers=self.headers
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def delete_user_from_company(self, company_id: str, user_id: str):
        response = requests.delete(
            f"{self.base_uri}/v1/companies/{company_id}/users/{user_id}",
            headers=self.headers,
        )
        if self.verbose:
            parse_response(response)
        return response.json()

    def update_conversation_message_by_id(
        self,
        conversation_id: str,
        message_id: str,
        new_message: str,
    ) -> Dict[str, Any]:
        """Update a conversation message by conversation and message IDs."""
        try:
            response = requests.put(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}/message/{message_id}",
                json={
                    "new_message": new_message,
                },
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def delete_conversation_message_by_id(
        self,
        conversation_id: str,
        message_id: str,
    ) -> Dict[str, Any]:
        """Delete a conversation message by conversation and message IDs."""
        try:
            response = requests.delete(
                headers=self.headers,
                url=f"{self.base_uri}/v1/conversation/{conversation_id}/message/{message_id}",
            )
            if self.verbose:
                parse_response(response)
            return response.json()
        except Exception as e:
            return self.handle_error(e)

    def get_unique_external_sources(
        self,
        agent_id: str,
        collection_number: str = "0",
    ) -> List[str]:
        """Get unique external memory sources for an agent by ID."""
        try:
            response = requests.get(
                headers=self.headers,
                url=f"{self.base_uri}/v1/agent/{agent_id}/memory/external_sources/{collection_number}",
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
