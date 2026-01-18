"""
AGiXT Python SDK Tests

These tests run against a live AGiXT server.
Set the following environment variables:
- AGIXT_URI: AGiXT server URI (default: http://localhost:7437)
- AGIXT_API_KEY: API key for authentication (default: test-api-key)
"""

import os
import uuid
import pytest
from agixtsdk import AGiXTSDK


# Configuration from environment
AGIXT_URI = os.getenv("AGIXT_URI", "http://localhost:7437")
AGIXT_API_KEY = os.getenv("AGIXT_API_KEY", "test-api-key")


@pytest.fixture(scope="module")
def agixt():
    """Create AGiXT SDK instance."""
    return AGiXTSDK(base_uri=AGIXT_URI, api_key=AGIXT_API_KEY)


@pytest.fixture(scope="module")
def test_user(agixt):
    """Create a test user and return credentials."""
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    response = agixt.register_user(email=email, first_name="Test", last_name="User")
    return {"email": email, "response": response}


@pytest.fixture(scope="module")
def test_agent(agixt):
    """Create a test agent and clean up after tests."""
    agent_name = f"TestAgent_{uuid.uuid4().hex[:8]}"
    agent = agixt.add_agent(
        agent_name=agent_name,
        settings={"provider": "default"},
    )
    yield {"name": agent_name, "data": agent}
    # Cleanup: delete agent after tests
    try:
        agent_id = agixt.get_agent_id_by_name(agent_name)
        if agent_id:
            agixt.delete_agent(agent_id)
    except Exception:
        pass


@pytest.fixture(scope="module")
def test_conversation(agixt, test_agent):
    """Create a test conversation."""
    agent_id = agixt.get_agent_id_by_name(test_agent["name"])
    conv_name = f"TestConversation_{uuid.uuid4().hex[:8]}"
    conv = agixt.new_conversation(agent_name=agent_id, conversation_name=conv_name)
    yield {"name": conv_name, "data": conv}
    # Cleanup
    try:
        conv_id = agixt.get_conversation_id_by_name(conv_name)
        if conv_id:
            agixt.delete_conversation(conv_id)
    except Exception:
        pass


class TestConnection:
    """Test basic connectivity to AGiXT server."""

    def test_server_reachable(self, agixt):
        """Test that the AGiXT server is reachable."""
        # If we can get providers, server is up
        providers = agixt.get_providers()
        assert providers is not None


class TestUserManagement:
    """Test user management endpoints."""

    def test_register_user(self, agixt):
        """Test user registration."""
        email = f"newuser_{uuid.uuid4().hex[:8]}@example.com"
        response = agixt.register_user(email=email, first_name="New", last_name="User")
        assert response is not None

    def test_user_exists(self, agixt, test_user):
        """Test checking if user exists."""
        exists = agixt.user_exists(test_user["email"])
        assert exists is True

    def test_get_user(self, agixt):
        """Test getting current user info."""
        user = agixt.get_user()
        # Should return user data or None
        assert user is None or isinstance(user, dict)


class TestAgentManagement:
    """Test agent management endpoints."""

    def test_get_agents(self, agixt):
        """Test listing agents."""
        agents = agixt.get_agents()
        assert isinstance(agents, (list, dict))

    def test_add_agent(self, agixt):
        """Test creating an agent."""
        agent_name = f"NewAgent_{uuid.uuid4().hex[:8]}"
        agent = agixt.add_agent(agent_name=agent_name, settings={"provider": "default"})
        assert agent is not None
        # Cleanup
        try:
            agent_id = agixt.get_agent_id_by_name(agent_name)
            if agent_id:
                agixt.delete_agent(agent_id)
        except Exception:
            pass

    def test_get_agent_id_by_name(self, agixt, test_agent):
        """Test getting agent ID by name."""
        agent_id = agixt.get_agent_id_by_name(test_agent["name"])
        assert agent_id is not None

    def test_get_agent_config(self, agixt, test_agent):
        """Test getting agent configuration."""
        agent_id = agixt.get_agent_id_by_name(test_agent["name"])
        config = agixt.get_agent_config(agent_id)
        assert config is not None


class TestConversationManagement:
    """Test conversation management endpoints."""

    def test_get_conversations(self, agixt):
        """Test listing conversations."""
        conversations = agixt.get_conversations()
        assert isinstance(conversations, (list, dict))

    def test_new_conversation(self, agixt, test_agent):
        """Test creating a conversation."""
        agent_id = agixt.get_agent_id_by_name(test_agent["name"])
        conv_name = f"NewConv_{uuid.uuid4().hex[:8]}"
        conv = agixt.new_conversation(agent_name=agent_id, conversation_name=conv_name)
        assert conv is not None
        # Cleanup
        try:
            conv_id = agixt.get_conversation_id_by_name(conv_name)
            if conv_id:
                agixt.delete_conversation(conv_id)
        except Exception:
            pass

    def test_get_conversation(self, agixt, test_conversation):
        """Test getting conversation history."""
        conv_id = agixt.get_conversation_id_by_name(test_conversation["name"])
        history = agixt.get_conversation(conv_id)
        assert history is not None


class TestProviders:
    """Test provider endpoints."""

    def test_get_providers(self, agixt):
        """Test getting all providers."""
        providers = agixt.get_providers()
        assert providers is not None

    def test_get_providers_by_service(self, agixt):
        """Test getting providers by service type."""
        providers = agixt.get_providers_by_service("llm")
        assert providers is not None


class TestChains:
    """Test chain management endpoints."""

    def test_get_chains(self, agixt):
        """Test listing chains."""
        chains = agixt.get_chains()
        assert isinstance(chains, (list, dict))


class TestPrompts:
    """Test prompt management endpoints."""

    def test_get_prompts(self, agixt):
        """Test listing prompts."""
        prompts = agixt.get_prompts()
        assert isinstance(prompts, (list, dict))

    def test_get_all_prompts(self, agixt):
        """Test getting all prompts."""
        prompts = agixt.get_all_prompts()
        assert prompts is not None


class TestExtensions:
    """Test extension endpoints."""

    def test_get_extensions(self, agixt):
        """Test getting extensions."""
        extensions = agixt.get_extensions()
        assert extensions is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
