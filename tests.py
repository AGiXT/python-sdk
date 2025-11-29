"""
Comprehensive tests for the AGiXT SDK.
Tests all SDK functions against the v1 API endpoints.
"""
import pytest
import uuid
import time
import sys
import os

# Add parent directory for local testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agixtsdk import AGiXTSDK

# Test configuration
BASE_URI = "https://api.josh.devxt.com"


class TestAGiXTSDK:
    """Test class for AGiXT SDK methods."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures - create SDK instance and test user."""
        cls.sdk = AGiXTSDK(base_uri=BASE_URI, verbose=True)
        
        # Register a test user to get authentication
        test_email = f"testuser_{uuid.uuid4()}@example.com"
        cls.sdk.register_user(
            email=test_email,
            first_name="Test",
            last_name="User"
        )
        
        # Store test data for cleanup
        cls.test_agent_ids = []
        cls.test_chain_ids = []
        cls.test_conversation_ids = []
        cls.test_prompt_ids = []
        
        # Create a test agent for agent-dependent tests
        cls.test_agent_name = f"TestAgent_{uuid.uuid4().hex[:8]}"
        result = cls.sdk.add_agent(agent_name=cls.test_agent_name, settings={})
        if isinstance(result, dict) and "id" in result:
            cls.test_agent_id = result["id"]
            cls.test_agent_ids.append(cls.test_agent_id)
        else:
            # Try to find agent by name in agents list
            agents = cls.sdk.get_agents()
            for agent in agents:
                if isinstance(agent, dict) and agent.get("name") == cls.test_agent_name:
                    cls.test_agent_id = agent.get("id")
                    cls.test_agent_ids.append(cls.test_agent_id)
                    break

    @classmethod
    def teardown_class(cls):
        """Clean up test data."""
        # Delete test agents
        for agent_id in cls.test_agent_ids:
            try:
                cls.sdk.delete_agent(agent_id)
            except Exception:
                pass
        
        # Delete test chains
        for chain_id in cls.test_chain_ids:
            try:
                cls.sdk.delete_chain(chain_id)
            except Exception:
                pass
        
        # Delete test conversations
        for conv_id in cls.test_conversation_ids:
            try:
                cls.sdk.delete_conversation(conv_id)
            except Exception:
                pass
        
        # Delete test prompts
        for prompt_id in cls.test_prompt_ids:
            try:
                cls.sdk.delete_prompt(prompt_id)
            except Exception:
                pass

    # ==========================================================================
    # User Authentication Tests
    # ==========================================================================

    def test_get_user(self):
        """Test getting current user info."""
        result = self.sdk.get_user()
        assert result is not None
        assert isinstance(result, dict)
        print(f"Get user result: {result}")

    def test_user_exists(self):
        """Test checking if user exists."""
        result = self.sdk.user_exists("test@example.com")
        assert result is not None
        print(f"User exists result: {result}")

    def test_update_user(self):
        """Test updating user info."""
        result = self.sdk.update_user(first_name="Updated", last_name="User")
        assert result is not None
        print(f"Update user result: {result}")

    # ==========================================================================
    # Provider Tests
    # ==========================================================================

    def test_get_providers(self):
        """Test getting all providers."""
        result = self.sdk.get_providers()
        assert result is not None
        assert isinstance(result, list)
        print(f"Get providers result: {len(result)} providers found")

    def test_get_providers_by_service(self):
        """Test getting providers by service type."""
        result = self.sdk.get_providers_by_service("llm")
        assert result is not None
        print(f"Get providers by service result: {result}")

    def test_get_embed_providers(self):
        """Test getting embedding providers."""
        result = self.sdk.get_embed_providers()
        assert result is not None
        print(f"Get embed providers result: {result}")

    def test_get_embedders(self):
        """Test getting embedders."""
        result = self.sdk.get_embedders()
        assert result is not None
        print(f"Get embedders result: {result}")

    # ==========================================================================
    # Agent Tests
    # ==========================================================================

    def test_get_agents(self):
        """Test getting all agents."""
        result = self.sdk.get_agents()
        assert result is not None
        assert isinstance(result, list)
        print(f"Get agents result: {len(result)} agents found")

    def test_add_agent(self):
        """Test creating a new agent."""
        agent_name = f"TestAddAgent_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_agent(agent_name=agent_name, settings={})
        assert result is not None
        print(f"Add agent result: {result}")
        
        # Extract and store agent ID for cleanup
        if isinstance(result, dict):
            if "id" in result:
                self.test_agent_ids.append(result["id"])
            elif "agent_id" in result:
                self.test_agent_ids.append(result["agent_id"])

    def test_get_agentconfig(self):
        """Test getting agent configuration."""
        result = self.sdk.get_agentconfig(self.test_agent_id)
        assert result is not None
        print(f"Get agent config result: {result}")

    def test_update_agent_settings(self):
        """Test updating agent settings."""
        result = self.sdk.update_agent_settings(
            agent_id=self.test_agent_id,
            settings={"test_setting": "test_value"}
        )
        assert result is not None
        print(f"Update agent settings result: {result}")

    def test_update_agent_commands(self):
        """Test updating agent commands."""
        result = self.sdk.update_agent_commands(
            agent_id=self.test_agent_id,
            commands={}
        )
        assert result is not None
        print(f"Update agent commands result: {result}")

    def test_rename_agent(self):
        """Test renaming an agent."""
        new_name = f"RenamedAgent_{uuid.uuid4().hex[:8]}"
        result = self.sdk.rename_agent(
            agent_id=self.test_agent_id,
            new_name=new_name
        )
        assert result is not None
        print(f"Rename agent result: {result}")

    def test_get_commands(self):
        """Test getting agent commands."""
        result = self.sdk.get_commands(self.test_agent_id)
        assert result is not None
        print(f"Get commands result: {result}")

    def test_get_persona(self):
        """Test getting agent persona."""
        result = self.sdk.get_persona(self.test_agent_id)
        # Persona may be empty for new agents
        print(f"Get persona result: {result}")

    def test_update_persona(self):
        """Test updating agent persona."""
        result = self.sdk.update_persona(
            agent_id=self.test_agent_id,
            persona="Test persona description"
        )
        assert result is not None
        print(f"Update persona result: {result}")

    # ==========================================================================
    # Conversation Tests
    # ==========================================================================

    def test_get_conversations(self):
        """Test getting all conversations."""
        result = self.sdk.get_conversations()
        assert result is not None
        print(f"Get conversations result: {result}")

    def test_get_conversations_with_ids(self):
        """Test getting conversations with IDs."""
        result = self.sdk.get_conversations_with_ids()
        assert result is not None
        print(f"Get conversations with IDs result: {result}")

    def test_new_conversation(self):
        """Test creating a new conversation."""
        conv_name = f"TestConversation_{uuid.uuid4().hex[:8]}"
        result = self.sdk.new_conversation(
            agent_id=self.test_agent_id,
            conversation_name=conv_name,
            conversation_content=[]
        )
        assert result is not None
        print(f"New conversation result: {result}")
        
        # Store conversation ID for cleanup and further tests
        if isinstance(result, dict):
            if "id" in result:
                self.test_conversation_ids.append(result["id"])
                self.__class__.test_conversation_id = result["id"]
            elif "conversation_id" in result:
                self.test_conversation_ids.append(result["conversation_id"])
                self.__class__.test_conversation_id = result["conversation_id"]

    def test_get_conversation(self):
        """Test getting a conversation."""
        if not hasattr(self, 'test_conversation_id'):
            pytest.skip("No test conversation available")
        
        result = self.sdk.get_conversation(
            conversation_id=self.test_conversation_id,
            limit=100,
            page=1
        )
        assert result is not None
        print(f"Get conversation result: {result}")

    def test_rename_conversation(self):
        """Test renaming a conversation."""
        if not hasattr(self, 'test_conversation_id'):
            pytest.skip("No test conversation available")
        
        new_name = f"RenamedConv_{uuid.uuid4().hex[:8]}"
        result = self.sdk.rename_conversation(
            conversation_id=self.test_conversation_id,
            new_name=new_name
        )
        assert result is not None
        print(f"Rename conversation result: {result}")

    def test_new_conversation_message(self):
        """Test adding a message to a conversation."""
        if not hasattr(self, 'test_conversation_id'):
            pytest.skip("No test conversation available")
        
        result = self.sdk.new_conversation_message(
            role="user",
            message="Test message",
            conversation_id=self.test_conversation_id
        )
        assert result is not None
        print(f"New conversation message result: {result}")

    # ==========================================================================
    # Chain Tests
    # ==========================================================================

    def test_get_chains(self):
        """Test getting all chains."""
        result = self.sdk.get_chains()
        assert result is not None
        print(f"Get chains result: {result}")

    def test_add_chain(self):
        """Test creating a new chain."""
        chain_name = f"TestChain_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_chain(chain_name=chain_name)
        assert result is not None
        print(f"Add chain result: {result}")
        
        # Store chain ID for cleanup and further tests
        if isinstance(result, dict):
            if "id" in result:
                self.test_chain_ids.append(result["id"])
                self.__class__.test_chain_id = result["id"]
            elif "chain_id" in result:
                self.test_chain_ids.append(result["chain_id"])
                self.__class__.test_chain_id = result["chain_id"]

    def test_get_chain(self):
        """Test getting a chain."""
        if not hasattr(self, 'test_chain_id'):
            pytest.skip("No test chain available")
        
        result = self.sdk.get_chain(self.test_chain_id)
        assert result is not None
        print(f"Get chain result: {result}")

    def test_get_chain_args(self):
        """Test getting chain arguments."""
        if not hasattr(self, 'test_chain_id'):
            pytest.skip("No test chain available")
        
        result = self.sdk.get_chain_args(self.test_chain_id)
        # May be empty for chains without args
        print(f"Get chain args result: {result}")

    def test_get_chain_responses(self):
        """Test getting chain responses."""
        if not hasattr(self, 'test_chain_id'):
            pytest.skip("No test chain available")
        
        result = self.sdk.get_chain_responses(self.test_chain_id)
        print(f"Get chain responses result: {result}")

    def test_rename_chain(self):
        """Test renaming a chain."""
        if not hasattr(self, 'test_chain_id'):
            pytest.skip("No test chain available")
        
        new_name = f"RenamedChain_{uuid.uuid4().hex[:8]}"
        result = self.sdk.rename_chain(
            chain_id=self.test_chain_id,
            new_name=new_name
        )
        assert result is not None
        print(f"Rename chain result: {result}")

    def test_import_chain(self):
        """Test importing a chain."""
        chain_name = f"ImportedChain_{uuid.uuid4().hex[:8]}"
        result = self.sdk.import_chain(
            chain_name=chain_name,
            steps={}
        )
        print(f"Import chain result: {result}")

    # ==========================================================================
    # Prompt Tests
    # ==========================================================================

    def test_get_prompts(self):
        """Test getting all prompts."""
        result = self.sdk.get_prompts()
        assert result is not None
        print(f"Get prompts result: {len(result) if isinstance(result, list) else result}")

    def test_get_prompt_categories(self):
        """Test getting prompt categories."""
        result = self.sdk.get_prompt_categories()
        assert result is not None
        print(f"Get prompt categories result: {result}")

    def test_add_prompt(self):
        """Test creating a new prompt."""
        prompt_name = f"TestPrompt_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_prompt(
            prompt_name=prompt_name,
            prompt="This is a test prompt with {input}",
            prompt_category="Default"
        )
        assert result is not None
        print(f"Add prompt result: {result}")
        
        # Store prompt ID for cleanup and further tests
        if isinstance(result, dict):
            if "id" in result:
                self.test_prompt_ids.append(result["id"])
                self.__class__.test_prompt_id = result["id"]
            elif "prompt_id" in result:
                self.test_prompt_ids.append(result["prompt_id"])
                self.__class__.test_prompt_id = result["prompt_id"]

    def test_get_prompt(self):
        """Test getting a prompt."""
        if not hasattr(self, 'test_prompt_id'):
            pytest.skip("No test prompt available")
        
        result = self.sdk.get_prompt(self.test_prompt_id)
        assert result is not None
        print(f"Get prompt result: {result}")

    def test_get_prompt_args(self):
        """Test getting prompt arguments."""
        if not hasattr(self, 'test_prompt_id'):
            pytest.skip("No test prompt available")
        
        result = self.sdk.get_prompt_args(self.test_prompt_id)
        print(f"Get prompt args result: {result}")

    def test_update_prompt(self):
        """Test updating a prompt."""
        if not hasattr(self, 'test_prompt_id'):
            pytest.skip("No test prompt available")
        
        result = self.sdk.update_prompt(
            prompt_id=self.test_prompt_id,
            prompt="Updated test prompt with {input}"
        )
        assert result is not None
        print(f"Update prompt result: {result}")

    def test_rename_prompt(self):
        """Test renaming a prompt."""
        if not hasattr(self, 'test_prompt_id'):
            pytest.skip("No test prompt available")
        
        new_name = f"RenamedPrompt_{uuid.uuid4().hex[:8]}"
        result = self.sdk.rename_prompt(
            prompt_id=self.test_prompt_id,
            new_name=new_name
        )
        assert result is not None
        print(f"Rename prompt result: {result}")

    # ==========================================================================
    # Extension Tests
    # ==========================================================================

    def test_get_extensions(self):
        """Test getting all extensions."""
        result = self.sdk.get_extensions()
        assert result is not None
        print(f"Get extensions result: {result}")

    def test_get_extension_settings(self):
        """Test getting extension settings."""
        result = self.sdk.get_extension_settings()
        print(f"Get extension settings result: {result}")

    def test_get_agent_extensions(self):
        """Test getting agent extensions."""
        result = self.sdk.get_agent_extensions(self.test_agent_id)
        print(f"Get agent extensions result: {result}")

    # ==========================================================================
    # Memory/Learning Tests
    # ==========================================================================

    def test_learn_text(self):
        """Test teaching agent text content."""
        result = self.sdk.learn_text(
            agent_id=self.test_agent_id,
            user_input="Learn this information",
            text="This is test content to learn.",
            collection_number="0"
        )
        assert result is not None
        print(f"Learn text result: {result}")

    def test_get_agent_memories(self):
        """Test getting agent memories."""
        result = self.sdk.get_agent_memories(
            agent_id=self.test_agent_id,
            user_input="test",
            limit=10,
            min_relevance_score=0.0
        )
        # May be empty if no memories yet
        print(f"Get agent memories result: {result}")

    def test_get_unique_external_sources(self):
        """Test getting unique external sources."""
        result = self.sdk.get_unique_external_sources(agent_id=self.test_agent_id)
        print(f"Get unique external sources result: {result}")

    def test_get_browsed_links(self):
        """Test getting browsed links."""
        result = self.sdk.get_browsed_links(
            agent_id=self.test_agent_id,
            collection_number="0"
        )
        print(f"Get browsed links result: {result}")

    def test_get_memories_external_sources(self):
        """Test getting memories external sources."""
        result = self.sdk.get_memories_external_sources(
            agent_id=self.test_agent_id,
            collection_number="0"
        )
        print(f"Get memories external sources result: {result}")

    # ==========================================================================
    # Prompt Agent / Chat Tests
    # ==========================================================================

    def test_prompt_agent(self):
        """Test prompting an agent."""
        result = self.sdk.prompt_agent(
            agent_id=self.test_agent_id,
            prompt_name="Chat",
            prompt_args={
                "user_input": "Hello, this is a test.",
                "disable_memory": True,
            }
        )
        # May fail if agent is not configured, but endpoint should respond
        print(f"Prompt agent result: {result}")

    # ==========================================================================
    # Feedback Tests
    # ==========================================================================

    def test_positive_feedback(self):
        """Test submitting positive feedback."""
        result = self.sdk.positive_feedback(
            agent_id=self.test_agent_id,
            message="Good response",
            user_input="Test input",
            feedback="Great job!",
            conversation_id=""
        )
        print(f"Positive feedback result: {result}")

    def test_negative_feedback(self):
        """Test submitting negative feedback."""
        result = self.sdk.negative_feedback(
            agent_id=self.test_agent_id,
            message="Bad response",
            user_input="Test input",
            feedback="Could be better",
            conversation_id=""
        )
        print(f"Negative feedback result: {result}")

    # ==========================================================================
    # Helper Method Tests
    # ==========================================================================

    def test_get_agent_id_by_name(self):
        """Test getting agent ID by name."""
        if hasattr(self.sdk, 'get_agent_id_by_name'):
            result = self.sdk.get_agent_id_by_name("XT")
            print(f"Get agent ID by name result: {result}")

    def test_get_chain_id_by_name(self):
        """Test getting chain ID by name."""
        if hasattr(self.sdk, 'get_chain_id_by_name'):
            result = self.sdk.get_chain_id_by_name("Smart Chat")
            print(f"Get chain ID by name result: {result}")

    def test_get_conversation_id_by_name(self):
        """Test getting conversation ID by name."""
        if hasattr(self.sdk, 'get_conversation_id_by_name'):
            result = self.sdk.get_conversation_id_by_name("Chat")
            print(f"Get conversation ID by name result: {result}")

    # ==========================================================================
    # OpenAI Compatibility Tests
    # ==========================================================================

    def test_chat_completions(self):
        """Test chat completions endpoint."""
        result = self.sdk.chat_completions(
            model="XT",  # Default agent name
            messages=[
                {"role": "user", "content": "Hello, this is a test."}
            ],
            user="test_conversation"
        )
        # This may timeout or require proper agent setup
        print(f"Chat completions result: {result}")


class TestAGiXTSDKCleanup:
    """Additional cleanup tests that should run last."""

    @classmethod
    def setup_class(cls):
        """Set up for cleanup tests."""
        cls.sdk = AGiXTSDK(base_uri=BASE_URI, verbose=True)
        
        # Register a test user
        test_email = f"cleanup_test_{uuid.uuid4()}@example.com"
        cls.sdk.register_user(
            email=test_email,
            first_name="Cleanup",
            last_name="User"
        )

    def test_delete_agent(self):
        """Test deleting an agent."""
        # Create an agent to delete
        agent_name = f"DeleteTestAgent_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_agent(agent_name=agent_name, settings={})
        
        agent_id = None
        if isinstance(result, dict):
            agent_id = result.get("id") or result.get("agent_id")
        
        if not agent_id:
            # Try to find by name
            agents = self.sdk.get_agents()
            for agent in agents:
                if isinstance(agent, dict) and agent.get("name") == agent_name:
                    agent_id = agent.get("id")
                    break
        
        if agent_id:
            delete_result = self.sdk.delete_agent(agent_id)
            assert delete_result is not None
            print(f"Delete agent result: {delete_result}")
        else:
            pytest.skip("Could not get agent ID for deletion test")

    def test_delete_chain(self):
        """Test deleting a chain."""
        # Create a chain to delete
        chain_name = f"DeleteTestChain_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_chain(chain_name=chain_name)
        
        chain_id = None
        if isinstance(result, dict):
            chain_id = result.get("id") or result.get("chain_id")
        
        if chain_id:
            delete_result = self.sdk.delete_chain(chain_id)
            assert delete_result is not None
            print(f"Delete chain result: {delete_result}")
        else:
            pytest.skip("Could not get chain ID for deletion test")

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        # First create an agent for the conversation
        agent_name = f"ConvDeleteAgent_{uuid.uuid4().hex[:8]}"
        agent_result = self.sdk.add_agent(agent_name=agent_name, settings={})
        
        agent_id = None
        if isinstance(agent_result, dict):
            agent_id = agent_result.get("id") or agent_result.get("agent_id")
        
        if not agent_id:
            pytest.skip("Could not create agent for conversation deletion test")
            return
        
        # Create a conversation to delete
        conv_name = f"DeleteTestConv_{uuid.uuid4().hex[:8]}"
        result = self.sdk.new_conversation(
            agent_id=agent_id,
            conversation_name=conv_name,
            conversation_content=[]
        )
        
        conv_id = None
        if isinstance(result, dict):
            conv_id = result.get("id") or result.get("conversation_id")
        
        if conv_id:
            delete_result = self.sdk.delete_conversation(conv_id)
            assert delete_result is not None
            print(f"Delete conversation result: {delete_result}")
        
        # Clean up the agent
        try:
            self.sdk.delete_agent(agent_id)
        except Exception:
            pass

    def test_delete_prompt(self):
        """Test deleting a prompt."""
        # Create a prompt to delete
        prompt_name = f"DeleteTestPrompt_{uuid.uuid4().hex[:8]}"
        result = self.sdk.add_prompt(
            prompt_name=prompt_name,
            prompt="Test prompt",
            prompt_category="Default"
        )
        
        prompt_id = None
        if isinstance(result, dict):
            prompt_id = result.get("id") or result.get("prompt_id")
        
        if prompt_id:
            delete_result = self.sdk.delete_prompt(prompt_id)
            assert delete_result is not None
            print(f"Delete prompt result: {delete_result}")
        else:
            pytest.skip("Could not get prompt ID for deletion test")


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])


if __name__ == "__main__":
    run_tests()
