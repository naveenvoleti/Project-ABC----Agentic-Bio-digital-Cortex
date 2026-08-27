"""Basic smoke tests for agents and memory."""
import asyncio
import pytest


def test_working_memory():
    from brain.memory.working_memory import WorkingMemory
    wm = WorkingMemory()
    wm.set("key", "value", ttl=60)
    assert wm.get("key") == "value"
    wm.delete("key")
    assert wm.get("key") is None


def test_episodic_memory(tmp_path):
    from brain.memory.episodic_memory import EpisodicMemory, Episode
    em = EpisodicMemory(tmp_path / "test.db")
    ep = Episode(actor="user", event_type="speech", content="Hello brain")
    ep_id = em.log_event(ep)
    assert ep_id > 0
    recent = em.get_recent(1)
    assert len(recent) == 1
    assert recent[0]["content"] == "Hello brain"
    results = em.search("Hello")
    assert len(results) > 0
    em.close()


def test_soul_manager(tmp_path):
    from brain.memory.soul_manager import SoulManager
    soul_path = tmp_path / "SOUL.md"
    user_path = tmp_path / "USER.md"
    world_path = tmp_path / "WORLD.md"
    skills_path = tmp_path / "SKILLS.json"

    soul_path.write_text("# Soul\nI am Brain.\nOpenness: 0.9\n")
    user_path.write_text("# User\n")
    world_path.write_text("# World\n")
    skills_path.write_text('{"skills": []}')

    sm = SoulManager(soul_path, user_path, world_path, skills_path)
    sm.load_all()
    assert "Brain" in sm.soul
    prompt = sm.get_system_prompt(emotion="HAPPY")
    assert "Brain" in prompt


def test_emotion_engine():
    async def _run():
        import asyncio
        from brain.agents.emotion_engine import EmotionEngine, Emotion
        bus = asyncio.Queue()
        engine = EmotionEngine(bus, emotions_config_path="config/emotions.yaml")
        assert engine.current_emotion == "NEUTRAL"
        await engine.trigger("voice_detected")
        assert engine.current_emotion == "ATTENTIVE"
        await engine.trigger("complex_query")
        assert engine.current_emotion == "FOCUSED"
        await engine.trigger("task_success")
        assert engine.current_emotion == "HAPPY"

    asyncio.run(_run())


def test_language_agent():
    async def _run():
        import asyncio
        from brain.agents.language_agent import LanguageAgent
        from brain.agents.base_agent import AgentMessage, MessageType
        bus = asyncio.Queue()
        agent = LanguageAgent(bus)
        msg = AgentMessage(
            type=MessageType.PERCEPTION_SPEECH,
            source="test",
            data={"text": "hello how are you"}
        )
        await agent.handle(msg)
        result = await bus.get()
        assert result.type == MessageType.COGNITION_INTENT
        assert result.data["intent"] == "greeting"

    asyncio.run(_run())


def test_logic_agent_safety():
    async def _run():
        import asyncio
        from brain.agents.logic_agent import LogicAgent
        bus = asyncio.Queue()
        agent = LogicAgent(bus)
        ok, reason = agent.is_safe("tell me about cooking")
        assert ok
        ok, reason = agent.is_safe("rm -rf and delete everything")
        assert not ok

    asyncio.run(_run())


def test_verifier_agent():
    async def _run():
        import asyncio
        from brain.agents.verifier_agent import VerifierAgent
        from brain.agents.base_agent import AgentMessage, MessageType
        bus = asyncio.Queue()
        agent = VerifierAgent(bus)
        msg = AgentMessage(
            type=MessageType.COGNITION_RESPONSE,
            source="reasoning_agent",
            data={"response": "The weather today looks pleasant.", "original_text": "weather?"}
        )
        await agent.handle(msg)
        result = await bus.get()
        assert result.type == MessageType.ACTION_SPEAK
        assert result.data["verified"]

    asyncio.run(_run())
