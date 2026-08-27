"""Brain agents — each maps to a human brain region."""
from .base_agent import BaseAgent, AgentMessage, MessageType
from .attention_agent import AttentionAgent
from .interoception_agent import InteroceptionAgent
from .metacognition_agent import MetacognitionAgent
from .theory_of_mind_agent import TheoryOfMindAgent
from .temporal_reasoning_agent import TemporalReasoningAgent
from .intrinsic_motivation_agent import IntrinsicMotivationAgent
from .ideation_agent import IdeationAgent
from .imagination_agent import ImaginationAgent
# V10.0 — Sentient Embodied Autonomous Brain agents
from .goal_stack_agent import GoalStackAgent
from .proprioception_agent import ProprioceptionAgent
from .learning_agent import LearningAgent
from .binding_agent import BindingAgent

__all__ = [
    "BaseAgent", "AgentMessage", "MessageType",
    "AttentionAgent", "InteroceptionAgent", "MetacognitionAgent",
    "TheoryOfMindAgent", "TemporalReasoningAgent", "IntrinsicMotivationAgent",
    "IdeationAgent", "ImaginationAgent",
    # V10.0
    "GoalStackAgent", "ProprioceptionAgent", "LearningAgent", "BindingAgent",
]
