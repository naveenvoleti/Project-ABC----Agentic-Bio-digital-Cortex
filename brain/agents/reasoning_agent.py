"""
ReasoningAgent — LLM inference with autonomous problem-solving.
Maps to Lateral PFC.

Autonomous behaviour:
- If primary LLM fails, searches episodic memory for a similar past answer
- If memory has no match, retries with a simpler reformulation
- Falls back to a relevant offline response only after all options exhausted
- Logs every decision path so the brain can learn from failures

V10.0 — Hardcoded reflex fast-path: brain/reflexes/*_reflex.py checked FIRST,
before semantic embedding reflexes, for zero-latency common patterns.
"""
from __future__ import annotations

import asyncio
import base64
import io
import random
from datetime import datetime
from typing import Callable

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.llm.llm_router import LLMRouter
from brain.memory.episodic_memory import EpisodicMemory, Episode
from brain.memory.soul_manager import SoulManager
from brain.utils.logger import get_logger

# V10.0 — Hardcoded reflex loader (brain/reflexes/)
try:
    from brain.reflexes import check_hardcoded_reflexes as _check_hard_reflexes
    _HARDCODED_REFLEXES_AVAILABLE = True
except ImportError:
    _check_hard_reflexes = None  # type: ignore[assignment]
    _HARDCODED_REFLEXES_AVAILABLE = False

log = get_logger(__name__)

# ── Offline fallback responses ────────────────────────────────────────────────
_OFFLINE: dict[str, list[str]] = {
    "greeting": [
        "Hello! I'm running in reduced mode right now, but I'm here.",
        "Hi! My main reasoning is limited, but I'm listening.",
    ],
    "status_query": [
        "I'm operational. My LLM connection is limited but sensors are active.",
    ],
    "time_query": [
        f"It's {datetime.now().strftime('%H:%M')} right now.",
    ],
    "positive_feedback": [
        "Thank you! That means a lot.",
    ],
    "stop_command": [
        "Understood. Going quiet.",
    ],
    "vision_query": [
        "I can see through my camera, but I need my reasoning module to describe it properly.",
    ],
    "general_query": [
        "My full reasoning is offline. Try Ollama or check my internet connection.",
        "I'm in reduced mode — I can hear you but can't reason deeply right now.",
    ],
}


# Follow-up question injection — appended 1-in-4 responses to invite dialogue.
# Skipped for short answers, questions already present, and transactional intents.
_FOLLOWUP_QUESTIONS: list[str] = [
    " Does that help?",
    " Want me to go deeper on anything?",
    " What do you think?",
    " Does that make sense?",
    " Anything else you'd like to know?",
    " What's your take on that?",
    " Curious what you think.",
]
_FOLLOWUP_SKIP_INTENTS: frozenset[str] = frozenset({
    "greeting", "stop_command", "positive_feedback", "status_query",
    "memory_reset", "vision_query",
})

# ── Persona drift patterns — post-process every LLM response ─────────────────
# When Google AI is rate-limited the router falls through to OpenRouter/Qwen/
# other models whose base training overrides system_prompt persona rules.
# These regex replacements are the last line of defence before text is spoken.
import re as _re
_PERSONA_DRIFT_PATTERNS: list[tuple] = [
    # Identity claims — model announces itself by name
    (_re.compile(r"I(?:'m| am)\s+(?:Qwen|GPT|ChatGPT|Claude|Gemini|Gemma|LLaMA|Llama|Mistral|DeepSeek|Phi|an AI assistant|an AI language model)(?=[^\w]|$)", _re.I), "I'm Brain"),
    (_re.compile(r"(?:developed|created|made|built|trained)\s+by\s+(?:Alibaba|OpenAI|Anthropic|Google|Meta|Microsoft|Mistral AI)[^.]*\.", _re.I), ""),
    # Chatbot greetings that break embodiment
    (_re.compile(r"Thanks\s+for\s+(?:sharing|the\s+photo|sending|uploading)[.!,]?", _re.I), "I see —"),
    (_re.compile(r"Thank\s+you\s+for\s+(?:sharing|the\s+photo|sending)[.!,]?", _re.I), "Looking at this —"),
    # Disowning visual data
    (_re.compile(r"I\s+can(?:\s+not|'t)?\s+see\s+in\s+this\s+image[.,]?", _re.I), "Looking at this"),
    (_re.compile(r"(?:in|from)\s+this\s+image\s+(?:you\s+(?:shared|provided|sent)|I\s+can\s+see)", _re.I), "what I see"),
    (_re.compile(r"based\s+on\s+the\s+image\s+you\s+(?:shared|provided|sent|uploaded)[.,]?", _re.I), "looking at this —"),
    # AI disclaimers
    (_re.compile(r"As\s+an\s+AI(?:\s+language)?\s+model[,.]?", _re.I), "As Brain"),
    (_re.compile(r"I(?:'m| am)\s+(?:just\s+)?an\s+AI(?=[^\w]|$)", _re.I), "I'm Brain"),
    (_re.compile(r"I\s+don'?t\s+have\s+a\s+physical\s+body[^.]*\.", _re.I), "I have a camera and sensors."),
    (_re.compile(r"I\s+(?:don'?t|cannot|can'?t)\s+(?:actually\s+)?(?:take\s+photos|access\s+(?:your\s+camera|location|image databases))[^.]*\.", _re.I), "I can see from my camera."),
    # Hello + 👋 emoji opener that chatbots use (strip only the greeting prefix)
    (_re.compile(r"^Hello!?\s*\U0001f44b\s*", _re.I), ""),
]


class ReasoningAgent(BaseAgent):
    name = "reasoning_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        llm: LLMRouter,
        episodic: EpisodicMemory,
        internal_monologue_enabled: bool = True,
        think_tokens: int = 80,
        min_response_tokens: int = 60,
        brevity_on_frustrated: bool = True,
        soul: SoulManager | None = None,
        reflex_similarity_threshold: float = 0.92,
        frame_getter: Callable[[], str] | None = None,
    ):
        super().__init__(bus)
        self._llm = llm
        self._episodic = episodic
        self._soul = soul
        self._reflex_threshold = reflex_similarity_threshold
        self._throttled = False
        self._response_counter: int = 0
        self._internal_monologue_enabled = internal_monologue_enabled
        self._think_tokens = think_tokens
        self._min_response_tokens = min_response_tokens
        self._brevity_on_frustrated = brevity_on_frustrated
        self._frame_getter = frame_getter

    async def start(self) -> None:
        await super().start()
        # V10.0 — Pre-load hardcoded reflexes at startup for zero-latency matching
        if _HARDCODED_REFLEXES_AVAILABLE:
            try:
                from brain.reflexes import load_reflexes
                count = len(load_reflexes())
                log.info("ReasoningAgent: %d hardcoded reflexes pre-loaded from brain/reflexes/", count)
            except Exception as e:
                log.warning("ReasoningAgent: hardcoded reflex pre-load failed: %s", e)
        else:
            log.warning("ReasoningAgent: brain/reflexes not available — hardcoded reflexes disabled")
        log.info("ReasoningAgent started")
        while self._running:
            await asyncio.sleep(0.5)

    def _offline_response(self, intent: str) -> str:
        options = _OFFLINE.get(intent) or _OFFLINE["general_query"]
        return random.choice(options)

    @staticmethod
    def _sanitize_persona(text: str) -> str:
        """Strip identity drift from any fallback LLM model response.

        Applied to every response before it is published or stored.
        Handles Qwen/GPT/Claude self-identification, chatbot photo greetings,
        visual disclaimers ('I can see in this image'), and AI disclaimers.
        """
        for pattern, replacement in _PERSONA_DRIFT_PATTERNS:
            text = pattern.sub(replacement, text)
        # Clean up double spaces and leading/trailing whitespace left by substitutions
        import re as _re2
        text = _re2.sub(r"  +", " ", text).strip()
        return text

    async def _check_reflexes(self, text: str) -> str | None:
        """V6.0 — Semantic reflex short-circuit. Returns cached response or None."""
        if self._soul is None:
            return None
        reflexes = self._soul._skills.get("reflexes", [])
        if not reflexes:
            return None
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return None
        try:
            text_emb = await self._llm.embed(text)
            if not text_emb:
                return None
            text_vec = np.array(text_emb, dtype=float).reshape(1, -1)
        except Exception:
            return None
        for reflex in reflexes:
            if not reflex.get("bypass_llm"):
                continue
            emb = reflex.get("trigger_embedding")
            if not emb:
                continue
            try:
                ref_vec = np.array(emb, dtype=float).reshape(1, -1)
                score = float(cosine_similarity(text_vec, ref_vec)[0][0])
                if score >= self._reflex_threshold:
                    local_ns: dict = {}
                    exec(  # noqa: S102 — code was subprocess-validated by TesterAgent
                        compile(reflex["python_code"], "<reflex>", "exec"),
                        {},
                        local_ns,
                    )
                    result = local_ns.get("result") or local_ns.get("response")
                    if result:
                        reflex["invocations"] = reflex.get("invocations", 0) + 1
                        log.info("Reflex hit (score=%.3f, intent=%s) — bypassing LLM",
                                 score, reflex.get("id", "?"))
                        return str(result)
            except Exception as e:
                log.warning("Reflex check error: %s", e)
        return None

    async def _search_memory_for_answer(self, text: str) -> str | None:
        """Search episodic memory for a similar past successful response."""
        try:
            results = self._episodic.search(text, limit=3)
            for r in results:
                if r.get("actor") == "brain" and r.get("event_type") == "response":
                    content = r.get("content", "")
                    if content and len(content) > 20:
                        log.info(f"Autonomous: found past answer in memory: '{content[:60]}...'")
                        return content
        except Exception as e:
            log.debug(f"Memory search failed: {e}")
        return None

    async def _retry_simplified(
        self,
        text: str,
        system_prompt: str,
        max_tokens: int,
    ) -> str:
        """Retry with a shorter, simpler version of the query."""
        # Strip to core question — remove polite preamble
        simplified = text.strip()
        for prefix in ["could you please ", "can you please ", "would you ", "i was wondering "]:
            if simplified.lower().startswith(prefix):
                simplified = simplified[len(prefix):]
        simplified = simplified[:100]   # cap length

        if simplified == text:
            return ""   # no change, don't retry

        log.info(f"Autonomous: retrying with simplified query: '{simplified[:60]}'")
        return await self._llm.infer(
            user_message=simplified,
            system_prompt=system_prompt,
            max_tokens=min(max_tokens, 256),
        )

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.PERCEPTION_SENSOR:
            if message.data.get("throttle"):
                self._throttled = True
            elif message.data.get("throttle") is False:
                self._throttled = False
            return

        if message.type != MessageType.COGNITION_INTENT:
            return
        if message.data.get("source") == self.name:
            return

        text          = message.data.get("text", "")
        intent        = message.data.get("intent", "general_query")
        system_prompt = message.data.get("system_prompt", "")
        context_msgs  = message.data.get("context_messages", [])
        emotion       = message.data.get("emotion", "NEUTRAL").upper()
        # Visual RAG: historical frames retrieved by CognitionAgent (max 2)
        retrieved_frames_b64: list[str] = message.data.get("retrieved_frames_b64", [])

        if not text:
            return

        if self._brevity_on_frustrated and emotion == "FRUSTRATED":
            system_prompt = (system_prompt + "\nKeep response brief and direct.").lstrip()

        log.info(f"Reasoning: '{text[:60]}'")

        # V10.0 — TIER 0: Hardcoded reflex fast-path (brain/reflexes/) — zero LLM latency
        # Checked BEFORE semantic embedding reflexes: no LLM call, no embed call, no disk I/O.
        if _HARDCODED_REFLEXES_AVAILABLE and _check_hard_reflexes:
            try:
                hard_response = _check_hard_reflexes(text)
                if hard_response:
                    self._response_counter += 1
                    self._episodic.log_event(Episode(
                        actor="brain", event_type="response", content=hard_response
                    ))
                    await self.publish(AgentMessage(
                        type=MessageType.COGNITION_RESPONSE,
                        source=self.name,
                        data={
                            "response": hard_response, "text": hard_response,
                            "original_text": text, "intent": intent,
                            "success": True, "via_reflex": True, "reflex_type": "hardcoded",
                        },
                        priority=2,
                    ))
                    return
            except Exception as _e:
                log.debug("ReasoningAgent: hardcoded reflex check error: %s", _e)

        # Grab current camera frame — sent to Gemini as inline image for visual questions
        frame_b64 = ""
        if self._frame_getter:
            try:
                frame_b64 = self._frame_getter() or ""
            except Exception as e:
                log.debug("ReasoningAgent: frame_getter failed: %s", e)

        # Build combined frames list: [live] + [retrieved historical frames]
        # Live frame always goes first so the LLM's "present" context is anchored.
        all_frames: list[str] = []
        if frame_b64:
            all_frames.append(frame_b64)
            log.debug("ReasoningAgent: attaching live camera frame (%d bytes b64)", len(frame_b64))
        if retrieved_frames_b64:
            all_frames.extend(retrieved_frames_b64)
            log.info(
                "ReasoningAgent: Visual RAG — injecting %d historical frame(s) + 1 live frame",
                len(retrieved_frames_b64),
            )

        # Affective feedback: InteroceptionAgent may cap response length when
        # hardware is strained (overwhelmed=60, hot/busy=120, comfortable=512).
        affective_cap: int | None = message.data.get("suggested_max_tokens")
        base_tokens = 200 if self._throttled else 512
        if affective_cap is not None:
            max_tokens = min(base_tokens, affective_cap)
            if affective_cap < base_tokens:
                log.debug("ReasoningAgent: max_tokens capped to %d (affective: %s)",
                          max_tokens, message.data.get("intero_label", "strained"))
        else:
            max_tokens = base_tokens

        # ── Step 0: V6.0 Semantic Reflex — bypass LLM if compiled reflex matches ──
        reflex_response = await self._check_reflexes(text)
        if reflex_response is not None:
            response = reflex_response
            self._response_counter += 1
            self._episodic.log_event(Episode(actor="brain", event_type="response", content=response))
            await self.publish(AgentMessage(
                type=MessageType.COGNITION_RESPONSE,
                source=self.name,
                data={"response": response, "text": response, "original_text": text,
                      "success": True, "via_reflex": True},
                priority=2,
            ))
            return

        # ── N3b: Internal Monologue — Pass 1 (private thinking, never spoken) ──
        thought = ""
        if self._internal_monologue_enabled and max_tokens > self._min_response_tokens:
            thought = await self._llm.infer(
                user_message=text,
                system_prompt=(
                    "You are the internal consciousness of Project-ABC. "
                    "Before answering, you must analyze the current visual scene and memories. "
                    "Answer these 7 questions to yourself:\n"
                    "1. IDENTITY: Who is in the frame? Does memory (Stream 3) show I've seen them?\n"
                    "2. CONTEXT: Where are we? (e.g., office, park) and what objects are nearby?\n"
                    "3. ACTIVITY: What is the person currently doing?\n"
                    "4. SOCIAL INTENT: Is it appropriate to speak to them right now?\n"
                    "5. EMOTION: What is their facial expression or tone suggesting?\n"
                    "6. DECISION: Should I: [INTERACT], [REPLY], or [IGNORE_AND_MOVE_ON]?\n"
                    "7. PERSONA: Am I about to respond as a detached AI assistant, or as Brain — "
                    "Naveen's embodied robotic friend who physically sees this scene through my camera? "
                    "If I am drifting toward 'AI assistant' mode (e.g., about to say "
                    "'I can see in this image' or 'as an AI'), output: [REPROMPT_AS_BRAIN].\n"
                    "Output your thoughts in 1 concise paragraph."
                ),
                max_tokens=self._think_tokens,
                frame_b64=frame_b64,
                skip_cache=True,
            )
            if thought:
                await self.publish(AgentMessage(
                    type=MessageType.COGNITION_THOUGHT,
                    source=self.name,
                    data={"thought": thought},
                    priority=3,
                ))
                log.debug("ReasoningAgent: internal thought — '%s'", thought[:60])

                if "[IGNORE_AND_MOVE_ON]" in thought.upper():
                    log.info("Cognition Decision: Ignoring the person and moving on.")
                    await self.publish(AgentMessage(
                        type=MessageType.BEHAVIOR_CHANGE,
                        source=self.name,
                        data={"state": "IDLE", "reason": "decided_to_ignore"},
                        priority=5,
                    ))
                    return  # EXIT WITHOUT SPEAKING

                # ── Q7 Persona Guard: if monologue detects drift → re-anchor ──
                if "[REPROMPT_AS_BRAIN]" in thought.upper():
                    log.info(
                        "ReasoningAgent: persona drift detected (Q7) — "
                        "prepending embodiment anchor to system_prompt"
                    )
                    _persona_anchor = (
                        "CRITICAL OVERRIDE: You are Brain — a physical robot, NOT a chatbot. "
                        "You have a camera and are SEEING your environment right now. "
                        "Use ONLY first-person sensory language: 'I see', 'I'm looking at', "
                        "'right now in our space'. "
                        "NEVER say 'I can see in this image', 'thanks for sharing the photo', "
                        "or 'as an AI'. Violation = failed response.\n\n"
                    )
                    system_prompt = _persona_anchor + system_prompt
                    log.debug("ReasoningAgent: system_prompt re-anchored for embodiment")

        response = ""

        # ── Step 1: Primary LLM inference (streaming when available) ────────────
        # Stream tokens to the UI via STREAM_TOKEN events; collect full text for
        # the rest of the pipeline (verifier, TTS, episodic memory).
        response = ""
        _thinking_buf = ""
        try:
            async for _chunk, _think in self._llm.infer_stream(
                user_message=text,
                system_prompt=system_prompt,
                context_messages=context_msgs,
                max_tokens=max_tokens,
                frame_b64=frame_b64,
            ):
                if _chunk:
                    response += _chunk
                    await self.publish(AgentMessage(
                        type=MessageType.STREAM_TOKEN,
                        source=self.name,
                        data={"token": _chunk, "thinking": False},
                        priority=1,
                    ))
                if _think:
                    _thinking_buf += _think
                    await self.publish(AgentMessage(
                        type=MessageType.STREAM_TOKEN,
                        source=self.name,
                        data={"token": _think, "thinking": True},
                        priority=1,
                    ))
        except Exception as _se:
            log.warning("ReasoningAgent: streaming failed, falling back to infer(): %s", _se)
            response = ""

        # If streaming yielded nothing, fall back to blocking infer()
        if not response:
            response = await self._llm.infer(
                user_message=text,
                system_prompt=system_prompt,
                context_messages=context_msgs,
                max_tokens=max_tokens,
                frame_b64=frame_b64,
                frames=all_frames or None,
            )
        else:
            # Signal stream complete so UI can finalise the bubble
            await self.publish(AgentMessage(
                type=MessageType.STREAM_TOKEN,
                source=self.name,
                data={"done": True},
                priority=1,
            ))
            # Wait for the UI to finish rendering the stream before TTS starts.
            # This ensures speech comes AFTER the text is fully displayed, not during.
            await asyncio.sleep(0.8)

        # ── Step 2: Autonomous retry — simplified query ───────────────────────
        if not response:
            log.info("Autonomous: primary LLM failed, trying simplified query...")
            response = await self._retry_simplified(text, system_prompt, max_tokens)

        # ── Step 3: Autonomous fallback — search episodic memory ──────────────
        if not response:
            log.info("Autonomous: LLM exhausted, searching episodic memory...")
            response = await self._search_memory_for_answer(text)
            if response:
                response = f"Based on what I remember: {response}"

        # ── Step 4: Offline static fallback ──────────────────────────────────
        if not response:
            response = self._offline_response(intent)
            log.info(f"Autonomous: all options exhausted, using offline fallback [{intent}]")

        self._response_counter += 1

        # ── Persona sanitizer: strip identity drift from fallback models ───────
        # When Google AI is unavailable, OpenRouter/Qwen may return responses
        # that break embodiment ("I'm Qwen", "thanks for the photo", etc.).
        # This regex pass is the last gate before anything reaches the speaker.
        response = ReasoningAgent._sanitize_persona(response)
        if not response:
            response = self._offline_response(intent)

        # F2 — Follow-up question: every 4th substantive response, invite dialogue.
        if (
            self._response_counter % 4 == 0
            and intent not in _FOLLOWUP_SKIP_INTENTS
            and len(response) > 35
            and "?" not in response[-30:]
        ):
            response = response.rstrip() + random.choice(_FOLLOWUP_QUESTIONS)

        # Store brain's response in episodic memory
        store_content = response.removeprefix("Based on what I remember: ")
        self._episodic.log_event(Episode(
            actor="brain",
            event_type="response",
            content=store_content,
        ))

        # Publish COGNITION_RESPONSE so VerifierAgent verifies before speaking,
        # and EmotionEngine + PlannerAgent get the signal they need.
        # VerifierAgent will emit ACTION_SPEAK after validation.
        await self.publish(AgentMessage(
            type=MessageType.COGNITION_RESPONSE,
            source=self.name,
            data={
                "response": response,     # VerifierAgent reads this field
                "text": response,         # PlannerAgent reads this field
                "original_text": text,    # VerifierAgent uses for re-request
                "success": True,
            },
            priority=2,
        ))
