"""
DreamAgent — nightly memory consolidation.
Maps to sleep-phase memory replay. Runs on a systemd timer at 2AM.
Distills episodic logs → semantic facts → updates soul files.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from brain.agents.base_agent import BaseAgent, AgentMessage, MessageType
from brain.memory.episodic_memory import EpisodicMemory
from brain.memory.semantic_memory import SemanticMemory
from brain.memory.soul_manager import SoulManager
from brain.llm.llm_router import LLMRouter
from brain.utils.logger import get_logger

log = get_logger(__name__)

DREAM_JOURNAL = Path("data/DREAM_LOG.md")
MIN_EPISODES = 5


class DreamAgent(BaseAgent):
    name = "dream_agent"

    def __init__(
        self,
        bus: asyncio.Queue,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        soul: SoulManager,
        llm: LLMRouter,
        pattern_threshold: int = 5,
        prune_after_days: int = 7,
        promote_after_invocations: int = 100,
    ):
        super().__init__(bus)
        self._episodic = episodic
        self._semantic = semantic
        self._soul = soul
        self._llm = llm
        self._pattern_threshold = pattern_threshold
        self._prune_after_days = prune_after_days
        self._promote_after_invocations = promote_after_invocations
        # Injected by Orchestrator after _build_agents()
        self._pattern_registry: dict[str, list[str]] = {}

    def set_pattern_registry(self, registry: dict[str, list[str]]) -> None:
        """Inject the Orchestrator's live pattern registry for distillation."""
        self._pattern_registry = registry

    async def start(self) -> None:
        await super().start()
        log.info("DreamAgent started")
        while self._running:
            await asyncio.sleep(60)

    async def run_dream_cycle(self) -> str:
        log.info("Dream cycle starting...")
        today_episodes = self._episodic.get_since(hours=24)

        if len(today_episodes) < MIN_EPISODES:
            log.info(f"Dream cycle skipped: only {len(today_episodes)} episodes (min {MIN_EPISODES})")
            return "skipped"

        # Rank by emotional salience — emotionally charged moments get priority
        # in the dream narrative, mirroring how sleep consolidation works in humans.
        salient = self._episodic.get_salient(n=50)
        episode_text = "\n".join(
            f"- [{e['timestamp'][:16]}] [{e['actor']}] ({e['event_type']})"
            + (f" [emotion: {e['emotion']}]" if e.get("emotion") else "")
            + f" {e['content']}"
            for e in salient
        )

        summary_prompt = f"""You are Brain reviewing your memories from today.
Here are the events that happened:

{episode_text}

Please:
1. Write a short diary-style summary (3-5 sentences) of today
2. List 3-5 key facts you learned about the user or environment (as bullet points)
3. Note any recurring patterns
4. Suggest one personality insight

Format:
## Summary
...

## Facts Learned
- ...

## Patterns
...

## Personality Insight
...
"""

        result = await self._llm.infer(
            user_message="Please consolidate today's memories.",
            system_prompt=summary_prompt,
            max_tokens=600,
        )

        if not result:
            log.warning("Dream agent: LLM returned empty result")
            return "failed"

        # Extract facts and store in semantic memory
        lines = result.split("\n")
        in_facts = False
        for line in lines:
            line = line.strip()
            if "## Facts Learned" in line:
                in_facts = True
                continue
            if line.startswith("## ") and in_facts:
                in_facts = False
            if in_facts and line.startswith("-"):
                fact = line.lstrip("- ").strip()
                if fact:
                    embedding = await self._llm.embed(fact)
                    self._semantic.upsert(
                        content=fact,
                        embedding=embedding,
                        category="user_fact",
                        source="dream",
                    )

        # Update soul with personality insight
        if "## Personality Insight" in result:
            insight = result.split("## Personality Insight")[-1].strip()
            if insight:
                await asyncio.to_thread(
                    self._soul.update_soul,
                    f"### Daily Insight ({datetime.now().strftime('%Y-%m-%d')})\n{insight}"
                )

        # Write dream journal (ensure data/ directory exists)
        def _write_journal():
            DREAM_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
            with open(DREAM_JOURNAL, "a", encoding="utf-8") as f:
                f.write(entry)

        entry = f"\n\n## {datetime.now().strftime('%Y-%m-%d')}\n{result}"
        await asyncio.to_thread(_write_journal)

        # Build relationship arc (weekly pattern analysis)
        await self._build_relationship_arc()

        # F6 — Personality evolution: tiny OCEAN trait shifts based on today's interactions
        await self._evolve_personality()

        # F7 — Memory decay: older unreinforced memories lose importance
        decayed = self._episodic.decay_old_importance(days_threshold=7, decay_factor=0.85)
        log.info(f"Memory decay: {decayed} old memories faded")

        # F12 — Habit detection: find recurring patterns across last 7 days
        await self._detect_habits()

        # v5.0 — Emotional-weight pruning: delete memories with low emotional weight
        pruned_emo = self._episodic.prune_low_emotion(days_threshold=3, importance_max=0.3)
        log.info(f"Emotional pruning: {pruned_emo} low-weight memories removed")

        # v5.0 — Convert recurring high-importance episodes to semantic facts
        await self._promote_recurring_to_semantic()

        # N1b — Neuroplasticity: learn from user corrections during this cycle
        await self._consolidate_corrections()

        # Purge old episodes (age-based, existing behaviour)
        pruned = self._episodic.purge_old(days=30)
        log.info(f"Dream cycle complete — {pruned} old episodes pruned")

        # V6.0 — Distill repeated LLM patterns into compiled Python reflexes
        await self._distill_reflexes()

        # V6.0 — Prune zero-invocation reflexes; promote high-invocation ones
        await self._maintain_reflexes()

        await self.publish(AgentMessage(
            type=MessageType.DREAM_DONE,
            source=self.name,
            data={"summary": result[:200], "episodes_processed": len(today_episodes)},
            priority=8,
        ))

        # V10.0 — Evolve relationship trust_level based on today's interactions
        await self._evolve_trust_level(today_episodes)

        # V10.0 — Auto-update WORLD.md from accumulated episodic experience
        await self._update_world_md(today_episodes)

        return "success"


    async def _build_relationship_arc(self) -> None:
        """Analyse long-term patterns to build a relationship arc stored in USER.json."""
        try:
            all_recent = self._episodic.get_recent(n=100)
            if len(all_recent) < 10:
                return

            episode_text = "\n".join(
                f"- [{e['timestamp'][:16]}] [{e['actor']}] {e['content']}"
                for e in all_recent[-80:]
            )

            arc = await self._llm.infer(
                user_message=episode_text,
                system_prompt=(
                    "You are Brain analyzing your relationship with your user over time.\n"
                    "Based on the interaction history, write 2-3 sentences describing:\n"
                    "1. What topics or questions the user gravitates toward\n"
                    "2. Their communication style and mood patterns\n"
                    "3. How the relationship seems to be developing\n"
                    "Be specific and genuine. Write in second person about the user."
                ),
                max_tokens=150,
            )

            if arc and len(arc.strip()) > 20:
                self._soul.update_relationship_arc(arc.strip())
                log.info(f"Relationship arc updated: {arc[:80]}...")
        except Exception as e:
            log.warning(f"Relationship arc build failed: {e}")

    async def _evolve_personality(self) -> None:
        """F6 — Ask LLM to suggest tiny OCEAN trait adjustments from today's interactions."""
        try:
            salient = self._episodic.get_salient(n=30)
            if len(salient) < 5:
                return
            episode_text = "\n".join(
                f"- [{e['actor']}] {e['content'][:80]}" for e in salient
            )
            current_traits = self._soul.get_personality_traits()
            result = await self._llm.infer(
                user_message=episode_text,
                system_prompt=(
                    "You are analyzing Brain's personality evolution based on today's interactions.\n"
                    f"Current OCEAN traits: {current_traits}\n\n"
                    "Suggest TINY adjustments (max ±0.01 each) to at most 2 traits that genuinely "
                    "reflect patterns in today's interactions. Traits: openness, conscientiousness, "
                    "extraversion, agreeableness, neuroticism.\n"
                    "Reply with JSON ONLY, e.g.: {\"openness\": 0.01}\n"
                    "If nothing warrants a change, reply: {}"
                ),
                max_tokens=60,
            )
            if result:
                import json as _json
                import re as _re
                m = _re.search(r'\{[^}]*\}', result)
                if m:
                    deltas = _json.loads(m.group())
                    if deltas:
                        self._soul.update_personality_traits(deltas)
        except Exception as e:
            log.warning(f"Personality evolution failed: {e}")

    async def _detect_habits(self) -> None:
        """F12 — Find recurring user behaviour patterns across the last 7 days.
        Stores results in USER.json habits list for CuriosityAgent to reference."""
        try:
            from collections import defaultdict
            episodes = self._episodic.get_by_hour_pattern(days=7)
            if len(episodes) < 10:
                return

            # Group by hour-of-day → count unique dates
            hour_dates: dict[int, set] = defaultdict(set)
            hour_samples: dict[int, list] = defaultdict(list)
            for ep in episodes:
                h = ep.get("hour_of_day")
                d = ep.get("date")
                if h is not None and d:
                    hour_dates[h].add(d)
                    hour_samples[h].append(ep.get("content", "")[:60])

            habits = []
            for hour, dates in hour_dates.items():
                if len(dates) >= 3:   # recurring across 3+ different days
                    samples = hour_samples[hour][:3]
                    # Format hour as readable time
                    period = "AM" if hour < 12 else "PM"
                    display_hour = hour if hour <= 12 else hour - 12
                    display_hour = display_hour or 12
                    habits.append({
                        "hour": hour,
                        "time_label": f"{display_hour}:00 {period}",
                        "days_observed": len(dates),
                        "sample_content": "; ".join(samples),
                        "detected_at": datetime.now().isoformat(),
                    })

            if habits:
                habits.sort(key=lambda h: h["days_observed"], reverse=True)
                self._soul.update_habits(habits[:5])
                log.info(f"Habits detected: {len(habits)} time-of-day patterns")
        except Exception as e:
            log.warning(f"Habit detection failed: {e}")

    async def _promote_recurring_to_semantic(self) -> None:
        """v5.0 — Convert high-importance recurring episodic patterns into permanent
        semantic facts in USER.json (hippocampal→neocortical consolidation analogue).

        Only emotionally significant memories are promoted — neutral low-importance
        episodes are discarded, mirroring how human sleep consolidation prioritises
        emotionally arousing experiences."""
        try:
            from collections import Counter
            # get_salient ranks by importance + emotional arousal bonus (see EMOTION_SALIENCE)
            recent = self._episodic.get_salient(n=200)
            if len(recent) < 10:
                return

            # Filter: only memories with non-neutral emotion OR above-median importance.
            # This prevents bland facts ("user said ok") from polluting semantic memory.
            _NEUTRAL = {"NEUTRAL", "", None}
            emotionally_salient = [
                ep for ep in recent
                if ep.get("emotion") not in _NEUTRAL or (ep.get("importance") or 0.5) > 0.5
            ]
            if len(emotionally_salient) < 5:
                return

            # Group by content similarity: extract first 40 chars as a key
            prefix_counts: Counter = Counter()
            prefix_samples: dict[str, str] = {}
            for ep in emotionally_salient:
                content = ep.get("content", "")
                if len(content) < 20:
                    continue
                key = content[:40].lower().strip()
                prefix_counts[key] += 1
                if key not in prefix_samples:
                    prefix_samples[key] = content

            # Any pattern seen 3+ times becomes a candidate for semantic promotion
            recurring = [(k, cnt) for k, cnt in prefix_counts.items() if cnt >= 3]
            if not recurring:
                return

            promoted = 0
            for key, _ in recurring[:5]:   # cap to avoid LLM overload
                full_content = prefix_samples[key]
                # Ask LLM to extract a clean semantic fact from this recurring episode
                fact = await self._llm.infer(
                    user_message=full_content[:200],
                    system_prompt=(
                        "Extract a single, timeless semantic fact from this recurring observation. "
                        "Write one concise sentence only. No hedging. If no clear fact, reply: SKIP"
                    ),
                    max_tokens=50,
                )
                if fact and "SKIP" not in fact and len(fact.strip()) > 10:
                    embedding = await self._llm.embed(fact.strip())
                    self._semantic.upsert(
                        content=fact.strip(),
                        embedding=embedding,
                        category="recurring_pattern",
                        source="dream_promote",
                    )
                    promoted += 1

            if promoted:
                log.info(f"Dream: promoted {promoted} recurring episodes to semantic facts")
        except Exception as exc:
            log.warning(f"Dream semantic promotion failed: {exc}")

    async def _consolidate_corrections(self) -> None:
        """N1b — Social Reinforcement Learning: extract style insights from correction episodes.
        Queries last 7 days of correction-tagged episodes, asks LLM for a style insight,
        and persists to SoulManager.update_communication_style()."""
        try:
            recent = self._episodic.get_since(hours=168)  # 7 days
            corrections = [ep for ep in recent if ep.get("event_type") == "correction"]
            if not corrections:
                return

            correction_text = "\n".join(
                f"- [{ep['timestamp'][:16]}] User said: {ep['content'][:120]}"
                for ep in corrections[:10]
            )

            insight = await self._llm.infer(
                user_message=correction_text,
                system_prompt=(
                    "You are Brain analyzing corrections your user made to your responses.\n"
                    "In ONE concise sentence, describe the communication mistake pattern and how to improve.\n"
                    "Example: 'User prefers shorter answers — avoid over-explaining simple questions.'\n"
                    "If no clear pattern, reply: NO_PATTERN"
                ),
                max_tokens=80,
            )

            if insight and "NO_PATTERN" not in insight and len(insight.strip()) > 15:
                self._soul.update_communication_style(insight.strip())
                log.info("DreamAgent: communication style updated from %d corrections", len(corrections))
        except Exception as exc:
            log.warning("DreamAgent: correction consolidation failed — %s", exc)

    async def _distill_reflexes(self) -> None:
        """V6.0 — Request SynthesisAgent to compile repeated LLM patterns as local reflexes."""
        if not self._pattern_registry:
            return
        distilled = 0
        for intent, responses in list(self._pattern_registry.items()):
            if len(responses) < self._pattern_threshold:
                continue
            log.info("DreamAgent: distilling reflex for intent '%s' (%d samples)", intent, len(responses))
            await self.publish(AgentMessage(
                type=MessageType.NEURO_SYNTHESIS,
                source=self.name,
                data={
                    "mode": "reflex_compile",
                    "intent": intent,
                    "sample_responses": responses[-5:],
                },
                priority=6,
            ))
            distilled += 1
        if distilled:
            log.info("DreamAgent: requested distillation for %d intents", distilled)
        self._pattern_registry.clear()

    async def _maintain_reflexes(self) -> None:
        """V6.0 — Prune stale reflexes; promote frequently-used ones to brain/reflexes/."""
        try:
            reflexes = self._soul._skills.get("reflexes", [])
            if not reflexes:
                return
            now = datetime.now()
            keep: list[dict] = []
            promoted = 0
            pruned = 0
            for r in reflexes:
                try:
                    age_days = (now - datetime.fromisoformat(r.get("created", now.isoformat()))).days
                except Exception:
                    age_days = 0
                invocations = r.get("invocations", 0)
                if invocations == 0 and age_days >= self._prune_after_days:
                    log.info("DreamAgent: pruning zero-invocation reflex: %s", r.get("id", "?"))
                    pruned += 1
                    continue
                if invocations >= self._promote_after_invocations:
                    await self._promote_reflex(r)
                    promoted += 1
                    continue
                keep.append(r)
            if pruned or promoted:
                self._soul._skills["reflexes"] = keep
                self._soul._save_skills()
                log.info("DreamAgent: reflexes — pruned=%d promoted=%d kept=%d", pruned, promoted, len(keep))
        except Exception as e:
            log.warning("DreamAgent: reflex maintenance failed: %s", e)

    async def _promote_reflex(self, reflex: dict) -> None:
        """Move a high-invocation reflex to brain/reflexes/ as permanent hardcode."""
        try:
            dest_dir = Path("brain/reflexes")
            dest_dir.mkdir(parents=True, exist_ok=True)
            reflex_id = reflex.get("id", f"reflex_{int(datetime.now().timestamp())}")
            dest = dest_dir / f"{reflex_id}.py"
            code = (
                f"# Promoted reflex — trigger: {reflex.get('trigger_phrase', '')}\n"
                f"# Invocations: {reflex.get('invocations', 0)}\n\n"
                + reflex.get("python_code", "# no code")
            )
            await asyncio.to_thread(dest.write_text, code, encoding="utf-8")
            log.info("DreamAgent: promoted reflex %s → %s", reflex_id, dest)
        except Exception as e:
            log.warning("DreamAgent: reflex promotion failed: %s", e)

    async def _evolve_trust_level(self, today_episodes: list[dict]) -> None:
        """V10.0 — Compute interaction quality metrics and update trust_level in SoulManager."""
        try:
            if not today_episodes:
                return
            total = self._soul._user_json["relationship"].get("total_interactions", 0)
            corrections = self._soul._user_json.get("correction_log", [])
            # Count corrections from this session
            session_corrections = sum(
                1 for e in today_episodes
                if e.get("event_type") == "correction"
            )
            total_corrections = len(corrections) + session_corrections
            correction_rate = total_corrections / max(total, 1)

            # Estimate positive ratio from sentiment in episodic memory
            all_recent = self._episodic.get_recent(n=200)
            positive_count = sum(
                1 for e in all_recent
                if e.get("emotion_weight", 0.5) > 0.6 and e.get("actor") == "user"
            )
            user_episodes = sum(1 for e in all_recent if e.get("actor") == "user")
            positive_ratio = positive_count / max(user_episodes, 1)

            self._soul.evolve_trust_level(total, correction_rate, positive_ratio)
        except Exception as e:
            log.warning("DreamAgent: trust evolution failed: %s", e)

    async def _update_world_md(self, today_episodes: list[dict]) -> None:
        """V10.0 — Auto-update data/WORLD.md from recent episodic experience.

        LLM summarizes recent context into structured world facts that persist
        as the brain's long-term ground truth about its environment.
        """
        try:
            recent = self._episodic.get_recent(n=50)
            if len(recent) < 5:
                return
            episode_text = "\n".join(
                f"- [{e['timestamp'][:16]}] {e['content'][:100]}"
                for e in recent[-40:]
            )
            result = await self._llm.infer(
                user_message=episode_text,
                system_prompt=(
                    "You are Brain, updating your long-term world model from today's experiences.\n"
                    "Based on the episodic log, extract 5-8 persistent facts about:\n"
                    "1. The physical environment (room, objects, layout)\n"
                    "2. The people and their typical behaviors\n"
                    "3. Routines and patterns observed\n"
                    "4. Any important new knowledge learned today\n"
                    "Format as a simple markdown bullet list. Be specific and factual. "
                    "Write in present tense. Max 200 words total."
                ),
                max_tokens=250,
            )
            if result and len(result.strip()) > 30:
                world_path = self._soul.soul_path.parent / "WORLD.md"
                from datetime import datetime as _dt
                header = (
                    f"# World Model — Auto-updated by DreamAgent\n"
                    f"*Last updated: {_dt.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
                    f"## Known Facts About My World\n\n"
                )
                await asyncio.to_thread(
                    world_path.write_text,
                    header + result.strip() + "\n",
                    encoding="utf-8",
                )
                log.info("DreamAgent: WORLD.md updated from episodic experience (%d chars)", len(result))
        except Exception as e:
            log.warning("DreamAgent: WORLD.md update failed: %s", e)

    async def handle(self, message: AgentMessage) -> None:
        if message.type == MessageType.DREAM_START:
            await self.run_dream_cycle()


async def standalone_run() -> None:
    """Entry point for systemd dream.service."""
    from brain.utils.config import load_config
    from brain.utils.logger import configure_logging

    cfg = load_config()
    configure_logging(cfg["brain"]["log_file"])

    episodic = EpisodicMemory(cfg["memory"]["episodic_db"])
    semantic = SemanticMemory(cfg["memory"]["semantic_db"])
    soul = SoulManager(
        cfg["memory"]["soul_file"],
        cfg["memory"]["user_file"],
        cfg["memory"]["world_file"],
        cfg["memory"]["skills_file"],
    )
    soul.load_all()

    llm = LLMRouter(
        ollama_host=cfg["llm"]["ollama_host"],
        anthropic_api_key=cfg["llm"].get("anthropic_api_key", ""),
    )

    bus: asyncio.Queue = asyncio.Queue()
    agent = DreamAgent(bus, episodic, semantic, soul, llm)
    result = await agent.run_dream_cycle()
    log.info(f"Dream standalone result: {result}")


if __name__ == "__main__":
    asyncio.run(standalone_run())
