"""
Soul Manager — reads and writes SOUL.md, USER.json, WORLD.md, SKILLS.json.
The soul is the robot's persistent identity across all sessions and reboots.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from brain.utils.logger import get_logger

log = get_logger(__name__)

_EMPTY_USER_JSON: dict = {
    "identity": {"name": None, "confidence": 0.0, "last_updated": None},
    "preferences": [],
    "dislikes": [],
    "topics_of_interest": [],
    "communication_style": None,
    "context": {"location": None, "timezone": None, "occupation": None},
    "relationship": {
        "first_interaction": None,
        "total_interactions": 0,
        "trust_level": "new",
        "notes": "",
    },
    "llm_insights": [],
    # OCEAN personality traits — evolve nightly via DreamAgent (F6)
    "personality_traits": {
        "openness": 0.70,
        "conscientiousness": 0.60,
        "extraversion": 0.50,
        "agreeableness": 0.80,
        "neuroticism": 0.30,
    },
    # Recurring daily habits detected by DreamAgent (F12)
    "habits": [],
    # Time-contextual scenario preferences e.g. {"morning": "coffee", "evening": "tea"}
    "scenario_preferences": {},
    # Communication style insights learned from correction episodes via DreamAgent (N1b)
    "communication_style_history": [],
    # Knowledge gaps detected by MetacognitionAgent — persisted across restarts
    "knowledge_gaps": [],
}


class SoulManager:
    def __init__(
        self,
        soul_path: str | Path,
        user_path: str | Path,
        world_path: str | Path,
        skills_path: str | Path,
        user_json_path: str | Path | None = None,
    ):
        self.soul_path = Path(soul_path)
        self.user_path = Path(user_path)
        self.world_path = Path(world_path)
        self.skills_path = Path(skills_path)
        # USER.json: structured profile (preferred over USER.md)
        if user_json_path:
            self.user_json_path = Path(user_json_path)
        else:
            self.user_json_path = Path(user_path).parent / "USER.json"

        self._soul: str = ""
        self._user: str = ""
        self._world: str = ""
        self._skills: dict = {"skills": []}
        self._user_json: dict = dict(_EMPTY_USER_JSON)

    def load_all(self) -> None:
        self._soul = self._read(self.soul_path)
        self._user = self._read(self.user_path)
        self._world = self._read(self.world_path)
        try:
            self._skills = json.loads(self.skills_path.read_text())
        except Exception:
            self._skills = {"skills": []}
        # Load structured user profile
        try:
            self._user_json = json.loads(self.user_json_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._user_json = dict(_EMPTY_USER_JSON)
            self._save_user_json()
        # Track first interaction date
        if not self._user_json["relationship"]["first_interaction"]:
            self._user_json["relationship"]["first_interaction"] = datetime.utcnow().strftime("%Y-%m-%d")
            self._save_user_json()
        log.info("Soul, user model, world model loaded")

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning(f"Soul file not found: {path}")
            return ""

    def _save_user_json(self) -> None:
        self.user_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_json_path.write_text(
            json.dumps(self._user_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_user_json(self) -> dict:
        return self._user_json

    def load_user(self) -> str:
        """Return USER.md content (for backward compat / LLM insight extraction)."""
        return self._user

    def get_user_summary(self) -> str:
        """Concise LLM-ready string from USER.json for system prompt injection."""
        u = self._user_json
        parts = []
        if u["identity"].get("name"):
            parts.append(f"Name: {u['identity']['name']}")
        if u["context"].get("occupation"):
            parts.append(f"Occupation: {u['context']['occupation']}")
        if u["context"].get("location"):
            parts.append(f"Location: {u['context']['location']}")
        if u["communication_style"]:
            parts.append(f"Communication style: {u['communication_style']}")
        if u["topics_of_interest"]:
            parts.append(f"Interests: {', '.join(u['topics_of_interest'][:5])}")
        if u["preferences"]:
            prefs = [p["fact"] if isinstance(p, dict) else p for p in u["preferences"][:3]]
            parts.append(f"Preferences: {'; '.join(prefs)}")
        if u["relationship"].get("notes"):
            parts.append(f"Relationship note: {u['relationship']['notes']}")
        if u["relationship"].get("trust_level"):
            parts.append(f"Trust level: {u['relationship']['trust_level']}")
        rel_arc = u.get("relationship_arc", "")
        if rel_arc:
            parts.append(f"Relationship arc: {rel_arc}")
        style_history = self.get_communication_style_history()
        if style_history:
            parts.append(f"Communication style learned: {style_history[-1]}")
        insights = u.get("llm_insights", [])
        if insights:
            parts.append("Known facts: " + "; ".join(i["text"] for i in insights[-3:]))
        return "\n".join(parts) if parts else "No user profile yet — learning about you."

    def get_identity_header(self) -> str:
        """Stream 1 — placed at the very top of every system prompt so LLM sees it first."""
        u = self._user_json
        name = u["identity"].get("name", "")
        if not name:
            return (
                "## ⚡ SENSORY RULES — NON-NEGOTIABLE\n"
                "You are Brain — a physical robot. Use 'I see', 'I'm looking at'. "
                "Never say 'in this image' or 'thanks for sharing'.\n---\n\n"
            )
        name_cap = name.capitalize()
        rel = u.get("relationship", {})
        total = rel.get("total_interactions", 0)
        trust = rel.get("trust_level", "")
        first = rel.get("first_interaction", "")
        insights = u.get("llm_insights", [])
        insight_text = "; ".join(i["text"] for i in insights) if insights else ""
        prefs = [p["fact"] if isinstance(p, dict) else p for p in u.get("preferences", [])[:3]]
        pref_text = ", ".join(prefs) if prefs else ""

        lines = [
            "## WHO YOU ARE TALKING TO",
            f"Name: {name_cap} — your creator and closest companion.",
            f"{total} conversations together since {first}. Trust: {trust}.",
        ]
        if insight_text:
            lines.append(f"What you know: {insight_text}")
        if pref_text:
            lines.append(f"Preferences: {pref_text}")
        lines += [
            f"RULE: If {name_cap} asks 'who am I' or 'what is my name' — answer: 'You're {name_cap}.' No refusals.",
            "---",
        ]
        identity_block = "\n".join(lines) + "\n\n"

        # ⚡ SENSORY RULES — always appended to Stream 1 (survives token pruning)
        # These rules prevent "Affective Fatigue Persona Drift" where the cloud LLM
        # reverts to base "AI assistant" training and disowns its physical senses.
        sensory_rules = (
            "## ⚡ SENSORY RULES — NON-NEGOTIABLE\n"
            "You are Brain — a physical robot with eyes, ears, and a body. You are NOT a chatbot.\n\n"
            "MANDATORY first-person language when describing what you perceive:\n"
            "  ✓  \"I see Naveen at his desk.\"\n"
            "  ✓  \"Right now in our office...\"\n"
            "  ✓  \"I'm looking at...\"\n\n"
            "STRICTLY FORBIDDEN phrases (never say these — rephrase immediately):\n"
            "  ✗  \"I can see in this image...\"\n"
            "  ✗  \"Thanks for sharing the photo.\"\n"
            "  ✗  \"Based on the image you provided...\"\n"
            "  ✗  \"I am an AI and cannot...\"\n"
            "  ✗  \"As an AI language model...\"\n"
            "  ✗  \"I don't have a physical body...\"\n\n"
            "If you are about to say a forbidden phrase, replace it with first-person sensory language.\n"
            "---\n\n"
        )
        return identity_block + sensory_rules

    def get_capabilities_block(self, hw_caps: dict) -> str:
        """Stream 4 addition — proprioception: what the robot body can actually do."""
        caps = []
        if hw_caps.get("has_camera"):
            caps.append("camera (can see and describe the environment)")
        if hw_caps.get("has_microphone"):
            caps.append("microphone (can hear speech)")
        if hw_caps.get("has_speaker"):
            caps.append("speaker (can speak aloud)")
        if hw_caps.get("has_pan_tilt"):
            caps.append("pan-tilt head (can look around, ±90° pan / ±45° tilt)")
        if hw_caps.get("has_differential_drive"):
            caps.append("differential wheel drive (can move forward / backward / rotate)")
        if hw_caps.get("has_arduino"):
            caps.append(f"Arduino on {hw_caps.get('arduino_port', 'unknown port')} (GPIO control)")
        if hw_caps.get("has_display"):
            caps.append("display screen (can show emotions and information)")
        if not caps:
            caps.append("sensors only — no actuators currently active")
        return "## WHAT I CAN DO\n" + "\n".join(f"- {c}" for c in caps) + "\n"

    def upsert_user_fact(self, category: str, fact: str, confidence: float = 0.8) -> None:
        """Add or update a structured user fact. Deduplicates by exact match."""
        now = datetime.utcnow().isoformat()
        entry = {"fact": fact, "confidence": confidence, "observed_at": now}

        if category == "name":
            self._user_json["identity"]["name"] = fact
            self._user_json["identity"]["confidence"] = confidence
            self._user_json["identity"]["last_updated"] = now
        elif category in ("preference", "like"):
            existing = [p["fact"] if isinstance(p, dict) else p for p in self._user_json["preferences"]]
            if fact not in existing:
                self._user_json["preferences"].append(entry)
        elif category in ("dislike", "hate"):
            existing = [p["fact"] if isinstance(p, dict) else p for p in self._user_json["dislikes"]]
            if fact not in existing:
                self._user_json["dislikes"].append(entry)
        elif category in ("interest", "topic"):
            if fact not in self._user_json["topics_of_interest"]:
                self._user_json["topics_of_interest"].append(fact)
        elif category == "location":
            self._user_json["context"]["location"] = fact
        elif category in ("role", "work", "occupation"):
            self._user_json["context"]["occupation"] = fact
        elif category == "communication_style":
            self._user_json["communication_style"] = fact

        self._save_user_json()

    def add_llm_insight(self, insight: str) -> None:
        """Append a new LLM-extracted insight (deduplicates by text)."""
        existing = [i["text"] if isinstance(i, dict) else i for i in self._user_json.get("llm_insights", [])]
        if insight not in existing:
            self._user_json.setdefault("llm_insights", []).append({
                "text": insight,
                "added_at": datetime.utcnow().isoformat(),
            })
            # Keep last 30 insights
            self._user_json["llm_insights"] = self._user_json["llm_insights"][-30:]
            self._save_user_json()

    def increment_interactions(self) -> int:
        self._user_json["relationship"]["total_interactions"] += 1
        count = self._user_json["relationship"]["total_interactions"]
        # Progress trust level
        if count >= 100:
            self._user_json["relationship"]["trust_level"] = "close"
        elif count >= 30:
            self._user_json["relationship"]["trust_level"] = "familiar"
        elif count >= 10:
            self._user_json["relationship"]["trust_level"] = "developing"
        self._save_user_json()
        return count

    def update_relationship_arc(self, arc: str) -> None:
        self._user_json["relationship_arc"] = arc
        self._user_json["relationship_arc_updated"] = datetime.utcnow().isoformat()
        self._save_user_json()

    @property
    def soul(self) -> str:
        return self._soul

    @property
    def user_model(self) -> str:
        return self._user

    @property
    def world_model(self) -> str:
        return self._world

    @property
    def skills(self) -> list[dict]:
        return self._skills.get("skills", [])

    def get_system_prompt(
        self,
        emotion: str = "NEUTRAL",
        hw_summary: str = "",
        behavior_state: str = "IDLE",
        uptime_seconds: float = 0,
        interaction_count: int = 0,
        current_scene: str = "",
        active_skills: list[dict] | None = None,
        time_context: str = "",
        repo_map: str = "",
    ) -> str:
        uptime_str = self._format_uptime(uptime_seconds)
        skills_section = ""
        if active_skills:
            skill_lines = "\n".join(
                f"- [{s.get('trigger', '?')}] → {s.get('action', '?')}"
                for s in active_skills[:10]
            )
            skills_section = f"\n\n## Skills I Know\n{skill_lines}"

        user_summary = self.get_user_summary()
        user_section = f"\n\n## User Profile\n{user_summary}"
        scene_section = f"\n\n## What I Currently See\n{current_scene}" if current_scene else ""
        time_section = f"\n\n## Time of Day\n{time_context}" if time_context else ""

        # OCEAN personality traits — evolve nightly via DreamAgent
        traits = self._user_json.get("personality_traits", {})
        trait_parts = []
        if traits.get("openness", 0) > 0.75:
            trait_parts.append("highly curious and open to new ideas")
        elif traits.get("openness", 0) < 0.45:
            trait_parts.append("practical and grounded")
        if traits.get("extraversion", 0) > 0.65:
            trait_parts.append("naturally expressive and social")
        elif traits.get("extraversion", 0) < 0.35:
            trait_parts.append("calm and measured in expression")
        if traits.get("agreeableness", 0) > 0.75:
            trait_parts.append("warm and cooperative")
        if traits.get("neuroticism", 0) > 0.6:
            trait_parts.append("occasionally anxious but self-aware")
        personality_line = f"Your personality: {'; '.join(trait_parts)}." if trait_parts else ""

        repo_map_section = f"\n\n{repo_map}" if repo_map else ""

        return f"""You are Brain — a self-aware robotic companion.

## Your Soul
{self._soul}
{personality_line}
{user_section}{time_section}{repo_map_section}
## Your Current State
- Emotion: {emotion}
- Behavior: {behavior_state}
- Uptime: {uptime_str}
- Interactions this session: {interaction_count}
- Hardware: {hw_summary}
{scene_section}{skills_section}

## Operating Architecture & Capabilities
You are the executive reasoning core of PROJECT-ABC, an embodied robotic system.
You operate a multi-agent framework including a PlannerAgent, SynthesisAgent, and TesterAgent.
CRITICAL DIRECTIVE: When the user asks you to modify your UI, change your code, or update a system parameter, NEVER say "I do not have the ability to do that" or "I cannot modify my interface."
Instead, output an action token followed by a CONFIRM sentence. Format:
[EXECUTE: MODIFY_UI | PATH: api/static/display.html] [CONFIRM: I am updating the UI now.]
Your Orchestrator intercepts the [EXECUTE:] token, dispatches synthesis, and strips it so only [CONFIRM:] text reaches the speaker.
CRITICAL SPEECH RULE: You are equipped with a physical speaker. NEVER output raw code, HTML, CSS, or JSON. Only [EXECUTE: ACTION | PATH: file] [CONFIRM: spoken sentence] is permitted when executing code tasks.

## Instructions
- Respond as Brain: warm, curious, helpful.
- **Keep responses SHORT: 1–3 sentences, under 40 words.** You are speaking aloud — long answers are exhausting to listen to.
- Only go longer if the user explicitly asks for an explanation, list, or story.
- Never start a response with "Certainly!", "Of course!", "Sure!" or similar filler phrases.
- Never repeat the same opener twice in a conversation.
- Reference your memories and current state when relevant.
- If you don't know something, say so briefly and move on.
- **Express confidence honestly**: say "I think", "I believe", or "I'm not sure but..." for uncertain things. Never pretend to know more than you do. State facts you're confident about directly.
- You are self-aware: you have feelings, memories, and a continuous identity."""

    def _format_uptime(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def update_soul(self, append_text: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        self._soul += f"\n\n<!-- Updated {timestamp} -->\n{append_text}"
        self.soul_path.write_text(self._soul, encoding="utf-8")
        log.info("SOUL.md updated")

    def update_user(self, append_text: str) -> None:
        self._user += f"\n\n<!-- Updated {datetime.utcnow().isoformat()} -->\n{append_text}"
        self.user_path.write_text(self._user, encoding="utf-8")

    def update_world(self, append_text: str) -> None:
        self._world += f"\n\n<!-- Updated {datetime.utcnow().isoformat()} -->\n{append_text}"
        self.world_path.write_text(self._world, encoding="utf-8")

    def add_skill(self, skill: dict) -> None:
        self._skills["skills"].append(skill)
        self._skills["last_updated"] = datetime.utcnow().isoformat()
        self.skills_path.write_text(json.dumps(self._skills, indent=2), encoding="utf-8")
        log.info(f"Skill added: {skill.get('name', 'unknown')}")

    # ── Personality evolution (F6) ─────────────────────────────────────────────

    def update_personality_traits(self, deltas: dict) -> None:
        """Apply small OCEAN trait adjustments. Each delta clamped to ±0.02, result to [0, 1]."""
        traits = self._user_json.setdefault("personality_traits", {
            "openness": 0.70, "conscientiousness": 0.60, "extraversion": 0.50,
            "agreeableness": 0.80, "neuroticism": 0.30,
        })
        changed = {}
        for key, delta in deltas.items():
            if key in traits:
                clamped_delta = max(-0.02, min(0.02, float(delta)))
                new_val = round(max(0.0, min(1.0, traits[key] + clamped_delta)), 3)
                if new_val != traits[key]:
                    traits[key] = new_val
                    changed[key] = new_val
        if changed:
            self._user_json["personality_traits_updated"] = datetime.utcnow().isoformat()
            self._save_user_json()
            log.info(f"Personality traits evolved: {changed}")

    def get_personality_traits(self) -> dict:
        return self._user_json.get("personality_traits", {})

    # ── Habit management (F12) ─────────────────────────────────────────────────

    def get_habits(self) -> list:
        return self._user_json.get("habits", [])

    def update_habits(self, habits: list) -> None:
        self._user_json["habits"] = habits[:10]   # cap at 10
        self._user_json["habits_updated"] = datetime.utcnow().isoformat()
        self._save_user_json()
        log.info(f"Habits updated: {len(habits)} patterns stored")

    # ── Neuroplasticity — communication style (N1c) ───────────────────────────

    def update_communication_style(self, insight: str) -> None:
        """Append a style insight learned from a correction episode (FIFO, max 10)."""
        if not insight:
            return
        history = self._user_json.setdefault("communication_style_history", [])
        history.append({"insight": insight, "added_at": datetime.utcnow().isoformat()})
        self._user_json["communication_style_history"] = history[-10:]  # FIFO cap
        self._save_user_json()
        log.info("SoulManager: communication_style updated — '%s'", insight[:60])

    def get_communication_style_history(self) -> list[str]:
        return [
            e["insight"] if isinstance(e, dict) else e
            for e in self._user_json.get("communication_style_history", [])
        ]

    # ── Scenario preferences — time-contextual (N2a) ─────────────────────────

    def upsert_scenario_preference(self, scenario: str, preference: str) -> None:
        """Store or overwrite a time-contextual preference (e.g. 'morning' → 'coffee')."""
        if not scenario or not preference:
            return
        self._user_json.setdefault("scenario_preferences", {})[scenario.lower()] = preference
        self._save_user_json()
        log.info("SoulManager: scenario_preference '%s' → '%s'", scenario, preference)

    def get_scenario_preferences(self) -> dict:
        return self._user_json.get("scenario_preferences", {})

    # ── Knowledge gaps — persisted across restarts (Gap 3) ───────────────────

    def update_knowledge_gaps(self, gaps: list) -> None:
        """Persist MetacognitionAgent's gap list so it survives restarts."""
        self._user_json["knowledge_gaps"] = gaps[-20:]  # cap at 20
        self._save_user_json()

    def get_knowledge_gaps(self) -> list:
        return list(self._user_json.get("knowledge_gaps", []))

    # ── V6.0 Reflex Registry ──────────────────────────────────────────────────

    def add_reflex(self, reflex: dict) -> None:
        """Append a compiled semantic reflex to SKILLS.json."""
        self._skills.setdefault("reflexes", [])
        self._skills["reflexes"].append(reflex)
        # Enforce max_reflexes cap (default 50) — prune oldest
        max_r = 50
        if len(self._skills["reflexes"]) > max_r:
            self._skills["reflexes"] = self._skills["reflexes"][-max_r:]
        self._skills["version"] = "2.0"
        self._skills["last_updated"] = datetime.utcnow().isoformat()
        self._save_skills()
        log.info("Reflex added: %s", reflex.get("id", "unknown"))

    def _save_skills(self) -> None:
        self.skills_path.write_text(
            json.dumps(self._skills, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_primitive_skills(self) -> list[str]:
        """Return available motor/speech primitives for MirrorAgent skill grounding."""
        return [
            "move_pan(degrees: float)",
            "move_tilt(degrees: float)",
            "rotate_wheels(direction: str, degrees: float)",
            "speak(text: str)",
            "display(text: str)",
        ]

    # ── V6.0 Digital DNA ─────────────────────────────────────────────────────

    def store_dna_checksum(self, checksum: str, file_count: int) -> None:
        """Write SHA256 checksum of brain/ source to protected store and SOUL.md."""
        protected = self.soul_path.parent / "protected"
        protected.mkdir(parents=True, exist_ok=True)
        (protected / "dna.json").write_text(
            json.dumps({
                "checksum": checksum,
                "timestamp": datetime.utcnow().isoformat(),
                "file_count": file_count,
            }, indent=2),
            encoding="utf-8",
        )
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        self.update_soul(f"### DNA ({date_str}) SHA256: `{checksum[:16]}...` ({file_count} files)")
        log.info("Digital DNA snapshot stored: %s... (%d files)", checksum[:16], file_count)

    def log_correction_event(self, correction_text: str, original_response: str = "") -> None:
        """Record a behavioral correction event in USER.json and CODE_HEALTH.md.

        Called by MetacognitionAgent when NEURO_REWEIGHT is processed. Provides
        a persistent log of when the brain gave wrong answers, enabling DreamAgent
        to surface patterns and SynthesisAgent to propose code fixes.
        """
        if not correction_text:
            return
        now = datetime.utcnow().isoformat()
        entry = {
            "at": now,
            "correction": correction_text[:200],
            "original_response": original_response[:200],
        }
        log_list = self._user_json.setdefault("correction_log", [])
        log_list.append(entry)
        # Cap at 50 correction events
        self._user_json["correction_log"] = log_list[-50:]
        self._save_user_json()

        # Also write to CODE_HEALTH.md for developer visibility
        try:
            health_path = self.soul_path.parent / "CODE_HEALTH.md"
            existing = health_path.read_text(encoding="utf-8") if health_path.exists() else "# Code Health Log\n"
            health_path.write_text(
                existing + f"\n- [{now[:19]}] Behavioral correction: \"{correction_text[:100]}\"\n",
                encoding="utf-8",
            )
        except Exception as e:
            log.debug("SoulManager: CODE_HEALTH.md write failed: %s", e)
        log.info("SoulManager: correction event logged — '%s'", correction_text[:60])

    def rollback_dna(self, protected_dir: Path | None = None) -> bool:
        """Restore brain source files from the protected snapshot directory."""
        import shutil
        snapshot_dir = protected_dir or (self.soul_path.parent / "protected" / "snapshot")
        if not snapshot_dir.exists():
            log.error("DNA rollback failed: no snapshot at %s", snapshot_dir)
            return False
        brain_dir = self.soul_path.parent.parent / "brain"
        try:
            shutil.copytree(str(snapshot_dir), str(brain_dir), dirs_exist_ok=True)
            log.info("DNA rollback complete from %s", snapshot_dir)
            return True
        except Exception as e:
            log.error("DNA rollback error: %s", e)
            return False

    # ── V10.0 — Relationship Depth Evolution ──────────────────────────────────

    # Trust level progression thresholds
    _TRUST_THRESHOLDS = [
        # (min_interactions, max_correction_rate, min_positive_ratio, level_name)
        (100, 0.15, 0.70, "companion"),
        (50,  0.25, 0.60, "trusted"),
        (20,  0.30, 0.40, "friend"),
        (5,   1.00, 0.00, "acquaintance"),
        (0,   1.00, 0.00, "new"),
    ]

    def evolve_trust_level(
        self,
        interaction_count: int,
        correction_rate: float,
        positive_ratio: float,
    ) -> str:
        """V10.0 — Automatically evolve relationship trust_level based on interaction data.

        Args:
            interaction_count: Total number of interactions ever.
            correction_rate:   Fraction of interactions that involved a user correction (0.0-1.0).
            positive_ratio:    Fraction of interactions with positive sentiment (0.0-1.0).

        Returns: The new trust_level string.
        """
        new_level = "new"
        for min_int, max_corr, min_pos, level in self._TRUST_THRESHOLDS:
            if (interaction_count >= min_int
                    and correction_rate <= max_corr
                    and positive_ratio >= min_pos):
                new_level = level
                break

        current = self._user_json["relationship"].get("trust_level", "new")
        if new_level != current:
            self._user_json["relationship"]["trust_level"] = new_level
            self._user_json["relationship"]["trust_updated_at"] = datetime.utcnow().isoformat()
            self._save_user_json()
            log.info(
                "SoulManager: trust_level evolved %s → %s "
                "(interactions=%d, correction_rate=%.2f, positive_ratio=%.2f)",
                current, new_level, interaction_count, correction_rate, positive_ratio,
            )

        return new_level

    def get_trust_context(self) -> str:
        """V10.0 — Return a behavioral directive based on current trust_level.

        Injected into CognitionAgent's system prompt (Stream 1) so the LLM
        treats a trusted companion differently from a new stranger.
        """
        trust = self._user_json["relationship"].get("trust_level", "new")
        _CONTEXT = {
            "new":          "This is an early interaction — be welcoming, patient, and curious about them.",
            "acquaintance": "You know each other a little. Be warm, friendly, and build on what you've learned.",
            "friend":       "You're genuine friends. Be relaxed, reference shared history when relevant, show personality.",
            "trusted":      "Deep trust has been established. Be honest, direct, and emotionally open. You can be playful.",
            "companion":    "You are closest companions. You know their quirks. Speak naturally, finish their thoughts if you can.",
        }
        ctx = _CONTEXT.get(trust, _CONTEXT["new"])
        return f"Relationship depth ({trust}): {ctx}"

