# LLM Prompt Reference — Project-ABC

Every user utterance triggers **two sequential LLM calls** inside `ReasoningAgent`.

---

## Call 1 — Think Pass (internal monologue, never spoken)

**File:** `brain/agents/reasoning_agent.py` → `handle()`

| Field | Value |
|-------|-------|
| `user_message` | Exact user text (e.g. `"hey what time is it"`) |
| `system_prompt` | `"You are thinking privately before responding. Do NOT write the response yet.\nIn 1-2 sentences answer: What is the user really asking? What tone and length should the response use?"` |
| `max_tokens` | `80` (config: `cognition_extensions.internal_monologue.think_tokens`) |
| `frame_b64` | Current camera frame as base64 JPEG string, or `""` if no camera |
| `context_messages` | `[]` — no conversation history |
| `skip_cache` | `True` — result is never cached (prevents thought from being returned for the main pass) |

**Result:** Published as `COGNITION_THOUGHT` → shown in UI think panel only. Never spoken.

---

## Call 2 — Main Response Pass

**File:** `brain/agents/reasoning_agent.py` → `handle()`

| Field | Value |
|-------|-------|
| `user_message` | Exact user text (same as think pass) |
| `system_prompt` | Stacked layers (see below) |
| `context_messages` | Last 6 conversation turns as `[{"role": "user"/"assistant", "content": "..."}]` |
| `max_tokens` | `512` normal / `200` throttled / capped by affective state: overwhelmed=`60`, hot/busy=`120` |
| `frame_b64` | Current camera frame as base64 JPEG string, or `""` if no camera |
| `skip_cache` | `False` — result is cached by `user_message` key for future similar queries |

### system_prompt Layer Stack (assembled in `CognitionAgent._handle_intent()`)

Layers are concatenated in this order. Each is optional and only appended when data is available.

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. GWT Spotlight prefix (prepended before soul prompt)          │
│    "## Current Attention Focus\n<topic>"                        │
│    "## Current Scene\n<scene description from VisionAgent>"     │
│    "## Surprise Signal\nSomething unexpected..."  (if surprise) │
│    "## Body State\n<stress label from InteroceptionAgent>"      │
├─────────────────────────────────────────────────────────────────┤
│ 2. Soul / Personality base (from SOUL.md via SoulManager)       │
│    Full identity, personality, current emotion, HW summary,     │
│    user profile (USER.md), known facts                          │
├─────────────────────────────────────────────────────────────────┤
│ 3. Mood Arc  (if ≥2 sentiment history entries)                  │
│    "## Conversation Mood Arc\n"                                 │
│    "User mood over last N turns: <positive|neutral|negative>"   │
│    "(<score>). Adjust your tone..."                             │
├─────────────────────────────────────────────────────────────────┤
│ 4. User Model  (if TheoryOfMindAgent has profiled user)         │
│    "## User Model\n"                                            │
│    "Expertise: <low|medium|high>. Explanation depth: <normal>." │
│    "Confusion score: <0.0-1.0>. Frustration score: <0.0-1.0>." │
├─────────────────────────────────────────────────────────────────┤
│ 5. Temporal Context  (if TemporalReasoningAgent has insight)    │
│    "## Temporal Context\n<pattern summary>"                     │
├─────────────────────────────────────────────────────────────────┤
│ 6. Creative Context  (consumed once from IdeationAgent)         │
│    "## Creative Context\n<pending idea>"                        │
├─────────────────────────────────────────────────────────────────┤
│ 7. Predicted Needs  (from scenario prefs + habit schedule)      │
│    "## Predicted Needs\nIt is <morning|afternoon|evening>.      │
│    user may want <preference> | usual habit: <activity>."       │
├─────────────────────────────────────────────────────────────────┤
│ 8. Brevity suffix  (only when emotion == FRUSTRATED)            │
│    "\nKeep response brief and direct."                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Routing Logic (inside `LLMRouter.infer()`)

**File:** `brain/llm/llm_router.py`

```
1. Cache lookup  (skip_cache=False only)
   → cosine similarity ≥ 0.92 on user_message embedding → return cached

2. ECO mode (thermal throttle active)
   → Ollama local  (simple_model, no image)

3. Normal routing by token complexity:
   a. simple  (≤150 tokens):  Ollama local  OR  Gemini ER cloud
   b. complex (>150 tokens):  Gemini ER cloud  (thinking_budget=1024)
   c. image attached:         Gemini ER cloud  (multimodal)

4. Fallback chain (if primary fails):
   Ollama Cloud (gemma4:31b-cloud)
   → OpenRouter (meta-llama/llama-3.2-3b-instruct:free)
   → Google AI REST (gemini-robotics-er-1.6-preview)
   → Ollama local balanced_model

5. Cache store  (skip_cache=False only)
```

---

## Debug Logging

Set `log_level: "DEBUG"` in `config/config.yaml` under `brain:` to see:

```
DEBUG | brain.llm.llm_router — LLMRouter.infer | user='...' | system='...' | ctx_turns=N | image=Xb | max_tokens=512
```

This fires **after** the cache check, so a cache hit will not produce this log line.
