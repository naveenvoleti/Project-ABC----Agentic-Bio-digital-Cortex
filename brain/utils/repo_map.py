"""
Repo map — scans the Project-ABC structure and returns a compact software context
string injected into the system prompt as Stream 5 so the LLM knows exactly
which files to modify when asked to change code or UI.
"""
from __future__ import annotations
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # …/Project-ABC

# (relative_path, human description)
_KEY_FILES = [
    ("api/static/display.html",         "Web UI display page — title, layout, chat bubbles"),
    ("api/rest_api.py",                  "FastAPI REST endpoints"),
    ("config/config.yaml",              "System configuration"),
    ("brain/orchestrator.py",           "Master coordinator (25-agent message bus)"),
    ("brain/agents/cognition_agent.py", "Memory retrieval & LLM context assembly"),
    ("brain/agents/synthesis_agent.py", "Code synthesis & hot-reload"),
    ("brain/agents/planner_agent.py",   "Multi-step task planning"),
    ("brain/llm/llm_router.py",         "LLM routing (Ollama / Google / Claude)"),
    ("brain/memory/soul_manager.py",    "System prompt & identity builder"),
]

# Intent keywords that map to specific target files
_INTENT_FILE_MAP = {
    "ui":        "api/static/display.html",
    "display":   "api/static/display.html",
    "title":     "api/static/display.html",
    "layout":    "api/static/display.html",
    "config":    "config/config.yaml",
    "setting":   "config/config.yaml",
}


def build_repo_map() -> str:
    """Return a compact software context string for the system prompt."""
    lines = [
        f"## CODEBASE MANIFEST",
        f"Project root: {_PROJECT_ROOT.as_posix()}",
        "",
        "### File Registry",
        '{ "UI":     "api/static/display.html",',
        '  "Config": "config/config.yaml",',
        '  "Brain":  "brain/orchestrator.py",',
        '  "Soul":   "brain/memory/soul_manager.py" }',
        "",
        "### All Key Files",
    ]
    for rel_path, desc in _KEY_FILES:
        exists = "✓" if (_PROJECT_ROOT / rel_path).exists() else "✗"
        lines.append(f"- {exists} {rel_path} — {desc}")

    # Surface the first <title> or project name line from the UI file
    ui_file = _PROJECT_ROOT / "api/static/display.html"
    if ui_file.exists():
        try:
            for i, line in enumerate(ui_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                stripped = line.strip()
                if "<title>" in stripped.lower() or ("project-abc" in stripped.lower() and len(stripped) < 120):
                    lines.append(f"\n### UI Landmarks")
                    lines.append(f"- display.html line {i}: {stripped[:100]}")
                    break
        except Exception:
            pass

    lines += [
        "",
        "### Action Token Protocol",
        "- Modify web UI  → [EXECUTE: MODIFY_UI | PATH: api/static/display.html] [CONFIRM: spoken text]",
        "- Modify config  → [EXECUTE: MODIFY_CONFIG | PATH: config/config.yaml] [CONFIRM: spoken text]",
        "- The Orchestrator strips [EXECUTE:] before speech; only [CONFIRM:] text is spoken aloud.",
    ]
    return "\n".join(lines)


def resolve_target_file(text: str) -> str:
    """Map a natural-language intent string to a repository file path."""
    lower = text.lower()
    for keyword, path in _INTENT_FILE_MAP.items():
        if keyword in lower:
            return path
    return "unknown"
