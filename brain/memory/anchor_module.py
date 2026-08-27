"""
Long-Term Anchor Module (Stage 2).
Locks core relational roles with zero decay.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.memory.soul_manager import SoulManager

class AnchorModule:
    """
    Permanent Synapses: Applies zero-decay anchor weights to map raw visual
    identities to structured core relational roles (e.g., Admin).
    """
    def __init__(self, soul: "SoulManager"):
        self._soul = soul

    def get_anchored_identity(self, raw_entity: str) -> str:
        """
        Maps a raw sensory name to a structural role.
        """
        if not raw_entity or raw_entity == "unknown_person":
            return raw_entity
            
        user_json = self._soul.load_user_json()
        primary_name = user_json.get("identity", {}).get("name", "")
        
        # If the detected face matches the system's primary user, lock it to the "Admin" role
        if primary_name and raw_entity.lower() == primary_name.lower():
            return f"{raw_entity} (Admin)"
            
        return raw_entity
