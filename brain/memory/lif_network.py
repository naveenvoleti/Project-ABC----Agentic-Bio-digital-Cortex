"""
Leaky Integrate-and-Fire (LIF) Network — Short-Term Cognition Array (Stage 1).
Provides biologically inspired habituation (satiation) for entity tracking.
"""
from __future__ import annotations

import time
from typing import Set, Tuple


class LIFNetwork:
    """
    Simulates a short-term cognition array using Leaky Integrate-and-Fire mechanics.
    - Satiation: Repeated exposure keeps voltage high, suppressing redundant triggers.
    - Interference: Time passing decays the voltage, allowing organic forgetting.
    """
    def __init__(
        self,
        decay_rate: float = 0.1,    # Voltage lost per second
        threshold: float = 1.0,     # Voltage required to be 'satiated'
        spike_boost: float = 2.0,   # Voltage gained per detection spike
        max_voltage: float = 5.0,   # Satiation limit
    ):
        self.neurons: dict[str, float] = {}
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.spike_boost = spike_boost
        self.max_voltage = max_voltage
        self.last_update = time.time()

    def step(self, active_entities: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Processes a timestep of the LIF array.
        
        Args:
            active_entities: Entities detected in the current sensory frame.
            
        Returns:
            (fired_entities, satiated_entities):
                - fired_entities: Entities that just crossed the threshold (newly satiated).
                - satiated_entities: All entities currently above threshold.
        """
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # Leaky phase: natural organic forgetting over time (Interference)
        expired = []
        for entity in self.neurons:
            self.neurons[entity] = max(0.0, self.neurons[entity] - (self.decay_rate * dt))
            if self.neurons[entity] <= 0.0:
                expired.append(entity)
                
        for entity in expired:
            del self.neurons[entity]

        fired_entities = set()
        satiated_entities = set()

        # Integrate and Fire phase (Satiation)
        for entity in active_entities:
            old_voltage = self.neurons.get(entity, 0.0)
            new_voltage = min(self.max_voltage, old_voltage + self.spike_boost)
            self.neurons[entity] = new_voltage

            # Trigger if it just crossed the threshold (was forgotten, now remembered)
            if old_voltage < self.threshold and new_voltage >= self.threshold:
                fired_entities.add(entity)

        # Collect all currently satiated entities
        for entity, voltage in self.neurons.items():
            if voltage >= self.threshold:
                satiated_entities.add(entity)

        return fired_entities, satiated_entities
