from enum import Enum
import numpy as np

class GossipState(Enum):
    ACTIVE = "ACTIVE"
    DAMPENED = "DAMPENED"
    EXPIRED = "EXPIRED"

class SourceReason(Enum):
    REGULAR_PROPAGATION = "REGULAR_PROPAGATION"
    HIGH_ENTROPY_PROPAGATION = "HIGH_ENTROPY_PROPAGATION"
    STABILIZING_FEEDBACK = "STABILIZING_FEEDBACK"

class TransientGossipToken:
    def __init__(
        self,
        token_id: str,
        origin_node: int,
        semantic_embedding: np.ndarray,
        emotional_embedding: np.ndarray,
        state: GossipState = GossipState.ACTIVE,
        issuance_time: float = 0.0,
        dynamic_ttl: float = 10.0,
        source_reason: SourceReason = SourceReason.REGULAR_PROPAGATION,
        current_intensity: float = 1.0,
        **kwargs
    ):
        self.token_id = token_id
        self.origin_node = origin_node
        self.semantic_embedding = semantic_embedding
        self.emotional_embedding = emotional_embedding
        self.state = kwargs.get("GossipState", state)
        self.issuance_time = issuance_time
        self.dynamic_ttl = dynamic_ttl
        self.source_reason = source_reason
        self.current_intensity = current_intensity

    def decay_intensity(self, local_entropy: float, dt: float = 1.0):
        """
        Differential damping decay:
        dI/dt = -lambda * I - mu * H(B)
        """
        lambda_const = 0.08
        mu_const = 0.12

        decay_rate = (lambda_const * self.current_intensity) + (mu_const * local_entropy)
        self.current_intensity = max(0.0, self.current_intensity - (decay_rate * dt))

        # Faster TTL consumption if already dampened
        ttl_cost = dt * (2.0 if self.state == GossipState.DAMPENED else 1.0)
        self.dynamic_ttl -= ttl_cost

        # Update state boundaries
        if self.current_intensity <= 0.0 or self.dynamic_ttl <= 0:
            self.state = GossipState.EXPIRED
        elif self.current_intensity <= 0.6:
            self.state = GossipState.DAMPENED
        else:
            self.state = GossipState.ACTIVE