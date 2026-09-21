import numpy as np
from enum import Enum

class TransientGossipToken:
    def __init__(self, token_id, expiration_time):
        self.token_id = token_id
        self.origin_node : float = None  # Optional: can be set later
        self.semantic_embedding : np.ndarray = None  # Optional: can be set later
        self.emotional_embedding : np.ndarray = None  # Optional: can be set later
        self.state : GossipState = GossipState.ACTIVE
        self.issuance_time : float = None  # Optional: can be set later
        self.dynamic_ttl : float = None  # Time-to-live that can be adjusted based on network conditions

    def is_expired(self, current_time):
        return current_time > self.expiration_time

    def __repr__(self):
        return f"TransientGossipToken(token_id={self.token_id}, expiration_time={self.expiration_time})"

    

class GossipState(Enum):
    ACTIVE = 1
    DAMPENED = 2
    EXPIRED = 3