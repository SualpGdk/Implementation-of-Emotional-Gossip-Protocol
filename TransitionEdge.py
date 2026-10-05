import numpy as np

class TransitionEdge:
    def __init__(self, source_node_id: int, target_node_id: int, weight: float, topicWeightingCoefficient: float = 1.0, emotionWeightingCoefficient: float = 1.0):
    
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.weight = weight
        self.affinity_score : float = 0.0
        self.emotional_delta : float = 0.0
        self.migration_count : int = 0
        self.topicWeightingCoefficient : float = topicWeightingCoefficient
        self.emotionWeightingCoefficient : float = emotionWeightingCoefficient
        self.historical_memory_factor : float = 0.9  # Default value, can be adjusted based on application needs

        self.source_node_last_advertised_entropy : float = 0.0
        self.target_node_last_advertised_entropy : float = 0.0

        # ADDED: (topic_vector, emotion_vector, entropy) as last advertised by each side
        self.source_snapshot = None
        self.target_snapshot = None

    def recalculate_affinity(self, source_topic_vector: np.ndarray, target_topic_vector: np.ndarray, source_emotion_vector: np.ndarray, target_emotion_vector: np.ndarray):
        """
        Recalculates the semantic affinity between the source and target nodes based on their topic vectors.

        Parameters
        ----------
        source_topic_vector : np.ndarray
            The topic vector of the source node.
        target_topic_vector : np.ndarray
            The topic vector of the target node.
        """
        
        self.affinity_score = np.exp(- self.topicWeightingCoefficient * np.linalg.norm(source_topic_vector - target_topic_vector)**2 - self.emotionWeightingCoefficient * np.linalg.norm(source_emotion_vector - target_emotion_vector)**2)

    def update_emotional_delta(self, source_emotion_vector: np.ndarray, target_emotion_vector: np.ndarray):
        """
        Updates the emotional delta between the source and target nodes based on their emotion vectors.

        Parameters
        ----------
        source_emotion_vector : np.ndarray
            The emotion vector of the source node.
        target_emotion_vector : np.ndarray
            The emotion vector of the target node.
        """
        # Calculate the emotional delta as the L1 norm (Manhattan distance) between the two emotion vectors
        self.emotional_delta = np.sum(np.abs(source_emotion_vector - target_emotion_vector))

    def update_weight(self, source_entropy : float, target_entropy : float):
        self.weight = self.historical_memory_factor * self.weight + (1 - self.historical_memory_factor) * (self.affinity_score * (1 - abs(source_entropy - target_entropy) / max(source_entropy, target_entropy, 1e-6)))

    def post_entropy(self, node_id : str, entropy : float):
        if node_id == self.source_node_id:
            self.source_node_last_advertised_entropy = entropy
        elif node_id == self.target_node_id:
            self.target_node_last_advertised_entropy = entropy
        else:
            raise ValueError(f"Node ID {node_id} does not match either source or target node IDs.")

    def get_opposing_side_entropy(self, node_id : str) -> float:
        if node_id == self.source_node_id:
            return self.target_node_last_advertised_entropy
        elif node_id == self.target_node_id:
            return self.source_node_last_advertised_entropy
        else:
            raise ValueError(f"Node ID {node_id} does not match either source or target node IDs.")

    # ------------------------------------------------------------------
    # ADDED: everything below is new (snapshot sharing, refresh, migration traffic)
    # ------------------------------------------------------------------
    def post_snapshot(self, node_id: int, topic: np.ndarray, emotion: np.ndarray, entropy: float):
        """Advertise this node's (v_topic, v_emotion, entropy) to the other side. Supersedes post_entropy."""
        snap = (np.array(topic, dtype=np.float32), np.array(emotion, dtype=np.float32), float(entropy))
        if node_id == self.source_node_id:
            self.source_snapshot = snap
            self.source_node_last_advertised_entropy = float(entropy)
        elif node_id == self.target_node_id:
            self.target_snapshot = snap
            self.target_node_last_advertised_entropy = float(entropy)
        else:
            raise ValueError(f"Node ID {node_id} does not match either source or target node IDs.")

    def get_opposing_snapshot(self, node_id: int):
        """Snapshot of the side that is NOT node_id (pass the caller's own id), or None if not posted yet."""
        if node_id == self.source_node_id:
            return self.target_snapshot
        elif node_id == self.target_node_id:
            return self.source_snapshot
        raise ValueError(f"Node ID {node_id} does not match either source or target node IDs.")

    def refresh(self):
        """Recompute A_ij, emotional delta and W_ij from both posted snapshots. Call once per step per edge."""
        if self.source_snapshot is None or self.target_snapshot is None:
            return
        s_topic, s_emo, s_h = self.source_snapshot
        t_topic, t_emo, t_h = self.target_snapshot
        self.recalculate_affinity(s_topic, t_topic, s_emo, t_emo)
        self.update_emotional_delta(s_emo, t_emo)
        self.update_weight(s_h, t_h)

    def record_migration(self, boost: float = 0.02):
        """Historical traffic: every migration makes this path slightly more fluid."""
        self.migration_count += 1
        self.weight = min(1.0, self.weight + boost)
