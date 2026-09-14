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