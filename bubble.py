import numpy as np

from processed_comment import ProcessedComment
from user import User
from TransientGossipToken import TransientGossipToken
from TransitionEdge import TransitionEdge

class BubbleNode:
    def __init__(self, node_id: int, topic_vector: np.ndarray, emotion_hist: np.ndarray,
                  topic_label: str = "", emotionAdaptationFactor: float = 1.0, topicAdaptationFactor: float = 1.0):

        self.id = node_id
        self.topic_label = topic_label
        self.v_topic = np.asarray(topic_vector, dtype=np.float32)
        self.v_emotion = np.asarray(emotion_hist, dtype=np.float32)
        self.emotionAdaptationFactor = emotionAdaptationFactor
        self.topicAdaptationFactor = topicAdaptationFactor

        # Ensure initial histogram validity
        tot = np.sum(self.v_emotion)
        if tot > 0:
            self.v_emotion /= tot

        self.neighbor_edges: dict[int, TransitionEdge] = {}
        self.active_participants: set = set()
        self.active_tokens: list[TransientGossipToken] = []

        self.last_sync_timestamp : float = None
        self.entropy : float = self.compute_entropy(self.v_emotion)  # Initial entropy calculation
        self.tresholdEntropy : float = 0.5  # Default threshold for entropy, can be adjusted based on application needs

    def add_new_comment(self, comment: ProcessedComment):
        """
        Updates the node's topic and emotion vectors based on a new comment.

        Parameters
        ----------
        comment : ProcessedComment
            The processed comment containing semantic and emotion embeddings.
        """
        # Update topic vector
        self.v_topic = (1 - self.topicAdaptationFactor) * (self.v_topic) + self.topicAdaptationFactor * comment.semantic_embedding

        # Update emotion histogram
        self.v_emotion = (1 - self.emotionAdaptationFactor) * (self.v_emotion) + self.emotionAdaptationFactor * comment.emotion_embedding
        self.entropy = self.compute_entropy(self.v_emotion)  # Update entropy after emotion histogram update
        
        # Add the comment to it's user
        user = comment.comment.author
        self.active_participants(user).add_comment(comment)

        # Add the comment's author to active participants
        self.active_participants.add(comment.comment.author)

    def add_new_user(self, user: User):
        """
        Adds a new user to the node's active participants.

        Parameters
        ----------
        user : User
            The user instance to be added.
        """
        if user in self.active_participants:
            raise ValueError(f"User '{user.username}' is already an active participant in this node.")

        self.active_participants.add(user)
        for comment in user.processed_comments:
            self.add_new_comment(comment)


    def compute_entropy(v_emotion: np.ndarray) -> float:
        # 1. Select strictly positive probabilities to avoid log2(0)
        p = v_emotion[v_emotion > 1e-12]
        
        # 2. Shannon Entropy calculation
        return float(-np.sum(p * np.log2(p)))

    def add_neighbor_edge(self, neighbor_node_id: int, transition_edge: TransitionEdge):
        """
        Adds a neighbor edge to the node's neighbor_edges dictionary.

        Parameters
        ----------
        neighbor_node_id : int
            The ID of the neighboring node.
        transition_edge : TransitionEdge
            The transition edge connecting this node to the neighboring node.
        """
        if neighbor_node_id in self.neighbor_edges:
            raise ValueError(f"Neighbor edge to node '{neighbor_node_id}' already exists.")
        
        self.neighbor_edges[neighbor_node_id] = transition_edge

    def propagate_token(self, token: TransientGossipToken):
        """
        Propagates a transient gossip token to all neighboring nodes.

        Parameters
        ----------
        token : TransientGossipToken
            The transient gossip token to be propagated.
        """
        for neighbor_id, edge in self.neighbor_edges.items():
            # Here you would implement the logic to send the token to the neighbor node.
            # This could involve network communication or other mechanisms depending on your architecture.
            print(f"Propagating token {token.token_id} to neighbor node {neighbor_id} via edge {edge}.")

    def receive_token(self, token: TransientGossipToken):
        """
        Receives a transient gossip token and adds it to the active tokens list.

        Parameters
        ----------
        token : TransientGossipToken
            The transient gossip token to be received.
        """
        if token not in self.active_tokens:
            self.active_tokens.append(token)