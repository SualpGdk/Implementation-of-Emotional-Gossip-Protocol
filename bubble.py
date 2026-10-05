from enum import Enum
import random
import numpy as np

from processed_comment import ProcessedComment
from user import User
from TransientGossipToken import GossipState, SourceReason, TransientGossipToken
from TransitionEdge import TransitionEdge
from migration import UserMigrationMixin, MigrationConfig  # ADDED: user migration


class BubbleState(Enum):
    HOT = 1
    NORMAL = 2


class BubbleNode(UserMigrationMixin):  # CHANGED: inherits the migration logic
    def __init__(self, node_id: int, topic_vector: np.ndarray, emotion_hist: np.ndarray,
                 topic_label: str = "",
                 emotionAdaptationFactor: float = 0.1,  # CHANGED (point 5): was 1.0, which meant no EMA at all
                 topicAdaptationFactor: float = 0.1,    # CHANGED (point 5): was 1.0
                 entropyThresholdFraction: float = 0.5):  # ADDED (point 3): threshold as a fraction of max entropy

        self.Node_State = BubbleState.NORMAL

        self.id = node_id
        self.topic_label = topic_label
        self.v_topic = np.asarray(topic_vector, dtype=np.float32)
        self.v_emotion = np.asarray(emotion_hist, dtype=np.float32)
        self.emotionAdaptationFactor = emotionAdaptationFactor
        self.topicAdaptationFactor = topicAdaptationFactor

        self.default_propagation_chance: float = 0.05
        self.max_propagation_chance: float = 0.35
        self.active_propagation_chance: float = self.default_propagation_chance  
        self.propagation_cooldown: int = 0
        self.base_cooldown: int = 5

        self.outbound_tokens_buffer: dict[int, list[TransientGossipToken]] = {}

        # Ensure initial histogram validity
        tot = np.sum(self.v_emotion)
        if tot > 0:
            self.v_emotion /= tot

        self.neighbor_edges: dict[int, TransitionEdge] = {}
        self.active_participants: dict[str, User] = {}
        self.active_tokens: list[TransientGossipToken] = []

        self.last_sync_timestamp: float = None
        self.entropy: float = self.compute_entropy(self.v_emotion)
        # CHANGED (point 3): was a flat 0.5 bits, which is below the entropy of most m-class histograms
        # (max is log2(m)), so nodes were almost always HOT. Now: fraction of the maximum entropy.
        self.tresholdEntropy: float = entropyThresholdFraction * float(np.log2(len(self.v_emotion)))

        self.token_counter: int = 0

        self.migration_cfg = MigrationConfig()  # ADDED: user migration settings

    def add_new_user(self, user: User):
        if user.username in self.active_participants:
            raise ValueError(f"User '{user.username}' is already an active participant in this node.")

        self.active_participants[user.username] = user
        for comment in user.processed_comments:
            self.add_new_comment(comment, from_user_sync=True)

    def add_new_comment(self, comment: ProcessedComment, from_user_sync: bool = False):
        self.v_topic = (1 - self.topicAdaptationFactor) * self.v_topic + self.topicAdaptationFactor * comment.semantic_embedding
        self.v_emotion = (1 - self.emotionAdaptationFactor) * self.v_emotion + self.emotionAdaptationFactor * comment.emotion_embedding
        
        tot = np.sum(self.v_emotion)
        if tot > 0:
            self.v_emotion /= tot

        self.entropy = self.compute_entropy(self.v_emotion)

        author_name = comment.comment.author
        if author_name in self.active_participants and not from_user_sync:
            self.active_participants[author_name].add_comment(comment)

    def compute_entropy(self, v_emotion: np.ndarray) -> float:
        p = v_emotion[v_emotion > 1e-12]
        return float(-np.sum(p * np.log2(p)))

    def add_neighbor_edge(self, neighbor_node_id: int, transition_edge: TransitionEdge):
        if neighbor_node_id in self.neighbor_edges:
            raise ValueError(f"Neighbor edge to node '{neighbor_node_id}' already exists.")
        self.neighbor_edges[neighbor_node_id] = transition_edge

    def propagate_token(self, token: TransientGossipToken):
        if not self.neighbor_edges:
            return

        neighbor_ids = list(self.neighbor_edges.keys())

        if self.Node_State == BubbleState.HOT:
            k = min(2, len(neighbor_ids))
            routing_scores = []
            for nid in neighbor_ids:
                edge = self.neighbor_edges[nid]
                # CHANGED (point 2): pass self.id (the caller), not nid; the old call returned our own entropy
                stability_margin = max(0.01, self.tresholdEntropy - edge.get_opposing_side_entropy(self.id))
                routing_scores.append(edge.weight * edge.affinity_score * stability_margin)

            softmax_probabilities = self.softmax(np.array(routing_scores, dtype=np.float32))
            selected_choices = np.random.choice(
                neighbor_ids,
                size=k,
                replace=False,
                p=softmax_probabilities
            )
        else:
            # Normal propagation: pick 1 neighbor uniformly or by affinity
            selected_choices = [random.choice(neighbor_ids)]

        for target_id in selected_choices:
            self.outbound_tokens_buffer.setdefault(target_id, []).append(token)
            edge = self.neighbor_edges[target_id]
            print(f"Propagating token {token.token_id} to neighbor node {target_id} via edge {edge}.")

    def direct_propagate_token(self, token: TransientGossipToken, exclude_node_id: int = None):
        """
        Propagates a token to neighboring nodes using softmax-weighted selection
        based on edge weight, affinity, and neighbor stability margins.
        Prevents echo loops by excluding the sender/origin node.
        """
        if not self.neighbor_edges:
            return

        # 1. Filter out the node that just sent or originated this token to prevent echoes
        candidate_ids = [
            nid for nid in self.neighbor_edges.keys()
            if nid != exclude_node_id and nid != token.origin_node
        ]

        if not candidate_ids:
            return

        # 2. Determine fanout
        if token.source_reason == SourceReason.HIGH_ENTROPY_PROPAGATION:
            fanout = 1  # Single targeted relief path per hop
        else:
            fanout = min(2, len(candidate_ids))
        if token.source_reason == SourceReason.HIGH_ENTROPY_PROPAGATION:
            # 3. Compute routing scores favoring calmer, high-affinity neighbors
            routing_scores = []
            for nid in candidate_ids:
                edge = self.neighbor_edges[nid]
                neighbor_entropy = edge.get_opposing_side_entropy(self.id)  # CHANGED (point 2): was (nid)
                
                # Prioritize neighbors with healthy stability margins (calmer nodes)
                stability_margin = max(0.01, self.tresholdEntropy - neighbor_entropy)
                routing_scores.append(edge.weight * edge.affinity_score * stability_margin)

            # 4. Softmax probability sampling without replacement
            softmax_probabilities = self.softmax(np.array(routing_scores, dtype=np.float32))
            selected_choices = np.random.choice(
                candidate_ids,
                size=min(fanout, len(candidate_ids)),
                replace=False,
                p=softmax_probabilities
            )

        elif token.source_reason == SourceReason.REGULAR_PROPAGATION:
            selected_choices = random.sample(candidate_ids, k=min(fanout, len(candidate_ids)))
            
        # 5. Dispatch to outbound buffer
        for target_id in selected_choices:
            self.outbound_tokens_buffer.setdefault(target_id, []).append(token)
            edge = self.neighbor_edges[target_id]
            print(f"Propagating token {token.token_id} to neighbor node {target_id} via edge {edge}.")

    def receive_token(self, token: TransientGossipToken):
            """
            Receives a transient gossip token. If the token is an alert from an overheated
            node, ingests the state and probabilistically emits a stabilizing counter-token.
            """
            if token.state == GossipState.EXPIRED:
                print(f"Node {self.id} received expired token {token.token_id}. Ignoring.")
                return
            
            # 1. Standard duplicate check
            if token in self.active_tokens:
                return
            self.active_tokens.append(token)
            self.token_counter += 1

            token.decay_intensity(local_entropy=self.entropy, dt=1.0)

            token_intensity_modifier = token.current_intensity

            
            h_before = self.entropy

            eff_beta = self.emotionAdaptationFactor * token_intensity_modifier
            self.v_emotion = (1.0 - eff_beta) * self.v_emotion + eff_beta * token.emotional_embedding
            tot = np.sum(self.v_emotion)
            if tot > 0:
                self.v_emotion /= tot

            # Recompute entropy
            self.entropy = self.compute_entropy(self.v_emotion)

            # 2. Handle High-Entropy Alert Tokens
            if token.source_reason == SourceReason.HIGH_ENTROPY_PROPAGATION:
                    
                # Evaluate thermodynamic response capacity
                delta_h = max(0.0, self.entropy - h_before)
                stability_margin = max(0.0, self.tresholdEntropy - self.entropy)

                # Sensitivity factor (kappa) modulates reaction intensity
                kappa = 3.0
                p_counter_propagate = float(np.clip(delta_h * stability_margin * kappa, 0.0, 1.0))

                # Also boost active_propagation_chance for subsequent routine steps
                self.active_propagation_chance = min(
                    self.max_propagation_chance,
                    self.active_propagation_chance + 0.1 * stability_margin
                )

                # Roll to fire an immediate stabilizing counter-token
                if random.random() < p_counter_propagate:
                    response_token = TransientGossipToken(
                        token_id=f"stabilizer_from_{self.id}_to_{token.origin_node}_{self.last_sync_timestamp}",
                        origin_node=self.id,
                        semantic_embedding=np.copy(self.v_topic),
                        emotional_embedding=np.copy(self.v_emotion),
                        GossipState=GossipState.ACTIVE,
                        issuance_time=self.last_sync_timestamp,
                        dynamic_ttl=5.0,
                        source_reason=SourceReason.REGULAR_PROPAGATION
                    )

                    # Route directly back to the origin node if connected, or through the edge
                    target_id = token.origin_node
                    if target_id in self.neighbor_edges:
                        self.outbound_tokens_buffer.setdefault(target_id, []).append(response_token)
                        print(f"Node {self.id} dispatched stabilizing token to overheated Node {target_id}.")

            if token.state != GossipState.EXPIRED:
                # Continue propagating the token if still active
                self.direct_propagate_token(token)

    def contains_user(self, username: str) -> bool:
        return username in self.active_participants

    def reorganization_treshold_reached(self) -> bool:
        return self.entropy > self.tresholdEntropy

    def advance_time_step(self, time_stamp: str):
        # CHANGED: was `edge.post_entropy(self.id, self.entropy)` per edge. post_snapshot advertises
        # (v_topic, v_emotion, entropy) and still updates the advertised entropy.
        self.post_snapshot()

        time_increment = self.convert_str_timestamp_to_float(time_stamp)
        self.last_sync_timestamp = time_increment if self.last_sync_timestamp is not None else 0.0

        if self.propagation_cooldown > 0:
            self.propagation_cooldown -= 1

        if self.reorganization_treshold_reached():
            self.Node_State = BubbleState.HOT
            if self.active_propagation_chance < self.max_propagation_chance:
                self.active_propagation_chance = min(self.active_propagation_chance + 0.05, self.max_propagation_chance)

            if self.propagation_cooldown == 0 and random.random() < self.active_propagation_chance:
                token = TransientGossipToken(
                    token_id=f"token_{self.token_counter}_from_node_{self.id}",
                    origin_node=self.id,
                    semantic_embedding=np.copy(self.v_topic),
                    emotional_embedding=np.copy(self.v_emotion),
                    GossipState=GossipState.ACTIVE,
                    issuance_time=self.last_sync_timestamp,
                    dynamic_ttl=10.0,
                    source_reason=SourceReason.HIGH_ENTROPY_PROPAGATION
                )
                self.propagate_token(token)
                self.propagation_cooldown = self.base_cooldown
        else:
            self.Node_State = BubbleState.NORMAL
            if self.active_propagation_chance > self.default_propagation_chance:
                self.active_propagation_chance = max(self.active_propagation_chance - 0.05, self.default_propagation_chance)

            if random.random() < self.active_propagation_chance:
                token = TransientGossipToken(
                    token_id=f"token_{len(self.active_tokens)}_from_node_{self.id}",
                    origin_node=self.id,
                    semantic_embedding=np.copy(self.v_topic),
                    emotional_embedding=np.copy(self.v_emotion),
                    GossipState=GossipState.ACTIVE,
                    issuance_time=self.last_sync_timestamp,
                    dynamic_ttl=10.0,
                    source_reason=SourceReason.REGULAR_PROPAGATION
                )
                self.propagate_token(token)

    def convert_str_timestamp_to_float(self, timestamp_str: str | float | int) -> float:
            """
            Converts a timestamp (ISO string, epoch string, integer step, or float)
            to a float representation.
            """
            if isinstance(timestamp_str, (int, float)):
                return float(timestamp_str)

            # 1. Check if the string is a raw numeric step or Unix epoch (e.g. '1', '1617234891.0')
            try:
                return float(timestamp_str)
            except ValueError:
                pass

            # 2. Parse ISO 8601 strings (e.g. '2026-10-05T22:30:00')
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp_str)
            return dt.timestamp()
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) 