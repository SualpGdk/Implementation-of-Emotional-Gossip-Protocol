import numpy as np
from bubble import BubbleNode
from TransitionEdge import TransitionEdge
from processed_comment import ProcessedComment
from user import User  # ADDED: needed to register unknown authors as participants
from migration import run_migration_round  # ADDED: user migration

class SimEngine:
    def __init__(self, bubble_nodes: list[BubbleNode], chronological_comments: list,
                 migration_interval: int = 1,        # ADDED: evaluate migrations every N comment-steps
                 max_inflow_fraction: float = 0.10):  # ADDED: per-destination inflow cap per round
        self.bubble_nodes = {node.id: node for node in bubble_nodes}
        self.chronological_comments = chronological_comments
        self.time_step = 0
        self.migration_interval = migration_interval    # ADDED
        self.max_inflow_fraction = max_inflow_fraction  # ADDED
        self.total_migrations = 0                       # ADDED

        self.topicWeightingCoefficient = 0.5
        self.emotionWeightingCoefficient = 0.5
        self.edges: list[TransitionEdge] = []
        edge_by_pair: dict[frozenset, TransitionEdge] = {}  # ADDED: one edge per unordered node pair

        # 1. Build and register top-3 affinity edges for each bubble
        # CHANGED (undirected edges): each pair gets ONE shared TransitionEdge registered on BOTH nodes.
        # Previously every (source -> target) pick created its own edge, registered on the source node only.
        # source_node_id / target_node_id on the edge are now just the two endpoints (legacy names).
        for source_node in self.bubble_nodes.values():
            affinity_scores = []
            for target_node in self.bubble_nodes.values():
                if source_node.id == target_node.id:
                    continue

                affinity_score = self.calculate_affinity_score(source_node, target_node)
                affinity_scores.append((target_node, affinity_score))

            top_candidates = sorted(affinity_scores, key=lambda x: x[1], reverse=True)[:3]

            for target_node, score in top_candidates:
                # ADDED (undirected edges): skip if the other endpoint already created this edge
                pair = frozenset((source_node.id, target_node.id))
                if pair in edge_by_pair:
                    continue
                edge = TransitionEdge(
                    source_node.id,
                    target_node.id,
                    weight=score,
                    topicWeightingCoefficient=self.topicWeightingCoefficient,
                    emotionWeightingCoefficient=self.emotionWeightingCoefficient
                )
                self.edges.append(edge)
                edge_by_pair[pair] = edge  # ADDED
                # Register edge directly on the node so neighbor_edges is populated
                source_node.add_neighbor_edge(target_node.id, edge)
                # ADDED (undirected edges): the same instance on the other endpoint, so both sides
                # share weight, affinity, snapshots and migration_count
                target_node.add_neighbor_edge(source_node.id, edge)

    def run_simulation(self):
        for user_comment in self.chronological_comments:
            self.time_step += 1
            timestamp_str = getattr(user_comment.comment, 'created_utc', str(self.time_step))

            # -----------------------------------------------------------------
            # Step 1: Route Incoming User Comment (Semantic & Participant Match)
            # -----------------------------------------------------------------
            self._route_comment(user_comment)

            # -----------------------------------------------------------------
            # Step 2: Advance Time & Detect Entropy Anomalies
            # -----------------------------------------------------------------
            for node in self.bubble_nodes.values():
                node.advance_time_step(timestamp_str)

            # -----------------------------------------------------------------
            # Step 3: Network Transport (Drain Buffers & Deliver Gossip Tokens)
            # -----------------------------------------------------------------
            self._deliver_outbound_tokens()

            # -----------------------------------------------------------------
            # Step 4: Maintain Edge Dynamics
            # -----------------------------------------------------------------
            self._update_network_edges()

            # -----------------------------------------------------------------
            # Step 5: User Migration (ADDED)
            # -----------------------------------------------------------------
            if self.time_step % self.migration_interval == 0:
                self.total_migrations += run_migration_round(
                    self.bubble_nodes, self.time_step, self.max_inflow_fraction)

        print("Reached end of comments. Simulation complete.")

    def _route_comment(self, user_comment: ProcessedComment):
        author_name = user_comment.comment.author
        for node in self.bubble_nodes.values():
            if node.contains_user(author_name):
                node.add_new_comment(user_comment, from_user_sync=False)
                return

        # Fallback: assign to the semantically closest bubble
        closest_node = None
        max_sim = -1.0
        for node in self.bubble_nodes.values():
            sim = self.calculate_cosine_similarity(node.v_topic, user_comment.semantic_embedding)
            if sim > max_sim:
                closest_node = node
                max_sim = sim

        if closest_node is not None:
            # ADDED: register the author as a participant first. Before, an unknown author's comment only
            # moved the bubble's vectors and the author was never remembered, so they could never migrate
            # and every later comment went through this fallback again.
            closest_node.add_new_user(User(author_name))
            closest_node.add_new_comment(user_comment, from_user_sync=False)

    def _deliver_outbound_tokens(self):
        """
        Transports all queued tokens across the network overlay
        from source outbound buffers into target receive_token pipelines.
        """
        for source_node in self.bubble_nodes.values():
            for target_id, tokens in list(source_node.outbound_tokens_buffer.items()):
                target_node = self.bubble_nodes.get(target_id)
                if not target_node:
                    continue

                while tokens:
                    token = tokens.pop(0)
                    target_node.receive_token(token)

    def _update_network_edges(self):
        """Updates affinity scores and posts entropy levels across all edges."""
        # CHANGED: was a per-edge affinity override + post_entropy for both ends (and W was never updated).
        # Nodes post fresh (v_topic, v_emotion, entropy) snapshots here, i.e. AFTER this step's comments and
        # token delivery, then each shared edge recomputes A_ij, emotional delta and W_ij exactly once.
        for node in self.bubble_nodes.values():
            node.post_snapshot()
        for edge in self.edges:
            edge.refresh()

    def calculate_affinity_score(self, source_node: BubbleNode, target_node: BubbleNode):
        topic_distance = np.linalg.norm(source_node.v_topic - target_node.v_topic) ** 2
        emotion_distance = np.linalg.norm(source_node.v_emotion - target_node.v_emotion) ** 2

        return float(np.exp(
            -self.topicWeightingCoefficient * topic_distance
            -self.emotionWeightingCoefficient * emotion_distance
        ))

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot_product / norm_product) if norm_product > 0 else 0.0