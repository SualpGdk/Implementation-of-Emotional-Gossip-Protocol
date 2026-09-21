from bubble import BubbleNode
import numpy as np
from TransitionEdge import TransitionEdge


class SimEngine:
    def __init__(self, bubble_nodes: list[BubbleNode], chronological_comments: list):

        # Setting initial environment for the simulation
        self.bubble_nodes = bubble_nodes
        self.chronological_comments = chronological_comments
        self.time_step = 0

        # Affinity weighting coefficients
        self.topicWeightingCoefficient = 0.5
        self.emotionWeightingCoefficient = 0.5

        self.edges = []

        for source_node in self.bubble_nodes:

            affinity_scores = []

            for target_node in self.bubble_nodes:

                # A node cannot have an edge to itself
                if source_node.id == target_node.id:
                    continue

                affinity_score = self.calculate_affinity_score(
                    source_node,
                    target_node
                )

                affinity_scores.append(
                    (target_node, affinity_score)
                )

            # Sort by affinity, highest first, and take top 3
            processed_affinity_scores = sorted(
                affinity_scores,
                key=lambda x: x[1],
                reverse=True
            )[:3]

            # Create edges to the 3 most affine nodes
            for target_node, score in processed_affinity_scores:

                edge = TransitionEdge(
                    source_node.id,
                    target_node.id,
                    weight=score,
                    topicWeightingCoefficient=self.topicWeightingCoefficient,
                    emotionWeightingCoefficient=self.emotionWeightingCoefficient
                )

                self.edges.append(edge)

    def run_simulation(self):
        for comment in self.chronological_comments:
            self.time_step += 1

            print(f"Time Step: {self.time_step}")

            for node in self.bubble_nodes:
                node.add_new_comment(comment)

    def calculate_affinity_score(
        self,
        source_node: BubbleNode,
        target_node: BubbleNode
    ):

        topic_distance = np.linalg.norm(
            source_node.v_topic - target_node.v_topic
        ) ** 2

        emotion_distance = np.linalg.norm(
            source_node.v_emotion - target_node.v_emotion
        ) ** 2

        affinity_score = np.exp(
            -self.topicWeightingCoefficient * topic_distance
            -self.emotionWeightingCoefficient * emotion_distance
        )

        return affinity_score