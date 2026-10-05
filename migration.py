"""
NEW FILE: semantic user migration between bubbles.

BubbleNode inherits UserMigrationMixin and must define:
    id, v_topic, v_emotion, entropy, tresholdEntropy,
    neighbor_edges, active_participants, softmax(), compute_entropy(),
    migration_cfg (a MigrationConfig)
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class MigrationConfig:
    w_topic: float = 0.6          # topic vs. emotion weight inside sim()
    temperature: float = 0.05     # softmax temperature (lower = more decisive)
    threshold: float = 0.55       # min p(best destination)
    margin: float = 0.10          # min p(best) - p(stay)
    patience: int = 3             # consecutive steps wanting the SAME target
    cooldown: int = 10            # steps a user is frozen after moving
    min_calm: float = 0.25        # floor for the calm factor (lower = harder push out of hot bubbles)
    landing_weight: float = 0.05  # how strongly a migrant imprints on its new bubble
    departure_pull: float = 0.05  # how far the old bubble drifts toward those who stayed
    edge_boost: float = 0.02      # W_ij bump per migration (historical traffic)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0.0, float(a @ b) / d) if d > 0 else 0.0


def hellinger_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """1 - Hellinger distance; both inputs must be normalized histograms."""
    p, q = np.clip(p, 0, None), np.clip(q, 0, None)
    return 1.0 - float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


class UserMigrationMixin:

    # ---------- helpers ----------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=np.float32), 0, None)
        s = v.sum()
        return v / s if s > 0 else v

    def _user_similarity(self, u_topic, u_emo, node_topic, node_emo) -> float:
        w = self.migration_cfg.w_topic
        return w * cosine(u_topic, node_topic) + (1 - w) * hellinger_similarity(u_emo, node_emo)

    def _calm(self, entropy: float) -> float:
        """1 for a perfectly calm bubble, falling to min_calm as entropy nears log2(m)."""
        h_max = max(float(np.log2(len(self.v_emotion))), 1e-6)
        return max(self.migration_cfg.min_calm, 1.0 - entropy / h_max)

    # ---------- decentralized state sharing ----------
    def post_snapshot(self):
        """Replaces the post_entropy loop in advance_time_step."""
        for edge in self.neighbor_edges.values():
            edge.post_snapshot(self.id, self.v_topic, self.v_emotion, self.entropy)

    # ---------- decision ----------
    def evaluate_user_migrations(self, step: int) -> list[tuple[str, int, float]]:
        """
        p(u -> k) = softmax_k( sim(u, k) * W_ik * calm(k) / temperature ),
        over {self} U neighbors, with W_ii = 1. A hot host has a low calm
        factor, which is the "gravitational push" from the paper.
        Returns [(username, target_node_id, confidence)]; nothing is mutated here.
        """
        cfg = self.migration_cfg

        candidates = []  # (nid, topic, emotion, entropy, weight)
        for nid, edge in self.neighbor_edges.items():
            snap = edge.get_opposing_snapshot(self.id)   # self.id, not nid
            if snap is not None:
                candidates.append((nid, *snap, edge.weight))
        if not candidates:
            return []

        moves = []
        for name, user in self.active_participants.items():
            if user.rolling_semantic_vector is None:
                continue
            if step - user.last_migration_step < cfg.cooldown:
                user.migration_streak, user.migration_target = 0, None
                continue

            u_topic = user.rolling_semantic_vector
            u_emo = user.rolling_emotion_histogram()

            ids = [self.id]
            scores = [self._user_similarity(u_topic, u_emo, self.v_topic, self.v_emotion)
                      * self._calm(self.entropy)]
            for nid, t, e, h, w in candidates:
                ids.append(nid)
                scores.append(self._user_similarity(u_topic, u_emo, t, e) * w * self._calm(h))

            p = self.softmax(np.array(scores, dtype=np.float32) / cfg.temperature)
            user.transition_probabilities = dict(zip(ids, p.tolist()))

            best = int(np.argmax(p))
            wants_move = (best != 0 and p[best] >= cfg.threshold
                          and p[best] - p[0] >= cfg.margin)
            if wants_move and user.migration_target == ids[best]:
                user.migration_streak += 1
            elif wants_move:
                user.migration_streak, user.migration_target = 1, ids[best]
            else:
                user.migration_streak, user.migration_target = 0, None

            if user.migration_streak >= cfg.patience:
                moves.append((name, ids[best], float(p[best])))
        return moves

    # ---------- execution ----------
    def remove_user(self, username: str):
        """Remove a user; the bubble's centroid drifts toward those who stayed."""
        user = self.active_participants.pop(username)
        rest = [u for u in self.active_participants.values()
                if u.rolling_semantic_vector is not None]
        r = self.migration_cfg.departure_pull
        if rest and r > 0:
            mean_t = np.mean([u.rolling_semantic_vector for u in rest], axis=0)
            mean_e = np.mean([u.rolling_emotion_histogram() for u in rest], axis=0)
            self.v_topic = ((1 - r) * self.v_topic + r * mean_t).astype(np.float32)
            self.v_emotion = self._normalize((1 - r) * self.v_emotion + r * mean_e)
            self.entropy = self.compute_entropy(self.v_emotion)
        return user

    def admit_migrant(self, user):
        """Soft landing: no history replay, just a small imprint of the user's current state."""
        if user.username in self.active_participants:
            raise ValueError(f"User '{user.username}' is already in node {self.id}.")
        self.active_participants[user.username] = user
        w = self.migration_cfg.landing_weight
        if user.rolling_semantic_vector is not None:
            self.v_topic = ((1 - w) * self.v_topic + w * user.rolling_semantic_vector).astype(np.float32)
            self.v_emotion = self._normalize((1 - w) * self.v_emotion + w * user.rolling_emotion_histogram())
            self.entropy = self.compute_entropy(self.v_emotion)


# ---------- coordinator (graph / simulation loop side) ----------
def refresh_edges(nodes: dict[int, "UserMigrationMixin"]):
    """Call once per step, after every node posted its snapshot. Updates A_ij and W_ij."""
    unique = {id(e): e for n in nodes.values() for e in n.neighbor_edges.values()}
    for edge in unique.values():
        edge.refresh()


def run_migration_round(nodes: dict[int, "UserMigrationMixin"], step: int,
                        max_inflow_fraction: float = 0.10) -> int:
    """
    Two phases so node iteration order does not matter:
      1. every node proposes moves against the same snapshot state
      2. proposals are applied, with a per-destination inflow cap to prevent
         stampedes that would turn the calm bubble hot.
    Rejected users keep their streak and retry next step, in confidence order.
    """
    proposals = [(n.id, name, tgt, conf)
                 for n in nodes.values()
                 for name, tgt, conf in n.evaluate_user_migrations(step)]
    random.shuffle(proposals)                         # break ties fairly...
    proposals.sort(key=lambda x: x[3], reverse=True)  # ...but most confident users get the capped slots first

    inflow, moved = defaultdict(int), 0
    for src, name, tgt, _ in proposals:
        cap = max(1, int(max_inflow_fraction * len(nodes[tgt].active_participants)))
        if inflow[tgt] >= cap:
            continue
        user = nodes[src].remove_user(name)
        nodes[tgt].admit_migrant(user)
        user.last_migration_step = step
        user.migration_streak, user.migration_target = 0, None
        nodes[src].neighbor_edges[tgt].record_migration(nodes[src].migration_cfg.edge_boost)
        inflow[tgt] += 1
        moved += 1
    return moved
