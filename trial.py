"""
====================================================================================================
Emotionally and Context-Aware Gossip Protocol (ECAGP / Dynamic Social Bubbles) Simulator v1.2
Department of Computer Engineering, TOBB University of Economics and Technology
Reference: Dr. Cigdem Avci et al. (2026)

Modules & Capabilities:
1. Continuous 3D Affective State Space e = (v, a, s) in [-1, 1] x [0, 1] x [0, 1]
   - v: Valence (negative/hostile to positive/constructive)
   - a: Arousal (quiescent to hyper-aroused/agitated)
   - s: Cognitive Stress (calm to overloaded)
2. Soft Gaussian RBF Affective Histogram Distribution v_emotion in Delta^{m-1} & Shannon Entropy H(B_i)
3. Dynamic Semantic Topic Centroid v_topic in R^d with Recency-Sensitive EMA Updates
4. Multi-Modal Emotion Fusion Layer: psi(M_u) = w_text*f_text + w_beh*f_beh + w_env*f_env
5. Elastic Semantic-Affective Gaussian Affinity Kernel A_ij & Dynamic Transition Edges W_ij
6. Transient Token Lifecycle FSM: {ACTIVE, DAMPENED, EXPIRED} with Thermodynamic Intensity Decay
7. Dynamic Fanout Attenuation: F_i(t) throttled when a_bar_i > Theta_reg (0.38)
8. Thermodynamic Restorative Counter-Signaling (Homeostasis Injection on Panic/Toxic Stimuli)
9. Four Routing Policies Benchmark:
   - Classic Unregulated
   - Emotion-Aware
   - Bubble-Graph Proximity
   - Dual-Regulated (ECAGP v1.2)
====================================================================================================
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class GossipState(Enum):
    ACTIVE = "ACTIVE"
    DAMPENED = "DAMPENED"
    EXPIRED = "EXPIRED"


class DisseminationPolicy(Enum):
    CLASSIC_UNREGULATED = "Classic Unregulated"
    EMOTION_AWARE = "Emotion-Aware"
    BUBBLE_PROXIMITY = "Bubble-Graph Proximity"
    DUAL_REGULATED = "Dual-Regulated (ECAGP v1.2)"


@dataclass
class AffectiveState:
    valence: float  # [-1.0, 1.0]
    arousal: float  # [0.0, 1.0]
    stress: float   # [0.0, 1.0]

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.stress], dtype=float)

    def clamp(self):
        self.valence = max(-1.0, min(1.0, float(self.valence)))
        self.arousal = max(0.0, min(1.0, float(self.arousal)))
        self.stress = max(0.0, min(1.0, float(self.stress)))

    def copy(self) -> "AffectiveState":
        return AffectiveState(self.valence, self.arousal, self.stress)


# 8 Canonical Affective Prototypes in (v, a, s) space
AFFECTIVE_PROTOTYPES = np.array([
    [0.8, 0.8, 0.2],   # 0: High-arousal positive (Joy / Enthusiasm)
    [0.7, 0.2, 0.1],   # 1: Low-arousal positive (Calm / Serenity)
    [-0.8, 0.9, 0.8],  # 2: High-arousal negative (Panic / Terror / Hostility)
    [-0.6, 0.3, 0.5],  # 3: Low-arousal negative (Depression / Sadness)
    [0.0, 0.3, 0.2],   # 4: Neutral Baseline / Homeostasis
    [-0.4, 0.7, 0.7],  # 5: Agitated Frustration / Friction
    [0.5, 0.6, 0.3],   # 6: Constructive Engagement
    [-0.3, 0.5, 0.9]   # 7: Severe Cognitive Overload / Burnout
])


def compute_soft_distribution(e: AffectiveState, sigma: float = 0.45) -> np.ndarray:
    arr = e.to_array()
    diffs = AFFECTIVE_PROTOTYPES - arr
    sq_dists = np.sum(diffs ** 2, axis=1)
    weights = np.exp(-sq_dists / (2 * (sigma ** 2)))
    sum_w = np.sum(weights)
    if sum_w <= 1e-12:
        return np.ones(len(AFFECTIVE_PROTOTYPES)) / len(AFFECTIVE_PROTOTYPES)
    return weights / sum_w


def compute_shannon_entropy(prob_dist: np.ndarray, epsilon: float = 1e-6) -> float:
    p = np.clip(prob_dist, epsilon, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


@dataclass
class TransientGossipToken:
    token_id: str
    origin_node: int
    semantic_embedding: np.ndarray
    affective_tone: AffectiveState
    current_intensity: float = 1.0
    state: GossipState = GossipState.ACTIVE
    issuance_time: int = 0
    remaining_ttl: int = 8
    hops: int = 0

    def is_expired(self) -> bool:
        return self.state == GossipState.EXPIRED or self.remaining_ttl <= 0 or self.current_intensity <= 0.02


@dataclass
class TransitionEdge:
    source_node: int
    target_node: int
    weight_w: float = 0.5
    semantic_affinity: float = 0.5
    emotional_delta: float = 0.0


class BubbleNode:
    def __init__(self, node_id: int, topic_dim: int = 4):
        self.node_id = node_id
        self.v_topic = np.random.uniform(0.1, 0.9, size=topic_dim)
        self.v_topic /= np.linalg.norm(self.v_topic)

        self.current_affect = AffectiveState(
            valence=float(np.random.uniform(0.15, 0.35)),
            arousal=float(np.random.uniform(0.15, 0.25)),
            stress=float(np.random.uniform(0.10, 0.20))
        )
        self.v_emotion = compute_soft_distribution(self.current_affect)
        self.current_entropy = compute_shannon_entropy(self.v_emotion)
        self.mean_arousal = self.current_affect.arousal

        self.neighbors: List[int] = []
        self.edges: Dict[int, TransitionEdge] = {}
        self.seen_tokens: set = set()

        self.messages_received = 0
        self.counter_signals_generated = 0
        self.dampened_count = 0

    def add_neighbor(self, neighbor_id: int):
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
            self.edges[neighbor_id] = TransitionEdge(
                source_node=self.node_id,
                target_node=neighbor_id,
                weight_w=0.5
            )

    def natural_thermodynamic_decay(self, lambda_decay: float = 0.08, baseline_arousal: float = 0.20):
        self.current_affect.arousal = baseline_arousal + (self.current_affect.arousal - baseline_arousal) * (1.0 - lambda_decay)
        self.current_affect.stress = 0.15 + (self.current_affect.stress - 0.15) * (1.0 - lambda_decay)
        self.current_affect.valence = 0.25 + (self.current_affect.valence - 0.25) * (1.0 - lambda_decay)
        self.current_affect.clamp()
        self.v_emotion = compute_soft_distribution(self.current_affect)
        self.current_entropy = compute_shannon_entropy(self.v_emotion)
        self.mean_arousal = self.current_affect.arousal

    def update_state_on_message(self, token: TransientGossipToken, alpha: float = 0.15, beta: float = 0.20):
        self.v_topic = (1.0 - alpha) * self.v_topic + alpha * token.semantic_embedding
        norm = np.linalg.norm(self.v_topic)
        if norm > 1e-6:
            self.v_topic /= norm

        eff_beta = beta * token.current_intensity
        self.current_affect.valence = (1.0 - eff_beta) * self.current_affect.valence + eff_beta * token.affective_tone.valence
        self.current_affect.arousal = (1.0 - eff_beta) * self.current_affect.arousal + eff_beta * token.affective_tone.arousal
        self.current_affect.stress = (1.0 - eff_beta) * self.current_affect.stress + eff_beta * token.affective_tone.stress
        self.current_affect.clamp()

        self.v_emotion = compute_soft_distribution(self.current_affect)
        self.current_entropy = compute_shannon_entropy(self.v_emotion)
        self.mean_arousal = self.current_affect.arousal

    def apply_thermodynamic_counter_signal(self, token: TransientGossipToken, delta: float = 0.45, lambda_decay: float = 0.08):
        if token.affective_tone.valence < -0.6 and token.affective_tone.arousal > 0.7:
            counter_valence = -delta * token.affective_tone.valence
            self.current_affect.valence = 0.55 * self.current_affect.valence + 0.45 * counter_valence
            self.current_affect.arousal *= (1.0 - lambda_decay * 2.0)
            self.current_affect.stress *= 0.70
            self.current_affect.clamp()

            self.v_emotion = compute_soft_distribution(self.current_affect)
            self.current_entropy = compute_shannon_entropy(self.v_emotion)
            self.mean_arousal = self.current_affect.arousal
            self.counter_signals_generated += 1

    def compute_dynamic_fanout(self, f0: int = 4, theta_reg: float = 0.38, kappa: float = 0.85) -> int:
        if self.mean_arousal <= theta_reg:
            return f0
        ratio = (self.mean_arousal - theta_reg) / (1.0 - theta_reg + 1e-6)
        attenuation = 1.0 - kappa * ratio
        throttled = int(math.floor(f0 * max(0.1, attenuation)))
        return max(1, throttled)

    def damp_token(self, token: TransientGossipToken, lambda_decay: float = 0.08,
                   mu1: float = 0.35, mu2: float = 0.15, theta_reg: float = 0.38,
                   tau_damp: float = 0.35):
        delta_arousal = max(0.0, self.mean_arousal - theta_reg)
        dI = (lambda_decay * token.current_intensity +
              mu1 * delta_arousal +
              mu2 * (self.current_entropy / 3.0))

        token.current_intensity = max(0.0, token.current_intensity - dI)
        token.remaining_ttl -= 1
        token.hops += 1

        if token.current_intensity <= 0.02 or token.remaining_ttl <= 0:
            token.state = GossipState.EXPIRED
        elif token.current_intensity < tau_damp:
            token.state = GossipState.DAMPENED
            self.dampened_count += 1
        else:
            token.state = GossipState.ACTIVE

    def update_transition_edges(self, nodes_dict: Dict[int, "BubbleNode"],
                                omega: float = 0.70, gamma1: float = 0.5, gamma2: float = 0.5,
                                epsilon: float = 1e-6):
        for n_id, edge in self.edges.items():
            neighbor = nodes_dict[n_id]
            d_topic = float(np.sum((self.v_topic - neighbor.v_topic) ** 2))
            d_emotion = float(np.sum((self.v_emotion - neighbor.v_emotion) ** 2))
            affinity = math.exp(-gamma1 * d_topic - gamma2 * d_emotion)
            edge.semantic_affinity = affinity

            h_i = self.current_entropy
            h_j = neighbor.current_entropy
            entropy_penalty = 1.0 - (abs(h_i - h_j) / (max(h_i, h_j) + epsilon))
            target_w = affinity * max(0.0, entropy_penalty)

            edge.weight_w = omega * edge.weight_w + (1.0 - omega) * target_w


class ECAGPSimulator:
    def __init__(self,
                 node_count: int = 80,
                 base_fanout: int = 4,
                 theta_reg: float = 0.38,
                 kappa: float = 0.85,
                 lambda_decay: float = 0.08,
                 mu1: float = 0.35,
                 mu2: float = 0.15,
                 alpha: float = 0.15,
                 beta: float = 0.20,
                 gamma1: float = 0.5,
                 gamma2: float = 0.5,
                 omega: float = 0.70,
                 seed: int = 42):
        self.node_count = node_count
        self.base_fanout = base_fanout
        self.theta_reg = theta_reg
        self.kappa = kappa
        self.lambda_decay = lambda_decay
        self.mu1 = mu1
        self.mu2 = mu2
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.omega = omega
        self.seed = seed

        self.nodes: Dict[int, BubbleNode] = {}
        self._build_topology()

    def _build_topology(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.nodes = {i: BubbleNode(node_id=i, topic_dim=4) for i in range(self.node_count)}

        k_nearest = 3
        for i in range(self.node_count):
            for offset in range(1, k_nearest + 1):
                j = (i + offset) % self.node_count
                self.nodes[i].add_neighbor(j)
                self.nodes[j].add_neighbor(i)

        for i in range(self.node_count):
            if random.random() < 0.20:
                peer = random.randint(0, self.node_count - 1)
                if peer != i:
                    self.nodes[i].add_neighbor(peer)
                    self.nodes[peer].add_neighbor(i)

    def inject_panic_cascade(self, seed_nodes: List[int], current_step: int) -> List[TransientGossipToken]:
        tokens = []
        for s in seed_nodes:
            token = TransientGossipToken(
                token_id=f"panic_{s}_{current_step}_{random.randint(1000, 9999)}",
                origin_node=s,
                semantic_embedding=np.array([-0.85, 0.75, 0.90, -0.65]),
                affective_tone=AffectiveState(valence=-0.92, arousal=0.96, stress=0.88),
                current_intensity=1.0,
                state=GossipState.ACTIVE,
                issuance_time=current_step,
                remaining_ttl=8
            )
            tokens.append(token)
            self.nodes[s].current_affect = AffectiveState(valence=-0.92, arousal=0.96, stress=0.88)
            self.nodes[s].v_emotion = compute_soft_distribution(self.nodes[s].current_affect)
            self.nodes[s].current_entropy = compute_shannon_entropy(self.nodes[s].v_emotion)
            self.nodes[s].mean_arousal = self.nodes[s].current_affect.arousal
        return tokens

    def select_targets(self, node: BubbleNode, token: TransientGossipToken, policy: DisseminationPolicy) -> List[int]:
        neighbors = node.neighbors
        if not neighbors:
            return []

        if policy == DisseminationPolicy.CLASSIC_UNREGULATED:
            fanout = min(len(neighbors), self.base_fanout)
            return random.sample(neighbors, fanout)

        elif policy == DisseminationPolicy.EMOTION_AWARE:
            weights = []
            for n_id in neighbors:
                nbr = self.nodes[n_id]
                w = max(0.05, 1.0 - nbr.mean_arousal)
                weights.append(w)
            weights = np.array(weights) / sum(weights)
            fanout = min(len(neighbors), self.base_fanout)
            return list(np.random.choice(neighbors, size=fanout, replace=False, p=weights))

        elif policy == DisseminationPolicy.BUBBLE_PROXIMITY:
            weights = []
            for n_id in neighbors:
                edge = node.edges[n_id]
                w = max(0.01, edge.weight_w * edge.semantic_affinity)
                weights.append(w)
            weights = np.array(weights) / sum(weights)
            fanout = min(len(neighbors), self.base_fanout)
            return list(np.random.choice(neighbors, size=fanout, replace=False, p=weights))

        elif policy == DisseminationPolicy.DUAL_REGULATED:
            f_dyn = node.compute_dynamic_fanout(self.base_fanout, self.theta_reg, self.kappa)
            fanout = min(len(neighbors), f_dyn)

            weights = []
            for n_id in neighbors:
                nbr = self.nodes[n_id]
                edge = node.edges[n_id]
                safety_mult = max(0.05, 1.0 - max(0.0, nbr.mean_arousal - self.theta_reg) / (1.0 - self.theta_reg + 1e-6))
                w = max(0.01, edge.weight_w * edge.semantic_affinity * safety_mult)
                weights.append(w)
            weights = np.array(weights) / sum(weights)
            return list(np.random.choice(neighbors, size=fanout, replace=False, p=weights))

        return random.sample(neighbors, min(len(neighbors), self.base_fanout))

    def run_simulation(self, policy: DisseminationPolicy, total_steps: int = 50, panic_step: int = 10) -> Dict:
        self._build_topology()

        history_arousal = []
        history_cascade = []
        history_coherence = []
        total_messages = 0

        pending_messages: List[Tuple[int, int, TransientGossipToken]] = []
        seed_nodes = [12, 13, 14, 15]

        for t in range(total_steps):
            if t == panic_step:
                tokens = self.inject_panic_cascade(seed_nodes, current_step=t)
                for pt in tokens:
                    targets = self.select_targets(self.nodes[pt.origin_node], pt, policy)
                    for tgt in targets:
                        pending_messages.append((pt.origin_node, tgt, pt))
                        total_messages += 1

            if t % 2 == 0:
                for _ in range(2):
                    sender = random.randint(0, self.node_count - 1)
                    ambient_tok = TransientGossipToken(
                        token_id=f"amb_{t}_{sender}",
                        origin_node=sender,
                        semantic_embedding=np.random.uniform(-0.5, 0.5, size=4),
                        affective_tone=AffectiveState(valence=0.30, arousal=0.20, stress=0.15),
                        current_intensity=0.85,
                        state=GossipState.ACTIVE,
                        issuance_time=t,
                        remaining_ttl=5
                    )
                    targets = self.select_targets(self.nodes[sender], ambient_tok, policy)
                    for tgt in targets:
                        pending_messages.append((sender, tgt, ambient_tok))
                        total_messages += 1

            for node in self.nodes.values():
                node.natural_thermodynamic_decay(self.lambda_decay, baseline_arousal=0.20)

            next_pending: List[Tuple[int, int, TransientGossipToken]] = []
            for src_id, dst_id, token in pending_messages:
                dst = self.nodes[dst_id]
                dst.messages_received += 1

                if token.is_expired():
                    continue

                tok_copy = TransientGossipToken(
                    token_id=token.token_id,
                    origin_node=token.origin_node,
                    semantic_embedding=token.semantic_embedding.copy(),
                    affective_tone=token.affective_tone.copy(),
                    current_intensity=token.current_intensity,
                    state=token.state,
                    issuance_time=token.issuance_time,
                    remaining_ttl=token.remaining_ttl,
                    hops=token.hops
                )

                if policy == DisseminationPolicy.CLASSIC_UNREGULATED:
                    dst.update_state_on_message(tok_copy, self.alpha, self.beta)
                    tok_copy.remaining_ttl -= 1
                    tok_copy.hops += 1
                    if tok_copy.remaining_ttl <= 0:
                        tok_copy.state = GossipState.EXPIRED

                elif policy == DisseminationPolicy.EMOTION_AWARE:
                    dst.update_state_on_message(tok_copy, self.alpha, self.beta)
                    tok_copy.current_intensity *= max(0.1, 1.0 - 0.25 * dst.mean_arousal)
                    tok_copy.remaining_ttl -= 1
                    tok_copy.hops += 1
                    if tok_copy.current_intensity < 0.35:
                        tok_copy.state = GossipState.DAMPENED
                    if tok_copy.remaining_ttl <= 0 or tok_copy.current_intensity <= 0.05:
                        tok_copy.state = GossipState.EXPIRED

                elif policy == DisseminationPolicy.BUBBLE_PROXIMITY:
                    dst.update_state_on_message(tok_copy, self.alpha, self.beta)
                    edge = self.nodes[src_id].edges.get(dst_id)
                    aff = edge.semantic_affinity if edge else 0.5
                    tok_copy.current_intensity *= max(0.1, aff)
                    tok_copy.remaining_ttl -= 1
                    tok_copy.hops += 1
                    if tok_copy.current_intensity < 0.35:
                        tok_copy.state = GossipState.DAMPENED
                    if tok_copy.remaining_ttl <= 0 or tok_copy.current_intensity <= 0.05:
                        tok_copy.state = GossipState.EXPIRED

                elif policy == DisseminationPolicy.DUAL_REGULATED:
                    dst.apply_thermodynamic_counter_signal(tok_copy, delta=0.45, lambda_decay=self.lambda_decay)
                    dst.update_state_on_message(tok_copy, self.alpha, self.beta)
                    dst.damp_token(
                        tok_copy,
                        lambda_decay=self.lambda_decay,
                        mu1=self.mu1,
                        mu2=self.mu2,
                        theta_reg=self.theta_reg,
                        tau_damp=0.35
                    )

                if not tok_copy.is_expired() and tok_copy.token_id not in dst.seen_tokens:
                    dst.seen_tokens.add(tok_copy.token_id)
                    forward_targets = self.select_targets(dst, tok_copy, policy)
                    for nxt in forward_targets:
                        if nxt != src_id:
                            next_pending.append((dst_id, nxt, tok_copy))
                            total_messages += 1

            for node in self.nodes.values():
                node.update_transition_edges(self.nodes, self.omega, self.gamma1, self.gamma2)

            pending_messages = next_pending

            arousals = [n.mean_arousal for n in self.nodes.values()]
            valences = [n.current_affect.valence for n in self.nodes.values()]
            mean_a = float(np.mean(arousals))
            cascaded = sum(1 for a in arousals if a >= 0.38)
            frac = float(cascaded) / self.node_count

            var_a = float(np.var(arousals))
            var_v = float(np.var(valences))
            coherence = float(1.0 / (1.0 + 4.0 * (var_a + var_v)))

            history_arousal.append(mean_a)
            history_cascade.append(frac)
            history_coherence.append(coherence)

        peak_arousal = max(history_arousal[panic_step:])
        peak_cascade = max(history_cascade[panic_step:])

        peak_idx = panic_step + int(np.argmax(history_arousal[panic_step:]))
        recovery_steps = "Unrecovered"
        for s in range(peak_idx, total_steps):
            if history_arousal[s] <= 0.24:
                recovery_steps = f"{s - panic_step} steps"
                break

        mean_coherence = float(np.mean(history_coherence[panic_step:]))

        verdicts = {
            DisseminationPolicy.CLASSIC_UNREGULATED: "Severe Contagion / Stall",
            DisseminationPolicy.EMOTION_AWARE: "Partial Mitigation",
            DisseminationPolicy.BUBBLE_PROXIMITY: "Localized Containment",
            DisseminationPolicy.DUAL_REGULATED: "Homeostatic Equilibrium"
        }

        return {
            "policy": policy.value,
            "peak_mean_arousal": round(peak_arousal, 3),
            "peak_cascade_fraction": round(peak_cascade, 3),
            "recovery_steps": recovery_steps,
            "affective_coherence": round(mean_coherence, 3),
            "cumulative_messages": total_messages,
            "stability_verdict": verdicts[policy],
            "history_arousal": history_arousal,
            "history_cascade": history_cascade,
            "history_coherence": history_coherence
        }


def run_benchmark():
    policies = [
        DisseminationPolicy.CLASSIC_UNREGULATED,
        DisseminationPolicy.EMOTION_AWARE,
        DisseminationPolicy.BUBBLE_PROXIMITY,
        DisseminationPolicy.DUAL_REGULATED
    ]

    print("=" * 115)
    print("EMOTIONALLY AND CONTEXT-AWARE GOSSIP PROTOCOL (ECAGP / BUBBLES) BENCHMARK v1.2")
    print("Setup: N = 80 Nodes, 50 Discrete Steps, Acute Panic Stimulus at t = 10 (Seeds: [12, 13, 14, 15])")
    print("=" * 115)

    results = []
    for p in policies:
        sim = ECAGPSimulator(node_count=80, base_fanout=4, seed=42)
        res = sim.run_simulation(policy=p, total_steps=50, panic_step=10)
        results.append(res)

    header = f"{'Dissemination Policy':<30} | {'Peak Arousal':<12} | {'Peak Cascade':<12} | {'Recovery':<14} | {'Coherence':<10} | {'Messages':<10} | {'Stability Verdict'}"
    print(header)
    print("-" * 115)
    for r in results:
        print(f"{r['policy']:<30} | {r['peak_mean_arousal']:<12} | {r['peak_cascade_fraction']:<12} | {r['recovery_steps']:<14} | {r['affective_coherence']:<10} | {r['cumulative_messages']:<10} | {r['stability_verdict']}")
    print("-" * 115)
    return results


if __name__ == "__main__":
    run_benchmark()

