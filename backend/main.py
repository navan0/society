# main.py
import math
import random
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Config
# -----------------------
N_AGENTS = 28

D_CORE = 40     # slow personality/values/etc.   (first 4 dims are interpretable)
D_STATE = 8     # fast mood/attention/etc.

EDGE_DECAY = 0.92
EVENT_DECAY = 0.93          # more persistent shocks
STATE_DECAY = 0.05
HOMEOSTASIS = 0.10
PLASTICITY_SCALE = 0.15
SENSITIVITY_SCALE = 0.25
CORE_DRIFT_SCALE = 0.006     # stronger slow core drift
STATE_NOISE = 0.028          # a bit more energy

# interactions per step (~90% of agents engage)
INTERACTIONS_PER_STEP = max(1, int(0.9 * N_AGENTS))

# how much fast state influences plotted coords (x≈optimism, y≈sociability)
STATE_VIS_GAIN = 0.6

# core movement driven by interaction type (chat attract, conflict repel)
CORE_ACTION_GAIN = 0.2

# exploration chance in partner choice (break echo chambers)
EXPLORE_EPS = 0.20

def t(x): return torch.tensor(x, dtype=torch.float32)


# -----------------------
# Models & Data
# -----------------------
class Agent:
    """
    Two-speed embedding:
      z_core  (D_CORE): slow personality/values/role/interests...
      z_state (D_STATE): fast state (mood, attention, etc.)
    Interpretable core dims (for UI):
      0: optimism, 1: sociability, 2: risk, 3: consumerism
    """
    def __init__(self, id_: int):
        self.id = id_
        # core init: small variance around 0
        self.z_core = torch.randn(D_CORE) * 0.2
        # emphasize interpretable spread a bit so the plot isn't flat
        self.z_core[:4] += t([random.uniform(-1.2, 1.2),
                              random.uniform(-1.0, 1.0),
                              random.uniform(-1.0, 1.2),
                              random.uniform(-1.0, 1.0)])

        # state starts at zero
        self.z_state = torch.zeros(D_STATE)
        self.mu_state = torch.zeros(D_STATE)  # personal baseline

        # heterogeneous per-dim dynamics
        self.alpha = torch.rand(D_STATE) * PLASTICITY_SCALE   # local influence
        self.beta = torch.rand(D_STATE) * SENSITIVITY_SCALE   # event sensitivity
        self.delta = torch.rand(D_CORE) * CORE_DRIFT_SCALE    # core drift rate

        # decaying memory for global events (impulse response)
        self.event_buf = torch.zeros(D_STATE)

        # how much state affects behavior (used in core action gain too)
        self.state_gate = 0.5 + 0.2 * torch.rand(1)  # ~0.5..0.7

    # ---- policy: simple probabilistic rule (no training) ----
    def act(self, partner_core: torch.Tensor, edge_w: float) -> int:
        """
        Returns one of {0:chat, 1:trade, 2:ignore, 3:conflict}
        using a simple mapping from similarity & consumerism.
        """
        sim = F.cosine_similarity(self.z_core, partner_core, dim=0).clamp(-1, 1).item()
        cons_self = float(torch.tanh(self.z_core[3]))      # consumerism
        opt_partner = float(torch.tanh(partner_core[0]))   # partner optimism
        edge_term = min(1.0, math.log1p(edge_w + 1e-6))    # repeat ties

        s_chat = 0.9 * sim + 0.2 * opt_partner + 0.3 * edge_term
        s_conf = -0.9 * sim + 0.2 * (1 - edge_term)
        s_trade = 0.4 * (1 - abs(sim)) + 0.5 * max(0.0, cons_self) + 0.2 * edge_term
        s_ign = 0.2 - 0.2 * edge_term

        logits = torch.tensor([s_chat, s_trade, s_ign, s_conf], dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return int(action)

    # ---- dynamics update per interaction ----
    def step_update(self, partner: "Agent", action: int, edge_w: float):
        """
        Stable two-speed update (trust-weighted, homeostatic, with decaying event impulses).
        """
        # decay event buffer each step
        self.event_buf.mul_(EVENT_DECAY)

        # similarity-based trust weighting
        sim = F.cosine_similarity(self.z_core, partner.z_core, dim=0).clamp(-1, 1)
        w = torch.sigmoid(1.2 * sim + 0.8 * min(2.0, math.log1p(edge_w + 1e-6)))  # ~0..1

        # action masks in STATE space: [mood, social_drive, risk_state, attention, consumer_state, fatigue, arousal, focus]
        mask_chat     = t([1, 1, 1, 0, 0, 0, 1, 0])
        mask_trade    = t([0, 0, 0, 1, 1, 0, 0, 1])
        mask_ignore   = t([0, 0, 0, 0, 0, 0, 0, 0])
        mask_conflict = t([1, 0, 1, 0, 0, 1, 1, 0])
        M = {0: mask_chat, 1: mask_trade, 2: mask_ignore, 3: mask_conflict}[action]

        # state dynamics
        event_shock = self.event_buf.clone()
        self.z_state = (1 - STATE_DECAY) * self.z_state
        self.z_state = self.z_state + (self.alpha * w) * (partner.z_state - self.z_state) * M
        self.z_state = self.z_state + self.beta * event_shock
        self.z_state = self.z_state + HOMEOSTASIS * (self.mu_state - self.z_state)
        self.z_state = self.z_state + STATE_NOISE * torch.randn_like(self.z_state)
        self.z_state.clamp_(-3.0, 3.0)

        # action-driven core motion (chat attract, conflict repel; trade mild; ignore none)
        core_mask = torch.zeros(D_CORE)
        core_mask[:16] = 1.0  # personality/values slice
        sign = {0: 1.0, 1: 0.3, 2: 0.0, 3: -1.0}[action]
        self.z_core = self.z_core + (self.delta * core_mask) * sign * self.state_gate * w * CORE_ACTION_GAIN * (partner.z_core - self.z_core)
        self.z_core.clamp_(-4.0, 4.0)

    def apply_event(self, ev_vec: torch.Tensor):
        """Add an impulse to the agent's event buffer (decays across steps)."""
        self.event_buf = self.event_buf + ev_vec

    def short_embedding(self) -> List[float]:
        """Return the 4 interpretable core dims for UI: [optimism, sociability, risk, consumerism]"""
        return [float(self.z_core[i].item()) for i in range(4)]


class Society:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.agents: List[Agent] = [Agent(i) for i in range(n_agents)]
        self.edges: Dict[Tuple[int, int], float] = {}  # undirected edge weights
        self.action_counts = {"chat": 0, "trade": 0, "ignore": 0, "conflict": 0}
        self.event_history: List[str] = []
        self.last: Optional[Dict] = None

        # homophily & exploration (avoid echo chambers)
        self.homophily = 1.0
        self.explore_eps = EXPLORE_EPS

        # planner state
        self.strategy = "homophily"   # 'homophily' | 'round_robin' | 'nearest' | 'tension'
        self.rr_ptr = 0               # round-robin pointer

    def reset(self):
        self.__init__(self.n_agents)

    def _edge_key(self, i: int, j: int) -> Tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _edge_get(self, i: int, j: int) -> float:
        return self.edges.get(self._edge_key(i, j), 0.0)

    def _edge_bump(self, i: int, j: int, amount: float):
        k = self._edge_key(i, j)
        self.edges[k] = self.edges.get(k, 0.0) + amount

    def _decay_edges(self):
        if not self.edges:
            return
        to_del = []
        for k, w in self.edges.items():
            w_new = w * EDGE_DECAY
            if w_new < 0.02:
                to_del.append(k)
            else:
                self.edges[k] = w_new
        for k in to_del:
            del self.edges[k]

    def _choose_partner(self, i: int) -> int:
        """
        Sample partner j for agent i with soft preference toward similar cores.
        ε-greedy exploration to break echo chambers.
        """
        if random.random() < self.explore_eps:
            js = [j for j in range(self.n_agents) if j != i]
            return random.choice(js)

        zi = self.agents[i].z_core
        weights = []
        ids = []
        for j in range(self.n_agents):
            if j == i:
                continue
            zj = self.agents[j].z_core
            sim = float(F.cosine_similarity(zi, zj, dim=0).clamp(-1, 1))
            w = math.exp(self.homophily * sim)
            ids.append(j)
            weights.append(w)
        s = sum(weights)
        if s <= 0:
            return random.choice(ids)
        r = random.random() * s
        c = 0.0
        for j, w in zip(ids, weights):
            c += w
            if r <= c:
                return j
        return ids[-1]

    # ---------- Pair planning ----------
    def _pair_count(self) -> int:
        # each pair consumes 2 agents; keep close to INTERACTIONS_PER_STEP
        return min(max(1, INTERACTIONS_PER_STEP), self.n_agents // 2)

    def _plan_pairs(self) -> List[Tuple[int, int]]:
        if self.strategy == "homophily":
            pairs = []
            for _ in range(INTERACTIONS_PER_STEP):
                i = random.randrange(self.n_agents)
                j = self._choose_partner(i)
                pairs.append((i, j))
            return pairs

        if self.strategy == "round_robin":
            n = self.n_agents
            k = self._pair_count()
            pairs = []
            start = self.rr_ptr
            used: Set[int] = set()
            for _ in range(k):
                a = start % n
                b = (start + 1) % n
                # skip used in this tick
                while (a in used or b in used or a == b):
                    start = (start + 2) % n
                    a = start % n
                    b = (start + 1) % n
                pairs.append((a, b))
                used.add(a); used.add(b)
                start = (start + 2) % n
            self.rr_ptr = start % n
            return pairs

        # Build similarity & edge matrices (small N => fine)
        Z = torch.stack([ag.z_core for ag in self.agents], dim=0)   # (N, D)
        Zn = F.normalize(Z, dim=1)
        sim = torch.matmul(Zn, Zn.T).clamp(-1, 1).cpu().numpy()     # (N,N)
        edge_w = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for (a, b), w in self.edges.items():
            edge_w[a, b] = edge_w[b, a] = float(w)
        if edge_w.size == 0:
            edge_w_norm = np.zeros_like(sim)
        else:
            mx = edge_w.max()
            edge_w_norm = np.log1p(edge_w) / max(1.0, np.log1p(mx if mx > 0 else 1.0))

        # arousal (state dim 6), attention (dim 3) as examples
        ar = np.array([float(ag.z_state[6]) for ag in self.agents], dtype=np.float32)
        att = np.array([float(ag.z_state[3]) for ag in self.agents], dtype=np.float32)

        N = self.n_agents
        used = np.zeros(N, dtype=bool)
        k = self._pair_count()
        pairs: List[Tuple[int, int]] = []

        def greedy_from_score(score_mat: np.ndarray):
            nonlocal pairs, used
            for _ in range(k):
                # mask used
                m = np.logical_not(used)
                cm = np.outer(m, m).astype(np.bool_)
                np.fill_diagonal(cm, False)
                if not cm.any():
                    break
                masked = score_mat.copy()
                masked[~cm] = -1e9
                i, j = np.unravel_index(np.argmax(masked), masked.shape)
                if masked[i, j] < -1e8:
                    break
                pairs.append((int(i), int(j)))
                used[i] = True; used[j] = True

        if self.strategy == "nearest":
            # prefer high similarity & existing ties; small anti-bias for repeated pairs
            score = 0.8 * sim + 0.2 * edge_w_norm - 0.05 * (1.0 - edge_w_norm)
            greedy_from_score(score)
            return pairs

        if self.strategy == "tension":
            # prefer dissimilar (1-sim), low existing tie, and high combined arousal/attention
            diff = 1.0 - sim
            ar_pair = (ar[:, None] + ar[None, :]) / 2.0
            att_pair = (att[:, None] + att[None, :]) / 2.0
            score = 0.7 * diff + 0.25 * (ar_pair + 0.5 * att_pair) + 0.2 * (1.0 - edge_w_norm)
            np.fill_diagonal(score, -1e9)
            greedy_from_score(score)
            return pairs

        # fallback
        return [(random.randrange(self.n_agents), self._choose_partner(random.randrange(self.n_agents)))
                for _ in range(INTERACTIONS_PER_STEP)]

    # ---------- Step ----------
    def step(self):
        """One simulation step."""
        self._decay_edges()

        pairs = self._plan_pairs()
        for (i, j) in pairs:
            ai, aj = self.agents[i], self.agents[j]
            w_ij = self._edge_get(i, j)
            action = ai.act(aj.z_core, w_ij)
            # symmetric update (both feel the interaction)
            ai.step_update(aj, action, w_ij)
            aj.step_update(ai, action, w_ij)

            # edge reinforcement by action
            if action == 0:       # chat
                self._edge_bump(i, j, 1.0)
                self.action_counts["chat"] += 1
                act_str = "chat"
            elif action == 1:     # trade
                self._edge_bump(i, j, 0.6)
                self.action_counts["trade"] += 1
                act_str = "trade"
            elif action == 2:     # ignore
                self._edge_bump(i, j, 0.0)
                self.action_counts["ignore"] += 1
                act_str = "ignore"
            else:                 # conflict
                self._edge_bump(i, j, 0.3)
                self.action_counts["conflict"] += 1
                act_str = "conflict"

            self.last = {"a": i, "b": j, "action": act_str}

        # prevent total collapse: inject tiny diversity when over-cohesive
        metrics_now = self._compute_metrics()
        if metrics_now["cohesion"] > 0.60:
            for ag in self.agents:
                ag.z_core[:2] += 0.03 * torch.randn(2)
                ag.z_core.clamp_(-4, 4)

        return self.snapshot()

    def apply_event(self, ev_type: str):
        """
        Global event as decaying impulse on STATE dims:
        [mood, social_drive, risk_state, attention, consumer_state, fatigue, arousal, focus]
        """
        if ev_type == "good_news":
            v = t([+0.6, +0.05, -0.05, +0.1, +0.05, -0.05, +0.2, +0.05])
        elif ev_type == "market_crash":
            v = t([-0.6, -0.05, +0.4, +0.05, -0.1, +0.1, +0.2, -0.05])
        elif ev_type == "ad_blitz":
            v = t([+0.05, +0.1, +0.05, +0.35, +0.5, +0.05, +0.1, +0.15])
        elif ev_type == "lockdown":
            v = t([-0.15, -0.6, +0.1, -0.05, -0.2, +0.25, -0.15, -0.2])
        else:
            raise ValueError("Unknown event type")

        for ag in self.agents:
            ag.apply_event(v)
        self.event_history.append(ev_type)
        return self.snapshot()

    # ---- metrics & serialization ----
    def snapshot(self) -> Dict:
        # include some fast state on axes so motion is visible
        agents4 = []
        for ag in self.agents:
            core = ag.short_embedding()  # [optimism, sociability, risk, consumerism]
            e0 = float(ag.z_state[0].item())  # mood
            e1 = float(ag.z_state[1].item())  # social_drive
            emb = [core[0] + STATE_VIS_GAIN * e0,
                   core[1] + STATE_VIS_GAIN * e1,
                   core[2],
                   core[3]]
            agents4.append(emb)

        edges_list = [{"a": a, "b": b, "w": float(w)} for (a, b), w in self.edges.items()]
        metrics = self._compute_metrics()

        return {
            "agents": [{"id": i, "embedding": emb} for i, emb in enumerate(agents4)],
            "edges": edges_list,
            "last": self.last,
            "metrics": metrics,
            "events": self.event_history[-100:],
        }

    def _compute_metrics(self) -> Dict:
        if self.n_agents == 0:
            return {}
        # use core first 8 dims for similarity metrics
        Z = torch.stack([ag.z_core[:8] for ag in self.agents], dim=0)  # (N, 8)
        Zn = F.normalize(Z, dim=1)
        sim_mat = torch.matmul(Zn, Zn.T)  # (N,N)

        N = self.n_agents
        mask = ~torch.eye(N, dtype=torch.bool)
        mean_sim = float(sim_mat[mask].mean().item())
        cohesion = max(-1.0, min(1.0, mean_sim))
        polarization = float(1.0 - max(-1.0, min(1.0, cohesion)))

        # PC1 spread (variance explained by first principal component)
        Z_np = Z.detach().cpu().numpy()
        Z_np = Z_np - Z_np.mean(axis=0, keepdims=True)
        cov = np.cov(Z_np.T)
        evals, _ = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        spread_pc1 = float(evals[-1] / evals.sum()) if evals.sum() > 1e-12 else 0.0

        mean_edge_w = float(np.mean([w for w in self.edges.values()])) if self.edges else 0.0

        traits = np.array([[float(ag.z_core[0]), float(ag.z_core[1]), float(ag.z_core[2]), float(ag.z_core[3])]
                           for ag in self.agents])
        means = traits.mean(axis=0)
        stds = traits.std(axis=0)

        return {
            "cohesion": cohesion,
            "polarization": polarization,
            "spread_pc1": spread_pc1,
            "mean_edge_weight": mean_edge_w,
            "trait_means": {
                "optimism": float(means[0]),
                "sociability": float(means[1]),
                "risk": float(means[2]),
                "consumerism": float(means[3]),
            },
            "trait_stds": {
                "optimism": float(stds[0]),
                "sociability": float(stds[1]),
                "risk": float(stds[2]),
                "consumerism": float(stds[3]),
            },
            "action_counts": self.action_counts.copy(),
        }


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

society = Society(N_AGENTS)


# Schemas
class PutTraits(BaseModel):
    traits: Dict[str, float]  # {"optimism": ..., "sociability": ..., "risk": ..., "consumerism": ...}

class PostEvent(BaseModel):
    type: str  # "good_news" | "market_crash" | "ad_blitz" | "lockdown"

class PlannerBody(BaseModel):
    strategy: str  # "homophily" | "round_robin" | "nearest" | "tension"
    explore_eps: Optional[float] = None  # optional: tweak epsilon


@app.get("/state")
def get_state():
    return society.snapshot()

@app.post("/step")
def post_step():
    return society.step()

@app.post("/reset")
def post_reset():
    global society
    society = Society(N_AGENTS)
    return {"ok": True, "state": society.snapshot()}

@app.put("/agents/{agent_id}")
def put_agent_traits(agent_id: int, body: PutTraits):
    if agent_id < 0 or agent_id >= society.n_agents:
        raise HTTPException(status_code=404, detail="agent not found")
    ag = society.agents[agent_id]
    mapping = {"optimism": 0, "sociability": 1, "risk": 2, "consumerism": 3}
    for k, idx in mapping.items():
        if k in body.traits:
            ag.z_core[idx] = float(body.traits[k])
    ag.z_core.clamp_(-4.0, 4.0)
    return {"ok": True, "state": society.snapshot()}

@app.post("/event")
def post_event(ev: PostEvent):
    try:
        snap = society.apply_event(ev.type)
    except ValueError:
        raise HTTPException(status_code=400, detail="unknown event type")
    return snap

@app.post("/planner")
def set_planner(body: PlannerBody):
    s = body.strategy.lower()
    if s not in {"homophily", "round_robin", "nearest", "tension"}:
        raise HTTPException(status_code=400, detail="unknown planner")
    society.strategy = s
    if body.explore_eps is not None:
        society.explore_eps = float(max(0.0, min(1.0, body.explore_eps)))
    return {"ok": True, "strategy": society.strategy, "explore_eps": society.explore_eps}


# local run: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
