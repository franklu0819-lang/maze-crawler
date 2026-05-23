"""PPO training — v20: enhanced exploration + mixed eval (50% self-best + 25% v50 best + 25% v49 best).

Changes from v19:
- ENTROPY_COEF: 0.02 → 0.05 (more exploration)
- PPO_CLIP: 0.2 → 0.1 (conservative updates)
- EVAL_GAMES: 100 → 200
- Eval opponents: 50% self-best + 25% v50 best + 25% v49 best
- Training: 100% self-play, unchanged
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS,
    BIT_N, BIT_E, BIT_S, BIT_W,
)

# ─── Constants ───────────────────────────────────────────────────────

GRID_R = 2
WALL_CH = 5
NUM_SCALARS = 24
INPUT_SIZE = (2*GRID_R+1)**2 * WALL_CH + NUM_SCALARS  # 125+24=149

ACTION_STRINGS = [
    "NORTH", "EAST", "WEST", "SOUTH",                # 0-3: move
    "JUMP_NORTH",                                     # 4
    "REMOVE_NORTH", "REMOVE_EAST", "REMOVE_WEST",    # 5-7: worker
    "BUILD_WORKER", "BUILD_SCOUT", "BUILD_MINER",     # 8-10: factory
    "TRANSFORM",                                       # 11: miner
    "IDLE",                                            # 12
]
NUM_ACTIONS = len(ACTION_STRINGS)

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.1
PPO_EPOCHS = 4
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
DELTA_GAP_W = 0.5      # gap signal weight
DELTA_UNITS_W = 0.1    # unit count change weight
UNIT_SURVIVAL = 0.01   # non-factory unit survival bonus per step
SHAPING_REMOVE = 0.1   # REMOVE shaping
SHAPING_TRANSFORM = 0.5 # TRANSFORM shaping
EVAL_EVERY = 10         # Evaluate every N iterations
EVAL_GAMES = 200        # Games per evaluation

# ─── Eval Opponents ─────────────────────────────────────────────────

V49_WEIGHTS = "nn_weights_v49.pt"
V50_WEIGHTS = "nn_weights_v50.pt"


# ─── Feature Extraction ──────────────────────────────────────────────

def extract_unit(obs, config, my_player, occupied, reserved, actions, uid, data):
    """Extract features and action mask for any unit type."""
    utype = data[0]
    c, r, energy = data[1], data[2], data[3]
    move_cd = data[5] if len(data) > 5 else 0
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    gap = r - obs.southBound
    w = config.width
    turn = STATE["turn"]
    stuck = STATE.get("factory_stuck", 0)

    # 5x5 wall grid centered on unit
    grid = np.zeros((5, 5, 5), dtype=np.float32)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nc, nr2 = c + dc, r + dr
            idx = (nr2 - obs.southBound) * w + nc
            if (0 <= nc < w and obs.southBound <= nr2 <= obs.northBound
                    and 0 <= idx < len(obs.walls)):
                v = obs.walls[idx]
                if v != -1:
                    grid[dr+2, dc+2] = [
                        float(bool(v & 1)), float(bool(v & 2)),
                        float(bool(v & 4)), float(bool(v & 8)), 1.0,
                    ]
    wall_flat = grid.flatten()

    # Unit counts
    sc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_SCOUT)
    wc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_WORKER)
    mc = sum(1 for d in obs.robots.values() if d[4] == my_player and d[0] == TYPE_MINER)
    has_nodes = float(bool(getattr(obs, "miningNodes", {})))

    # Factory info
    factory_pos = None
    for uid2, d2 in obs.robots.items():
        if d2[4] == my_player and d2[0] == TYPE_FACTORY:
            factory_pos = (d2[1], d2[2])
            break

    # Non-factory: distance/direction to factory
    dist_factory = 0.0
    at_factory_spawn = 0.0
    dir_factory = [0.0, 0.0, 0.0, 0.0]
    if factory_pos and utype != TYPE_FACTORY:
        fc, fr = factory_pos
        dist_factory = (abs(c - fc) + abs(r - fr)) / 20.0
        at_factory_spawn = float(c == fc and r == fr + 1)
        if fr > r: dir_factory[0] = 1.0
        if fc > c: dir_factory[1] = 1.0
        if fc < c: dir_factory[2] = 1.0
        if fr < r: dir_factory[3] = 1.0

    # Mining node check
    visible_nodes = set(parse_key(k) for k in (getattr(obs, "miningNodes", {}) or {}))
    is_on_node = float((c, r) in visible_nodes)

    # Nearest enemy
    nearest_enemy = 20.0
    for uid2, d2 in obs.robots.items():
        if d2[4] != my_player:
            dist = abs(d2[1] - c) + abs(d2[2] - r)
            if dist < nearest_enemy:
                nearest_enemy = dist
    nearest_enemy /= 10.0

    scalars = np.array([
        float(utype == TYPE_FACTORY), float(utype == TYPE_SCOUT),
        float(utype == TYPE_WORKER), float(utype == TYPE_MINER),
        gap / 20.0, energy / 1000.0, move_cd / 5.0,
        c / max(1, w - 1), turn / 500.0,
        sc / 3.0, wc / 2.0, mc / 2.0,
        has_nodes, stuck / 10.0,
        jump_cd / 20.0, build_cd / 10.0,
        dist_factory, at_factory_spawn,
        *dir_factory, nearest_enemy, is_on_node,
    ], dtype=np.float32)

    features = np.concatenate([wall_flat, scalars])

    # ── Action mask ──
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # MOVE (0-3): all types
    if move_cd == 0:
        for i, d in enumerate(["NORTH", "EAST", "WEST", "SOUTH"]):
            if not can_go(obs, config, c, r, d):
                continue
            dc2, dr2, _ = DIRS[d]
            nxt = (c + dc2, r + dr2)
            if nxt in reserved:
                continue
            if utype == TYPE_FACTORY:
                occ = occupied.get(nxt, [])
                friendlies = [o for o in occ if o[1][4] == my_player and o[1][0] != TYPE_FACTORY]
                if friendlies:
                    all_moving = all(o[0] in actions and actions[o[0]] in DIRS for o in friendlies)
                    if not all_moving:
                        continue
                mask[i] = 1.0
            else:
                if not friendly_at(occupied, nxt, my_player):
                    mask[i] = 1.0

    # JUMP_NORTH (4): factory only
    if utype == TYPE_FACTORY and jump_cd == 0 and turn > 2:
        if in_bounds(c, r + 2, obs, config):
            mask[4] = 1.0

    # REMOVE (5-7): worker only
    if utype == TYPE_WORKER:
        wall_cost = getattr(config, "wallRemoveCost", 100)
        if energy >= wall_cost:
            wv = wb(obs, config, c, r)
            if wv is not None:
                if wv & BIT_N: mask[5] = 1.0
                if wv & BIT_E: mask[6] = 1.0
                if wv & BIT_W: mask[7] = 1.0

    # BUILD (8-10): factory only
    if utype == TYPE_FACTORY and move_cd != 0 and build_cd == 0:
        s_ok = can_go(obs, config, c, r, "NORTH") and in_bounds(c, r + 1, obs, config)
        if s_ok:
            spawn = (c, r + 1)
            if not friendly_at(occupied, spawn, my_player):
                if energy >= getattr(config, "workerCost", 200): mask[8] = 1.0
                if energy >= getattr(config, "scoutCost", 50): mask[9] = 1.0
                if has_nodes and energy >= getattr(config, "minerCost", 300): mask[10] = 1.0

    # TRANSFORM (11): miner only
    if utype == TYPE_MINER:
        transform_cost = getattr(config, "transformCost", 100)
        if (c, r) in visible_nodes and energy >= transform_cost + 1:
            mask[11] = 1.0

    # IDLE (12): always valid
    mask[12] = 1.0
    if mask.sum() == 0:
        mask[12] = 1.0

    return features, mask


# ─── Action Execution ────────────────────────────────────────────────

def execute_action(uid, data, action_idx, actions_dict, reserved):
    c, r = data[1], data[2]
    a_str = ACTION_STRINGS[action_idx]
    actions_dict[uid] = a_str

    if action_idx <= 3:  # MOVE
        dc, dr, _ = DIRS[a_str]
        reserved.add((c + dc, r + dr))
    elif action_idx == 4:  # JUMP_NORTH
        reserved.add((c, r + 2))
    elif 5 <= action_idx <= 7:  # REMOVE
        reserved.add((c, r))
    elif 8 <= action_idx <= 10:  # BUILD
        reserved.add((c, r + 1))
    elif action_idx == 11:  # TRANSFORM
        reserved.add((c, r))
    else:  # IDLE
        reserved.add((c, r))


# ─── Actor-Critic Network ────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, NUM_ACTIONS)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        h = self.backbone(x)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        probs = torch.softmax(logits, dim=-1)
        value = self.value_head(h).squeeze(-1)
        return probs, value

    def get_action(self, x, mask=None):
        probs, value = self.forward(x, mask)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ─── PPO Game Runner ─────────────────────────────────────────────────

def _unit_count(robots, player):
    return sum(1 for d in robots.values() if d[4] == player and d[0] != TYPE_FACTORY)


_INITIAL_STATE = {"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}}


def _swap_state(player_states, player_id):
    saved = (STATE["turn"], STATE["nodes"], STATE["last_factory_pos"],
             STATE["factory_stuck"], STATE["walls"])
    ps = player_states[player_id]
    STATE["turn"] = ps["turn"]
    STATE["nodes"] = ps["nodes"]
    STATE["last_factory_pos"] = ps["last_factory_pos"]
    STATE["factory_stuck"] = ps["factory_stuck"]
    STATE["walls"] = ps["walls"]
    return saved


def _restore_state(player_states, player_id, saved):
    ps = player_states[player_id]
    ps["turn"] = STATE["turn"]
    ps["nodes"] = STATE["nodes"]
    ps["last_factory_pos"] = STATE["last_factory_pos"]
    ps["factory_stuck"] = STATE["factory_stuck"]
    ps["walls"] = STATE["walls"]
    STATE["turn"], STATE["nodes"], STATE["last_factory_pos"], \
        STATE["factory_stuck"], STATE["walls"] = saved


def run_ppo_game(model, seed, opponent="self", explore=True):
    player_states = {0: dict(_INITIAL_STATE), 1: dict(_INITIAL_STATE)}
    player_states[0]["nodes"] = set()
    player_states[1]["nodes"] = set()

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    unit_trajs = {}
    prev_units = [None]
    prev_gap = [None]
    first_turn = [True]

    def ppo_player(obs, config):
        my_player = obs.player
        saved = _swap_state(player_states, my_player)
        try:
            update_state(obs, config, my_player)

            # Compute reward for player 0 only
            team_r = 0.0
            if my_player == 0:
                cur_units = _unit_count(obs.robots, my_player)
                cur_gap = 0
                for uid2, d2 in obs.robots.items():
                    if d2[4] == my_player and d2[0] == TYPE_FACTORY:
                        cur_gap = d2[2] - obs.southBound
                        break

                if first_turn[0]:
                    prev_units[0] = cur_units
                    prev_gap[0] = cur_gap
                    first_turn[0] = False
                    team_r = 0.0
                else:
                    delta_units = (cur_units - prev_units[0]) * DELTA_UNITS_W
                    delta_gap = (cur_gap - prev_gap[0]) * DELTA_GAP_W
                    team_r = delta_gap + delta_units
                    prev_units[0] = cur_units
                    prev_gap[0] = cur_gap

            actions = {}
            reserved = set()
            occupied = {}
            for uid2, d2 in obs.robots.items():
                occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

            for unit_type in [TYPE_SCOUT, TYPE_WORKER, TYPE_MINER, TYPE_FACTORY]:
                for uid2, d2 in obs.robots.items():
                    if uid2 in actions:
                        continue
                    if d2[4] != my_player or d2[0] != unit_type:
                        continue

                    feat, msk = extract_unit(obs, config, my_player, occupied,
                                              reserved, actions, uid2, d2)
                    if feat is None:
                        continue

                    s = torch.FloatTensor(feat).unsqueeze(0)
                    m = torch.FloatTensor(msk).unsqueeze(0)
                    with torch.no_grad():
                        if my_player == 0 and explore:
                            ai, log_p, val = model.get_action(s, m)
                            ai_item = ai.item()
                            log_p_item = log_p.item()
                            val_item = val.item()
                        else:
                            probs, val = model(s, m)
                            ai_item = torch.argmax(probs).item()
                            log_p_item = 0.0
                            val_item = val.item()

                    # Collect trajectories for player 0 only
                    if my_player == 0:
                        if 5 <= ai_item <= 7:
                            shaping = SHAPING_REMOVE
                        elif ai_item == 11:
                            shaping = SHAPING_TRANSFORM
                        else:
                            shaping = 0.0
                        if d2[0] != TYPE_FACTORY:
                            shaping += UNIT_SURVIVAL
                        step_r = team_r + shaping

                        if uid2 not in unit_trajs:
                            unit_trajs[uid2] = []
                        unit_trajs[uid2].append(
                            (feat.copy(), ai_item, msk.copy(), log_p_item, val_item, step_r))

                    execute_action(uid2, d2, ai_item, actions, reserved)

            return actions
        finally:
            _restore_state(player_states, my_player, saved)

    if opponent == "self":
        env.run([ppo_player, ppo_player])
    else:
        env.run([ppo_player, "random"])

    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return unit_trajs, r0, r1


# ─── GAE ─────────────────────────────────────────────────────────────

def compute_gae(traj, terminal_reward, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(traj)
    if T == 0:
        return [], []

    rewards = [traj[i][5] for i in range(T)]
    values = [traj[i][4] for i in range(T)]
    rewards[-1] += terminal_reward

    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


# ─── PPO Update ──────────────────────────────────────────────────────

def ppo_update(model, optimizer, all_feat, all_act, all_mask,
               all_old_lp, all_adv, all_ret):
    states = torch.FloatTensor(np.array(all_feat))
    actions = torch.LongTensor(all_act)
    masks = torch.FloatTensor(np.array(all_mask))
    old_log_probs = torch.FloatTensor(all_old_lp)
    advantages = torch.FloatTensor(all_adv)
    returns = torch.FloatTensor(all_ret)

    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    probs, values = model(states, masks)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((values - returns) ** 2).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()


# ─── Training Loop ───────────────────────────────────────────────────

def _next_version():
    v = 1
    while os.path.exists(f"nn_weights_v{v}.pt"):
        v += 1
    return v


def train(num_iter=2000, batch=100, lr=0.0003, version=None, terminal_win=5.0):
    if version is None:
        version = _next_version()
    save_path = f"nn_weights_v{version}.pt"
    log_path = f"train_v20_v{version}.log"

    print(f"  train_v20.py — mixed eval: 50% self-best + 25% v50 best + 25% v49 best")
    print(f"  Step reward: delta_gap*{DELTA_GAP_W} + delta_units*{DELTA_UNITS_W}")
    print(f"  Terminal: +{terminal_win} / -1 / 0 (draw)")
    print(f"  Shaping: REMOVE +{SHAPING_REMOVE}, TRANSFORM +{SHAPING_TRANSFORM}, non-factory survival +{UNIT_SURVIVAL}")
    print(f"  Training: 100% self-play | Mixed eval every {EVAL_EVERY} iters ({EVAL_GAMES} games)")
    print(f"  Entropy: {ENTROPY_COEF}")
    print(f"  Iterations: {num_iter} | Batch: {batch}")
    print(f"  Weights -> {save_path}")
    print(f"  Log -> {log_path}")

    # Load v49 best as fixed eval opponent
    v49_model = ActorCritic()
    v49_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), V49_WEIGHTS)
    if not os.path.exists(v49_path):
        print(f"  ERROR: {v49_path} not found!")
        sys.exit(1)
    v49_model.load_state_dict(torch.load(v49_path, map_location="cpu"))
    v49_model.eval()
    print(f"  Loaded v49 eval opponent from {v49_path}")

    # Load v50 best as fixed eval opponent (replaces v6)
    v50_model = ActorCritic()
    v50_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), V50_WEIGHTS)
    if not os.path.exists(v50_path):
        print(f"  ERROR: {v50_path} not found!")
        sys.exit(1)
    v50_model.load_state_dict(torch.load(v50_path, map_location="cpu"))
    v50_model.eval()
    print(f"  Loaded v50 eval opponent from {v50_path}")

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_wr = 0.0
    best_model = None
    t0 = time.time()

    with open(log_path, "w") as log_f:
        log_f.write(f"train_v20.py | eval=50%self+25%v50+25%v49 | iters={num_iter} | batch={batch}\n\n")

        for it in range(num_iter):
            all_feat, all_act, all_mask, all_old_lp = [], [], [], []
            all_adv, all_ret = [], []
            sp_wins = 0

            for _ in range(batch):
                seed = random.randint(0, 999999)
                unit_trajs, r0, r1 = run_ppo_game(model, seed, opponent="self", explore=True)
                if r0 > r1: sp_wins += 1

                terminal_r = terminal_win if r0 > r1 else (-1.0 if r0 < r1 else 0.0)
                for uid, traj in unit_trajs.items():
                    advs, rets = compute_gae(traj, terminal_r)
                    for i, (feat, ai, msk, lp, val, sr) in enumerate(traj):
                        all_feat.append(feat)
                        all_act.append(ai)
                        all_mask.append(msk)
                        all_old_lp.append(lp)
                        all_adv.append(advs[i])
                        all_ret.append(rets[i])

            if not all_adv:
                continue

            for _ in range(PPO_EPOCHS):
                pl, vl, el = ppo_update(model, optimizer, all_feat, all_act,
                                         all_mask, all_old_lp, all_adv, all_ret)

            sp_wr = sp_wins / batch * 100
            elapsed = time.time() - t0

            # ── Periodic mixed eval: 50% self-best, 25% v50 best, 25% v49 best ──
            eval_str = ""
            if (it + 1) % EVAL_EVERY == 0:
                n_self = EVAL_GAMES // 2
                n_v50 = EVAL_GAMES // 4
                n_v49 = EVAL_GAMES - n_self - n_v50
                ev_wins = 0
                ei = 0

                def _make_eval_agent(m0, m1, ps):
                    """Generic eval agent: player 0 uses m0, player 1 uses m1."""
                    def _agent(obs, config):
                        mp = obs.player
                        saved = _swap_state(ps, mp)
                        try:
                            update_state(obs, config, mp)
                            actions = {}
                            reserved = set()
                            occupied = {}
                            for uid2, d2 in obs.robots.items():
                                occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))
                            for uid2, d2 in obs.robots.items():
                                if d2[4] != mp or uid2 in actions:
                                    continue
                                feat, msk = extract_unit(obs, config, mp, occupied,
                                                          reserved, actions, uid2, d2)
                                if feat is None:
                                    continue
                                s = torch.FloatTensor(feat).unsqueeze(0)
                                m = torch.FloatTensor(msk).unsqueeze(0)
                                with torch.no_grad():
                                    probs, _ = (m0 if mp == 0 else m1)(s, m)
                                    ai = torch.argmax(probs).item()
                                execute_action(uid2, d2, ai, actions, reserved)
                            return actions
                        finally:
                            _restore_state(ps, mp, saved)
                    return _agent

                # 50% vs self-best
                opponent_model = best_model if best_model is not None else model
                for _ in range(n_self):
                    seed = ei * 137 + 42
                    _pstates = {0: dict(_INITIAL_STATE), 1: dict(_INITIAL_STATE)}
                    _pstates[0]["nodes"] = set()
                    _pstates[1]["nodes"] = set()
                    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
                    env.run([_make_eval_agent(model, opponent_model, _pstates),
                             _make_eval_agent(model, opponent_model, _pstates)])
                    final = env.steps[-1]
                    if final[0].reward > final[1].reward: ev_wins += 1
                    ei += 1

                # 25% vs v50 best
                for _ in range(n_v50):
                    seed = ei * 137 + 42
                    _pstates = {0: dict(_INITIAL_STATE), 1: dict(_INITIAL_STATE)}
                    _pstates[0]["nodes"] = set()
                    _pstates[1]["nodes"] = set()
                    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
                    env.run([_make_eval_agent(model, v50_model, _pstates),
                             _make_eval_agent(model, v50_model, _pstates)])
                    final = env.steps[-1]
                    if final[0].reward > final[1].reward: ev_wins += 1
                    ei += 1

                # 25% vs v49 best
                for _ in range(n_v49):
                    seed = ei * 137 + 42
                    _pstates = {0: dict(_INITIAL_STATE), 1: dict(_INITIAL_STATE)}
                    _pstates[0]["nodes"] = set()
                    _pstates[1]["nodes"] = set()
                    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
                    env.run([_make_eval_agent(model, v49_model, _pstates),
                             _make_eval_agent(model, v49_model, _pstates)])
                    final = env.steps[-1]
                    if final[0].reward > final[1].reward: ev_wins += 1
                    ei += 1

                ev_wr = ev_wins / EVAL_GAMES * 100
                eval_str = f" eval_M={ev_wr:5.1f}%"
                if ev_wr > best_wr:
                    best_wr = ev_wr
                    best_model = ActorCritic()
                    best_model.load_state_dict(model.state_dict())
                    best_model.eval()
                    torch.save(model.state_dict(), save_path)
                    line = f"[{it+1:4d}/{num_iter}] SP={sp_wr:5.1f}% best={best_wr:5.1f}% p={pl:.4f} v={vl:.4f} e={el:.3f} n={len(all_feat)}{eval_str} t={elapsed:.0f}s\n"
                    log_f.write(f"  -> New best mixed eval WR {best_wr:.0f}% saved\n")
                    log_f.flush()
                else:
                    line = f"[{it+1:4d}/{num_iter}] SP={sp_wr:5.1f}% best={best_wr:5.1f}% p={pl:.4f} v={vl:.4f} e={el:.3f} n={len(all_feat)}{eval_str} t={elapsed:.0f}s\n"

            else:
                line = f"[{it+1:4d}/{num_iter}] SP={sp_wr:5.1f}% best={best_wr:5.1f}% p={pl:.4f} v={vl:.4f} e={el:.3f} n={len(all_feat)} t={elapsed:.0f}s\n"

            print(line, end="")
            log_f.write(line)
            log_f.flush()

    final_path = f"nn_weights_v{version}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal weights saved to {final_path}")
    print(f"Best weights saved to {save_path} (eval WR: {best_wr:.1f}%)")
    return model, version, best_wr


def export_weights(model, version, path=None):
    if path is None:
        path = f"nn_weights_v{version}.py"
    sd = model.state_dict()
    with open(path, "w") as f:
        f.write('"""Auto-generated NN weights."""\nimport numpy as np\n\nWEIGHTS = {\n')
        for name, tensor in sd.items():
            arr = tensor.detach().numpy()
            f.write(f"    '{name}': np.array({arr.tolist()}, dtype=np.float32),\n")
        f.write('}\n')
    print(f"Weights exported to {path}")


def _make_nn_agent(model):
    def nn_agent(obs, config):
        my_player = obs.player
        update_state(obs, config, my_player)
        actions = {}
        reserved = set()
        occupied = {}
        for uid2, d2 in obs.robots.items():
            occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))
        for uid2, d2 in obs.robots.items():
            if d2[4] != my_player or uid2 in actions:
                continue
            feat, msk = extract_unit(obs, config, my_player, occupied,
                                      reserved, actions, uid2, d2)
            if feat is None:
                continue
            s = torch.FloatTensor(feat).unsqueeze(0)
            m = torch.FloatTensor(msk).unsqueeze(0)
            with torch.no_grad():
                probs, _ = model(s, m)
                ai = torch.argmax(probs).item()
            execute_action(uid2, d2, ai, actions, reserved)
        return actions
    return nn_agent


def _run_eval(model, opponent_fn, num_games, label):
    model.eval()
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                      "factory_stuck": 0, "walls": {}})
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)
        env.run([_make_nn_agent(model), opponent_fn])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        if r0 > r1: wins += 1
        elif r0 < r1: losses += 1
        else: draws += 1
    print(f"Eval vs {label}: {wins}W-{losses}L-{draws}D ({wins/num_games*100:.1f}%)", flush=True)
    return wins / num_games


def _run_eval_nn(model, opponent_model, num_games, label):
    opponent_model.eval()
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        _pstates = {0: dict(_INITIAL_STATE), 1: dict(_INITIAL_STATE)}
        _pstates[0]["nodes"] = set()
        _pstates[1]["nodes"] = set()
        env = make("crawl", configuration={"randomSeed": seed}, debug=True)

        def _make_p_agent(m0, m1, ps):
            def _agent(obs, config):
                mp = obs.player
                saved = _swap_state(ps, mp)
                try:
                    update_state(obs, config, mp)
                    actions = {}
                    reserved = set()
                    occupied = {}
                    for uid2, d2 in obs.robots.items():
                        occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))
                    for uid2, d2 in obs.robots.items():
                        if d2[4] != mp or uid2 in actions:
                            continue
                        feat, msk = extract_unit(obs, config, mp, occupied,
                                                  reserved, actions, uid2, d2)
                        if feat is None:
                            continue
                        s = torch.FloatTensor(feat).unsqueeze(0)
                        m = torch.FloatTensor(msk).unsqueeze(0)
                        with torch.no_grad():
                            probs, _ = (m0 if mp == 0 else m1)(s, m)
                            ai = torch.argmax(probs).item()
                        execute_action(uid2, d2, ai, actions, reserved)
                    return actions
                finally:
                    _restore_state(ps, mp, saved)
            return _agent

        env.run([_make_p_agent(model, opponent_model, _pstates),
                 _make_p_agent(model, opponent_model, _pstates)])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        if r0 > r1: wins += 1
        elif r0 < r1: losses += 1
        else: draws += 1
    print(f"Eval vs {label}: {wins}W-{losses}L-{draws}D ({wins/num_games*100:.1f}%)", flush=True)
    return wins / num_games


def final_evaluate(model, version):
    model.load_state_dict(torch.load(f"nn_weights_v{version}.pt"))
    model.eval()
    print(f"\nLoaded best weights (nn_weights_v{version}.pt)", flush=True)

    _run_eval(model, "random", 500, "random")

    v49_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), V49_WEIGHTS)
    if os.path.exists(v49_path):
        v49 = ActorCritic()
        v49.load_state_dict(torch.load(v49_path, map_location="cpu"))
        v49.eval()
        _run_eval_nn(model, v49, 500, "v49 best")

    v50_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), V50_WEIGHTS)
    if os.path.exists(v50_path):
        v50 = ActorCritic()
        v50.load_state_dict(torch.load(v50_path, map_location="cpu"))
        v50.eval()
        _run_eval_nn(model, v50, 500, "v50 best")


VERSION_OVERRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else None
NUM_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
TERMINAL_WIN_ARG = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

if __name__ == "__main__":
    model, ver, best = train(num_iter=NUM_ITER, batch=100, lr=0.0003,
                             version=VERSION_OVERRIDE, terminal_win=TERMINAL_WIN_ARG)
    final_evaluate(model, ver)
    export_weights(model, ver)
