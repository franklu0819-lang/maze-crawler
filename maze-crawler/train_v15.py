"""PPO training with NN controlling ALL units + per-unit behavioral shaping.

Factory, scout, worker, miner all use the same network.
13-action space with type-specific masking.
Team reward: delta_gap*0.1 + delta_units*0.05 + terminal (win: total_e/200, loss: -1).
Per-unit shaping: REMOVE +0.05, TRANSFORM +0.05, scout ahead +0.01.
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
PPO_CLIP = 0.2
PPO_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


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


# ─── Reward ──────────────────────────────────────────────────────────

def _total_energy(obs, player):
    total = 0
    for d in obs.robots.values():
        if d[4] == player:
            total += d[3]
    return total


def _unit_count(robots, player):
    return sum(1 for d in robots.values() if d[4] == player and d[0] != TYPE_FACTORY)


def _compute_step_reward(obs, config, my_player, prev_total_energy, prev_units, prev_gap):
    total_e = _total_energy(obs, my_player)
    delta_e = (total_e - prev_total_energy) / 1000.0
    cur_units = _unit_count(obs.robots, my_player)
    delta_units = (cur_units - prev_units) * DELTA_UNITS

    my_gap = 0
    for uid, d in obs.robots.items():
        if d[4] == my_player and d[0] == TYPE_FACTORY:
            my_gap = d[2] - obs.southBound
            break
    delta_gap = (my_gap - prev_gap) * 1.0

    return delta_e + delta_units + delta_gap + 0.01


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

def _reset_state():
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}})


def _compute_shaping(utype, action_idx, factory_pos, unit_pos):
    """Per-unit behavioral shaping reward."""
    shaping = 0.0
    if utype == TYPE_WORKER and 5 <= action_idx <= 7:
        shaping += 0.05
    if utype == TYPE_MINER and action_idx == 11:
        shaping += 0.1
    if utype == TYPE_SCOUT and factory_pos is not None:
        if unit_pos[1] > factory_pos[1]:
            shaping += 0.01
    return shaping


def run_ppo_game(model, seed, explore=True):
    _reset_state()
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)
    unit_trajs = {}  # uid -> [(feat, ai, mask, log_p, val, step_r)]
    prev_total_energy = [None]
    prev_units = [None]
    prev_gap = [None]
    first_turn = [True]

    def ppo_agent(obs, config):
        my_player = obs.player
        update_state(obs, config, my_player)

        cur_total_energy = _total_energy(obs, my_player)
        cur_units = _unit_count(obs.robots, my_player)
        cur_gap = 0
        factory_pos = None
        for uid2, d2 in obs.robots.items():
            if d2[4] == my_player and d2[0] == TYPE_FACTORY:
                cur_gap = d2[2] - obs.southBound
                factory_pos = (d2[1], d2[2])
                break

        if first_turn[0]:
            prev_total_energy[0] = cur_total_energy
            prev_units[0] = cur_units
            prev_gap[0] = cur_gap
            first_turn[0] = False
            team_r = 0.0
        else:
            team_r = _compute_step_reward(obs, config, my_player,
                                          prev_total_energy[0], prev_units[0], prev_gap[0])
            prev_total_energy[0] = cur_total_energy
            prev_units[0] = cur_units
            prev_gap[0] = cur_gap

        actions = {}
        reserved = set()
        occupied = {}
        for uid2, d2 in obs.robots.items():
            occupied.setdefault((d2[1], d2[2]), []).append((uid2, d2))

        # Process units in order: scouts -> workers -> miners -> factory
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
                    if explore:
                        ai, log_p, val = model.get_action(s, m)
                        ai_item = ai.item()
                        log_p_item = log_p.item()
                        val_item = val.item()
                    else:
                        probs, val = model(s, m)
                        ai_item = torch.argmax(probs).item()
                        log_p_item = 0.0
                        val_item = val.item()

                unit_pos = (d2[1], d2[2])
                shaping = _compute_shaping(unit_type, ai_item, factory_pos, unit_pos)
                step_r = team_r + shaping

                if uid2 not in unit_trajs:
                    unit_trajs[uid2] = []
                unit_trajs[uid2].append(
                    (feat.copy(), ai_item, msk.copy(), log_p_item, val_item, step_r))

                execute_action(uid2, d2, ai_item, actions, reserved)

        return actions

    env.run([ppo_agent, "random"])
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


def train(num_iter=200, batch=50, lr=0.0003, version=None):
    if version is None:
        version = _next_version()
    save_path = f"nn_weights_v{version}.pt"
    print(f"=== Train v{version} (All-unit NN, PPO vs random, v10-based reward) ===")
    print(f"  Input: {INPUT_SIZE} | Actions: {NUM_ACTIONS}")
    print(f"  Team: delta_e/1000 + delta_gap*1.0 + delta_units*{DELTA_UNITS} + survival*0.01")
    print(f"  Shaping: REMOVE +0.05 | TRANSFORM +0.1 | scout_ahead +0.01")
    print(f"  Terminal: +5/-1")
    print(f"  Iterations: {num_iter} | Batch: {batch}")
    print(f"  Weights -> {save_path}")

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_wr = 0
    t0 = time.time()

    for it in range(num_iter):
        all_feat, all_act, all_mask, all_old_lp = [], [], [], []
        all_adv, all_ret = [], []
        wins = 0

        for _ in range(batch):
            seed = random.randint(0, 999999)
            unit_trajs, r0, r1 = run_ppo_game(model, seed, explore=True)

            if r0 > r1:
                wins += 1

            terminal_r = 5.0 if r0 > r1 else (-1.0 if r0 < r1 else 0.0)
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

        wr = wins / batch * 100
        elapsed = time.time() - t0
        print(f"[{it+1:3d}/{num_iter}] WR={wr:5.1f}% p_loss={pl:.4f} "
              f"v_loss={vl:.4f} ent={el:.3f} n={len(all_feat)} t={elapsed:.0f}s")

        if wr > best_wr:
            best_wr = wr
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best {best_wr:.0f}% saved")

    final_path = f"nn_weights_v{version}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final weights saved to {final_path}")
    return model, version, best_wr


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate_vs_random(model, num_games=500):
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        seed = i * 137 + 42
        _, r0, r1 = run_ppo_game(model, seed, explore=False)
        if r0 > r1: wins += 1
        elif r0 < r1: losses += 1
        else: draws += 1
    print(f"Eval vs random: {wins}W-{losses}L-{draws}D ({wins/num_games*100:.1f}%)")
    return wins, losses, draws


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


VERSION_OVERRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else None
NUM_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DELTA_UNITS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

if __name__ == "__main__":
    model, ver, best = train(num_iter=NUM_ITER, batch=100, lr=0.0003, version=VERSION_OVERRIDE)
    model.load_state_dict(torch.load(f"nn_weights_v{ver}.pt"))
    evaluate_vs_random(model)
    export_weights(model, ver)
