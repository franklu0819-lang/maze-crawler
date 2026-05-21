"""Diagnose reward function by running games and logging reward components."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from kaggle_environments import make

from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER,
    parse_key, in_bounds, wb, can_go, update_state,
    friendly_at, DIRS,
)

def _total_energy(obs, player):
    total = 0
    for d in obs.robots.values():
        if d[4] == player:
            total += d[3]
    return total

def _unit_count(obs, player):
    return sum(1 for d in obs.robots.values() if d[4] == player and d[0] != TYPE_FACTORY)

def run_diagnostic(seed):
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}})
    env = make("crawl", configuration={"randomSeed": seed}, debug=True)

    prev_total_energy = None
    prev_units = None
    prev_gap = None
    first_turn = True
    logs = []

    def agent(obs, config):
        nonlocal prev_total_energy, prev_units, prev_gap, first_turn

        my_player = obs.player
        update_state(obs, config, my_player)

        cur_e = _total_energy(obs, my_player)
        cur_units = _unit_count(obs, my_player)
        cur_gap = 0
        for uid, d in obs.robots.items():
            if d[4] == my_player and d[0] == TYPE_FACTORY:
                cur_gap = d[2] - obs.southBound
                break

        if first_turn:
            prev_total_energy = cur_e
            prev_units = cur_units
            prev_gap = cur_gap
            first_turn = False
            delta_e = 0.0
            delta_gap = 0.0
            delta_units = 0.0
        else:
            delta_e = (cur_e - prev_total_energy) / 1000.0
            delta_gap = (cur_gap - prev_gap) * 0.1
            delta_units = (cur_units - prev_units) * 0.05
            prev_total_energy = cur_e
            prev_units = cur_units
            prev_gap = cur_gap

        step_r = delta_e + delta_gap + delta_units

        # Log actions
        actions = {}
        action_summary = []
        for uid, d in obs.robots.items():
            if d[4] != my_player:
                continue
            utype_names = {TYPE_FACTORY: "FAC", TYPE_SCOUT: "SCT", TYPE_WORKER: "WRK", TYPE_MINER: "MNR"}
            action_summary.append(f"{utype_names.get(d[0],'?')}(e={d[3]:.0f})")
        actions_str = " ".join(action_summary)

        logs.append({
            "turn": STATE["turn"],
            "delta_e": delta_e,
            "delta_gap": delta_gap,
            "delta_units": delta_units,
            "step_r": step_r,
            "total_e": cur_e,
            "gap": cur_gap,
            "units": cur_units,
            "robots": actions_str,
        })

        # Simple random-ish policy: move north, build when possible
        reserved = set()
        occupied = {}
        for uid, d in obs.robots.items():
            occupied.setdefault((d[1], d[2]), []).append((uid, d))
        for uid, d in obs.robots.items():
            if d[4] != my_player:
                continue
            utype = d[0]
            c, r, energy = d[1], d[2], d[3]
            if utype == TYPE_FACTORY:
                move_cd = d[5] if len(d) > 5 else 0
                build_cd = d[7] if len(d) > 7 else 0
                if move_cd == 0:
                    for d_dir in ["NORTH", "EAST", "WEST"]:
                        if can_go(obs, config, c, r, d_dir):
                            dc, dr, _ = DIRS[d_dir]
                            nxt = (c + dc, r + dr)
                            if nxt not in reserved:
                                actions[uid] = d_dir
                                reserved.add(nxt)
                                break
                if uid not in actions and build_cd == 0:
                    if energy >= 200 and can_go(obs, config, c, r, "NORTH"):
                        actions[uid] = "BUILD_WORKER"
                        reserved.add((c, r + 1))
                if uid not in actions:
                    actions[uid] = "IDLE"
                    reserved.add((c, r))
            else:
                if uid not in actions:
                    for d_dir in ["NORTH", "EAST", "WEST"]:
                        if can_go(obs, config, c, r, d_dir):
                            dc, dr, _ = DIRS[d_dir]
                            nxt = (c + dc, r + dr)
                            if nxt not in reserved and not friendly_at(occupied, nxt, my_player):
                                actions[uid] = d_dir
                                reserved.add(nxt)
                                break
                if uid not in actions:
                    actions[uid] = "IDLE"
                    reserved.add((c, r))
        return actions

    env.run([agent, "random"])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    return logs, r0, r1

# Run 3 games
for game_idx in range(3):
    seed = random.randint(0, 999999)
    logs, r0, r1 = run_diagnostic(seed)
    result = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
    print(f"\n{'='*80}")
    print(f"Game {game_idx+1} (seed={seed}): {result} r0={r0:.1f} r1={r1:.1f}")
    print(f"{'='*80}")

    # Aggregate stats
    delta_es = [l["delta_e"] for l in logs]
    delta_gaps = [l["delta_gap"] for l in logs]
    delta_units_all = [l["delta_units"] for l in logs]
    step_rs = [l["step_r"] for l in logs]

    print(f"  Steps: {len(logs)}")
    print(f"  delta_e/1000:   min={min(delta_es):.4f}  max={max(delta_es):.4f}  "
          f"mean={np.mean(delta_es):.4f}  sum={sum(delta_es):.2f}")
    print(f"  delta_gap*0.1:  min={min(delta_gaps):.4f}  max={max(delta_gaps):.4f}  "
          f"mean={np.mean(delta_gaps):.4f}  sum={sum(delta_gaps):.2f}")
    print(f"  delta_units*0.05: min={min(delta_units_all):.4f}  max={max(delta_units_all):.4f}  "
          f"mean={np.mean(delta_units_all):.4f}  sum={sum(delta_units_all):.2f}")
    print(f"  step_r total:   {sum(step_rs):.4f}")

    # Show every 20th step
    print(f"\n  {'Turn':>4s}  {'d_e/1k':>7s}  {'d_gap':>6s}  {'d_unt':>6s}  {'step_r':>7s}  "
          f"{'total_e':>8s}  {'gap':>4s}  {'units':>5s}  robots")
    print(f"  {'-'*75}")
    for i, l in enumerate(logs):
        if i % 20 == 0 or abs(l["delta_e"]) > 0.1 or abs(l["delta_units"]) > 0.01:
            print(f"  {l['turn']:4d}  {l['delta_e']:+7.4f}  {l['delta_gap']:+6.3f}  "
                  f"{l['delta_units']:+6.3f}  {l['step_r']:+7.4f}  "
                  f"{l['total_e']:8.0f}  {l['gap']:4d}  {l['units']:5d}  {l['robots'][:60]}")
