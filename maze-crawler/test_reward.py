"""Simulate reward function on actual games to verify reward dynamics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from kaggle_environments import make
from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER, update_state, agent as agent_v1,
)

DELTA_UNITS = 0.1
DELTA_GAP = 0.1
REWARD_SCALE = 1000.0
SHAPING_REMOVE = 0.2
SHAPING_TRANSFORM = 0.2


def _total_energy(robots, player):
    return sum(d[3] for d in robots.values() if d[4] == player)


def _unit_count(robots, player):
    return sum(1 for d in robots.values() if d[4] == player and d[0] != TYPE_FACTORY)


def _get_factory_info(robots, player):
    for uid, d in robots.items():
        if d[4] == player and d[0] == TYPE_FACTORY:
            return uid, d[1], d[2]
    return None, None, None


def _get_action_type(uid, robots, player):
    """Guess action from state changes — simplified heuristic."""
    return "unknown"


def analyze_game(seed):
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}})

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)

    history = []  # list of per-step dicts
    prev = {"energy": None, "units": None, "gap": None}

    # Track per-unit positions to detect actions
    prev_positions = {}

    def tracker(obs, config):
        my_player = obs.player
        update_state(obs, config, my_player)

        cur_energy = _total_energy(obs.robots, my_player)
        cur_units = _unit_count(obs.robots, my_player)
        _, _, factory_row = _get_factory_info(obs.robots, my_player)
        cur_gap = (factory_row - obs.southBound) if factory_row is not None else 0

        step_info = {"turn": STATE["turn"], "energy": cur_energy,
                     "units": cur_units, "gap": cur_gap, "southBound": obs.southBound}

        if prev["energy"] is not None and "actions" not in step_info:
            delta_e = (cur_energy - prev["energy"]) / REWARD_SCALE
            delta_units = (cur_units - prev["units"]) * DELTA_UNITS
            delta_gap = (cur_gap - prev["gap"]) * DELTA_GAP

            # Detect specific actions from position changes
            actions_taken = []
            for uid, d in obs.robots.items():
                if d[4] != my_player:
                    continue
                prev_pos = prev_positions.get(uid)
                cur_pos = (d[1], d[2])
                if prev_pos is None:
                    utype_name = ["Factory", "Scout", "Worker", "Miner"][d[0]]
                    actions_taken.append(f"NEW {utype_name} at {cur_pos}")
                elif prev_pos != cur_pos:
                    utype_name = ["Factory", "Scout", "Worker", "Miner"][d[0]]
                    dc = cur_pos[0] - prev_pos[0]
                    dr = cur_pos[1] - prev_pos[1]
                    direction = ""
                    if dr > 0: direction = "N"
                    elif dr < 0: direction = "S"
                    if dc > 0: direction += "E"
                    elif dc < 0: direction += "W"
                    if abs(dr) == 2 and dc == 0:
                        direction = "JUMP_N"
                    actions_taken.append(f"{utype_name} {prev_pos}->{cur_pos} ({direction})")

            # Detect disappeared units
            current_uids = {uid for uid, d in obs.robots.items() if d[4] == my_player}
            for uid in prev_positions:
                if uid not in current_uids and not uid.startswith("factory"):
                    actions_taken.append(f"UNIT LOST: {uid} was at {prev_positions[uid]}")

            # Count REMOVE and TRANSFORM heuristically:
            # REMOVE: worker stayed in place, energy dropped by ~100
            # TRANSFORM: miner disappeared, was on mining node
            remove_count = 0
            transform_count = 0

            step_info["delta_e"] = delta_e
            step_info["delta_units"] = delta_units
            step_info["delta_gap"] = delta_gap
            step_info["base_reward"] = delta_e + delta_units + delta_gap
            step_info["actions"] = actions_taken

        history.append(step_info)

        # Update tracking
        prev["energy"] = cur_energy
        prev["units"] = cur_units
        prev["gap"] = cur_gap
        prev_positions.clear()
        for uid, d in obs.robots.items():
            if d[4] == my_player:
                prev_positions[uid] = (d[1], d[2])

        # Use agent_v1 for realistic behavior
        actions = agent_v1(obs, config)
        return actions

    env.run([tracker, "random"])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
    return history, outcome, r0, r1


def print_analysis(history, outcome, r0, r1, seed):
    print(f"\n{'='*70}")
    print(f"Game seed={seed} | Outcome: {outcome} (r0={r0:.1f}, r1={r1:.1f})")
    print(f"{'='*70}")

    cum_e = cum_units = cum_gap = cum_base = 0.0
    print(f"\n{'Turn':>4} {'dE/1k':>7} {'dUnits':>7} {'dGap':>7} {'Base':>7} "
          f"{'CumBase':>7} {'Energy':>7} {'Units':>5} {'Gap':>5} {'Actions'}")
    print("-" * 100)

    for i, step in enumerate(history):
        if "delta_e" not in step:
            print(f"{step['turn']:>4} {'---':>7} {'---':>7} {'---':>7} {'---':>7} "
                  f"{'---':>7} {step['energy']:>7.0f} {step['units']:>5} {step['gap']:>5} "
                  f"(first turn)")
            continue

        cum_e += step["delta_e"]
        cum_units += step["delta_units"]
        cum_gap += step["delta_gap"]
        cum_base += step["base_reward"]

        actions_str = ", ".join(step["actions"][:4]) if step["actions"] else ""
        if len(step["actions"]) > 4:
            actions_str += f" +{len(step['actions'])-4} more"

        print(f"{step['turn']:>4} {step['delta_e']:>+7.3f} {step['delta_units']:>+7.3f} "
              f"{step['delta_gap']:>+7.3f} {step['base_reward']:>+7.3f} "
              f"{cum_base:>+7.3f} {step['energy']:>7.0f} {step['units']:>5} {step['gap']:>5} "
              f"{actions_str}")

    # Terminal
    terminal_r = 1.0 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.5)
    print(f"\n{'TERM':>4} {'':>7} {'':>7} {'':>7} {terminal_r:>+7.1f} "
          f"{cum_base + terminal_r:>+7.3f} {'':>7} {'':>5} {'':>5} {outcome}")
    print(f"\nCumulative breakdown:")
    print(f"  delta_energy: {cum_e:>+.3f}")
    print(f"  delta_units:  {cum_units:>+.3f}")
    print(f"  delta_gap:    {cum_gap:>+.3f}")
    print(f"  step total:   {cum_base:>+.3f}")
    print(f"  terminal:     {terminal_r:>+.3f}")
    print(f"  TOTAL:        {cum_base + terminal_r:>+.3f}")


if __name__ == "__main__":
    seeds = [42, 137, 256, 999, 1234]
    for seed in seeds:
        history, outcome, r0, r1 = analyze_game(seed)
        print_analysis(history, outcome, r0, r1, seed)
