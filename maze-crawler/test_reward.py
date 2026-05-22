"""Batch reward function statistics over many games."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from kaggle_environments import make
from agent_v1 import (
    STATE, TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER, update_state, agent as agent_v1,
)

DELTA_GAP_W = 0.5
DELTA_UNITS_W = 0.1
SHAPING_REMOVE = 0.1
SHAPING_TRANSFORM = 0.5
UNIT_SURVIVAL = 0.01
TERMINAL_WIN = 5.0
TERMINAL_LOSS = -1.0


def _unit_count(robots, player):
    return sum(1 for d in robots.values() if d[4] == player and d[0] != TYPE_FACTORY)


def _factory_gap(robots, player, southBound):
    for d in robots.values():
        if d[4] == player and d[0] == TYPE_FACTORY:
            return d[2] - southBound
    return 0


def run_game(seed):
    STATE.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                  "factory_stuck": 0, "walls": {}})

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)

    step_rewards = []  # per-step team_r values
    factory_step_r = []
    scout_step_r = []
    worker_step_r = []
    miner_step_r = []

    prev_units = None
    prev_gap = None
    first_turn = True

    def tracker(obs, config):
        nonlocal prev_units, prev_gap, first_turn

        my_player = obs.player
        update_state(obs, config, my_player)

        cur_units = _unit_count(obs.robots, my_player)
        cur_gap = _factory_gap(obs.robots, my_player, obs.southBound)

        if first_turn:
            prev_units = cur_units
            prev_gap = cur_gap
            first_turn = False
            team_r = 0.0
        else:
            delta_gap = (cur_gap - prev_gap) * DELTA_GAP_W
            delta_units = (cur_units - prev_units) * DELTA_UNITS_W
            team_r = delta_gap + delta_units
            prev_units = cur_units
            prev_gap = cur_gap

        step_rewards.append(team_r)

        # Per-unit step rewards
        for uid, d in obs.robots.items():
            if d[4] != my_player:
                continue
            utype = d[0]
            unit_r = team_r
            if utype != TYPE_FACTORY:
                unit_r += UNIT_SURVIVAL
            if utype == TYPE_FACTORY:
                factory_step_r.append(unit_r)
            elif utype == TYPE_SCOUT:
                scout_step_r.append(unit_r)
            elif utype == TYPE_WORKER:
                worker_step_r.append(unit_r)
            elif utype == TYPE_MINER:
                miner_step_r.append(unit_r)

        actions = agent_v1(obs, config)
        return actions

    env.run([tracker, "random"])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    outcome = "WIN" if r0 > r1 else ("LOSS" if r0 < r1 else "DRAW")
    terminal_r = TERMINAL_WIN if outcome == "WIN" else (TERMINAL_LOSS if outcome == "LOSS" else 0.0)

    cum_step = sum(step_rewards)
    total = cum_step + terminal_r

    return {
        "outcome": outcome,
        "cum_step": cum_step,
        "terminal": terminal_r,
        "total": total,
        "num_steps": len(step_rewards),
        "step_mean": np.mean(step_rewards) if step_rewards else 0,
        "step_std": np.std(step_rewards) if step_rewards else 0,
        "factory_mean": np.mean(factory_step_r) if factory_step_r else 0,
        "scout_mean": np.mean(scout_step_r) if scout_step_r else 0,
        "worker_mean": np.mean(worker_step_r) if worker_step_r else 0,
        "miner_mean": np.mean(miner_step_r) if miner_step_r else 0,
        "scout_steps": len(scout_step_r),
        "worker_steps": len(worker_step_r),
        "miner_steps": len(miner_step_r),
        "factory_steps": len(factory_step_r),
    }


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"Running {num_games} games with reward function:")
    print(f"  team_r = delta_gap*{DELTA_GAP_W} + delta_units*{DELTA_UNITS_W}")
    print(f"  non-factory survival = +{UNIT_SURVIVAL}/step")
    print(f"  shaping: REMOVE +{SHAPING_REMOVE}, TRANSFORM +{SHAPING_TRANSFORM}")
    print(f"  terminal: +{TERMINAL_WIN} / {TERMINAL_LOSS} / 0")
    print()

    results = []
    for i in range(num_games):
        seed = i * 137 + 42
        r = run_game(seed)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_games} done...")

    # Aggregate
    wins = [r for r in results if r["outcome"] == "WIN"]
    losses = [r for r in results if r["outcome"] == "LOSS"]
    draws = [r for r in results if r["outcome"] == "DRAW"]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(wins)}W-{len(losses)}L-{len(draws)}D "
          f"({len(wins)/num_games*100:.1f}%) over {num_games} games")
    print(f"{'='*60}")

    for label, group in [("WIN", wins), ("LOSS", losses), ("DRAW", draws), ("ALL", results)]:
        if not group:
            continue
        n = len(group)
        cum_steps = [r["cum_step"] for r in group]
        totals = [r["total"] for r in group]
        print(f"\n--- {label} ({n} games) ---")
        print(f"  cum_step:  mean={np.mean(cum_steps):+.3f}  "
              f"std={np.std(cum_steps):.3f}  "
              f"min={min(cum_steps):+.3f}  max={max(cum_steps):+.3f}")
        print(f"  total:     mean={np.mean(totals):+.3f}  "
              f"std={np.std(totals):.3f}  "
              f"min={min(totals):+.3f}  max={max(totals):+.3f}")
        print(f"  steps:     mean={np.mean([r['num_steps'] for r in group]):.0f}")

        for utype in ["factory", "scout", "worker", "miner"]:
            means = [r[f"{utype}_mean"] for r in group]
            steps = [r[f"{utype}_steps"] for r in group]
            avg_steps = np.mean(steps)
            if avg_steps > 0:
                print(f"  {utype:>8}: mean_step_r={np.mean(means):+.4f}  "
                      f"avg_steps={avg_steps:.1f}")

    # Distribution check: does step_total separate WIN from LOSS?
    print(f"\n--- Separation Check ---")
    if wins and losses:
        win_totals = [r["cum_step"] for r in wins]
        loss_totals = [r["cum_step"] for r in losses]
        overlap = sum(1 for w in win_totals if w < max(loss_totals)) + \
                  sum(1 for l in loss_totals if l > min(win_totals))
        print(f"  WIN step_total range: [{min(win_totals):+.2f}, {max(win_totals):+.2f}]")
        print(f"  LOSS step_total range: [{min(loss_totals):+.2f}, {max(loss_totals):+.2f}]")
        print(f"  Overlap count: {overlap}/{len(wins)+len(losses)} "
              f"({overlap/(len(wins)+len(losses))*100:.0f}%)")

        win_all = [r["total"] for r in wins]
        loss_all = [r["total"] for r in losses]
        print(f"  WIN total range: [{min(win_all):+.2f}, {max(win_all):+.2f}]")
        print(f"  LOSS total range: [{min(loss_all):+.2f}, {max(loss_all):+.2f}]")
