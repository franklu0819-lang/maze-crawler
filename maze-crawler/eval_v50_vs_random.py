"""Eval: v50 (all-unit NN) vs random agent, 500 games."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from kaggle_environments import make

from train_v17 import ActorCritic, extract_unit, execute_action
from agent_v1 import update_state, STATE as S49

model49 = ActorCritic()
model49.load_state_dict(torch.load('nn_weights_v50.pt'))
model49.eval()
print("Loaded v50 best weights", flush=True)

NUM_GAMES = 500
wins, losses, draws = 0, 0, 0

for i in range(NUM_GAMES):
    seed = i * 137 + 42

    S49.update({"turn": 0, "nodes": set(), "last_factory_pos": None,
                "factory_stuck": 0, "walls": {}})

    env = make("crawl", configuration={"randomSeed": seed}, debug=True)

    def make_v50_agent():
        def v50_agent(obs, config):
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
                    probs, _ = model49(s, m)
                    ai = torch.argmax(probs).item()
                execute_action(uid2, d2, ai, actions, reserved)
            return actions
        return v50_agent

    env.run([make_v50_agent(), "random"])
    final = env.steps[-1]
    r0, r1 = final[0].reward, final[1].reward
    if r0 > r1:
        wins += 1
    elif r0 < r1:
        losses += 1
    else:
        draws += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{NUM_GAMES}: {wins}W-{losses}L-{draws}D ({wins/(i+1)*100:.1f}%)", flush=True)

print(f"\n=== v50 vs random: {wins}W-{losses}L-{draws}D ({wins/NUM_GAMES*100:.1f}%) ===", flush=True)
