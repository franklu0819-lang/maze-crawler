"""Analyze escape mechanism prerequisites at the moment factory gets fatally stuck."""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make
import agent_v1

def load_opp(name):
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"agent_submit_{name}.py")
    spec = importlib.util.spec_from_file_location(f"agent_{name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def analyze_death(seed, opp_mod):
    agent_v1.STATE.update({
        "turn": 0, "walls": {}, "nodes": set(),
        "last_factory_pos": None, "factory_stuck": 0,
        "mine_invested": None, "mine_wait": False, "mine_wait_since": 0,
    })
    if hasattr(opp_mod, 'STATE'):
        opp_mod.STATE.update({
            "turn": 0, "walls": {}, "nodes": set(),
            "last_factory_pos": None, "factory_stuck": 0,
            "mine_invested": None, "mine_wait": False, "mine_wait_since": 0,
        })

    env = _make("crawl", configuration={"randomSeed": seed}, debug=True)
    config = env.configuration
    env.run([agent_v1.agent, opp_mod.agent])
    steps = env.steps

    # Find death step and last JUMP
    factory_death = len(steps)
    jumps = []  # (step, from_pos, to_pos)
    worker_death = None
    prev_factory = None

    for si in range(len(steps)):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        factory = next((v for v in robots.values() if v[4] == 0 and v[0] == 0), None)

        if factory is None:
            factory_death = si
            break

        c, r = factory[1], factory[2]

        # Detect JUMP
        if prev_factory and r - prev_factory[1] >= 2:
            jumps.append((si, prev_factory, (c, r)))

        # Detect worker death
        has_worker = any(v[4] == 0 and v[0] == 2 for v in robots.values())
        if not has_worker and worker_death is None and si > 5:
            # Find last step worker was alive
            for si2 in range(si - 1, max(0, si - 10), -1):
                obs2 = steps[si2][0].observation
                if any(v[4] == 0 and v[0] == 2 for v in obs2.robots.values()):
                    worker_death = si2
                    break
            if worker_death is None:
                worker_death = 0  # Worker never existed or died before tracking

        prev_factory = (c, r)

    # Find the "fatal stuck" moment: last JUMP before death that landed in dead end
    last_jump = jumps[-1] if jumps else None

    print(f"\n  Seed {seed}: death step={factory_death}")
    print(f"  Worker died at step: {worker_death}")
    print(f"  Last JUMP: step {last_jump[0]}, {last_jump[1]}→{last_jump[2]}" if last_jump else "  No JUMPs")

    if last_jump:
        jstep, jfrom, jto = last_jump
        cd_expires = jstep + 20
        steps_to_cd = cd_expires - factory_death
        print(f"  JUMP CD expires: step {cd_expires} ({steps_to_cd:+d} steps {'after' if steps_to_cd > 0 else 'before'} death)")

        # Check landing cell walls at jump time
        obs_j = steps[jstep][0].observation
        jc, jr = jto
        n_w = "WALL" if not agent_v1.can_go(obs_j, config, jc, jr, "NORTH") else "open"
        e_w = "WALL" if not agent_v1.can_go(obs_j, config, jc, jr, "EAST") else "open"
        w_w = "WALL" if not agent_v1.can_go(obs_j, config, jc, jr, "WEST") else "open"
        s_w = "WALL" if not agent_v1.can_go(obs_j, config, jc, jr, "SOUTH") else "open"
        print(f"  Landing cell ({jc},{jr}) walls: N={n_w} E={e_w} W={w_w} S={s_w}")

        # Was landing cell known or unknown?
        landing_w = agent_v1.wb(obs_j, config, jc, jr)
        print(f"  Landing cell wall data: {'UNKNOWN' if landing_w is None else f'known ({landing_w})'}")

        # Check north wall of landing - could worker clear it?
        w_bit_n = 1  # BIT_N
        if landing_w is not None and (landing_w & w_bit_n):
            print(f"  North wall at landing: YES — worker could clear (costs 100 energy)")
        else:
            print(f"  North wall at landing: {'NO (open)' if landing_w is not None else 'UNKNOWN'}")

    # Check factory energy and gap at various points
    print(f"\n  Timeline at key moments:")
    for label, step_target in [("Worker death", worker_death), ("Last JUMP", last_jump[0] if last_jump else None), ("Death-20", factory_death - 20), ("Death-10", factory_death - 10), ("Death-1", factory_death - 1)]:
        if step_target is None or step_target < 0 or step_target >= len(steps):
            continue
        obs = steps[step_target][0].observation
        robots = obs.get("robots", {})
        f = next((v for v in robots.values() if v[4] == 0 and v[0] == 0), None)
        if f is None:
            continue
        gap = f[2] - obs.southBound
        has_w = any(v[4] == 0 and v[0] == 2 for v in robots.values())
        # JUMP CD
        jump_cd = f[6] if len(f) > 6 else 0
        has_mine = any(v[2] == 0 for v in obs.get("mines", {}).values())
        print(f"    Step {step_target:>4}: gap={gap:>2} E={f[3]:>6.0f} worker={'ALIVE' if has_w else 'DEAD '} "
              f"jump_cd={jump_cd:>2} mine={'YES' if has_mine else 'NO '}")


# Test v50 losses
v50_mod = load_opp("v50")
v15_mod = load_opp("v15")

print("=" * 70)
print("V50 LOSSES (4 games)")
print("=" * 70)
for seed in [590, 1412, 6070, 11687]:
    analyze_death(seed, v50_mod)

print("\n" + "=" * 70)
print("V15 LOSSES (9 games)")
print("=" * 70)
for seed in [2645, 2919, 5111, 6344, 6618, 7440, 7851, 8125, 10591]:
    analyze_death(seed, v15_mod)
