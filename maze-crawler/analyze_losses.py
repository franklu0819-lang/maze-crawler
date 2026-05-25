"""Detailed analysis of loss games - trace factory behavior step by step."""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_environments import make as _make
import agent_v1

# Load opponent
OPP_FILE = sys.argv[1] if len(sys.argv) > 1 else "agent_submit_v50.py"
OPP_NAME = OPP_FILE.replace("agent_submit_", "").replace(".py", "")
spec = importlib.util.spec_from_file_location("opp", os.path.join(os.path.dirname(os.path.abspath(__file__)), OPP_FILE))
opp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opp_mod)

SEEDS = [int(x) for x in " ".join(sys.argv[2:]).replace(",", " ").split()] if len(sys.argv) > 2 else []

def analyze_loss(seed):
    # Reset states
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
    total = len(steps)

    # Find factory death
    factory_death = total
    for si in range(total):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        if not any(v[4] == 0 and v[0] == 0 for v in robots.values()):
            factory_death = si
            break

    print(f"\n{'='*80}")
    print(f"SEED {seed} | Steps: {total} | Factory life: {factory_death}")
    print(f"{'='*80}")

    # Scan entire game for key events
    events = []
    max_gap = 0
    gap_drops = []  # (step, gap_before, gap_after)
    prev_gap = None
    prev_pos = None
    jump_steps = []
    south_steps = []  # steps where factory moved south
    stuck_periods = []  # (start_step, end_step, duration)
    in_stuck = False
    stuck_start = None
    worker_alive_until = None
    mine_built_step = None
    factory_energies = []

    for si in range(min(total, factory_death + 1)):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        my = {k: v for k, v in robots.items() if v[4] == 0}
        factory = next((v for v in my.values() if v[0] == 0), None)
        if factory is None:
            break

        c, r, energy = factory[1], factory[2], factory[3]
        gap = r - obs.southBound
        pos = (c, r)
        factory_energies.append((si, energy))

        # Track gap
        max_gap = max(max_gap, gap)
        if prev_gap is not None and gap < prev_gap - 1:
            gap_drops.append((si, prev_gap, gap))
        prev_gap = gap

        # Track position / stuck
        if prev_pos is not None:
            if r <= prev_pos[1]:  # No northward progress
                if not in_stuck:
                    in_stuck = True
                    stuck_start = si - 1
            else:
                if in_stuck:
                    stuck_periods.append((stuck_start, si - 1, si - 1 - stuck_start))
                    in_stuck = False
        prev_pos = pos

        # Track worker
        has_worker = any(v[0] == 2 for v in my.values())
        if has_worker:
            worker_alive_until = si

        # Track mines
        my_mines = sum(1 for v in obs.get("mines", {}).values() if v[2] == 0)
        if my_mines > 0 and mine_built_step is None:
            mine_built_step = si
            events.append(f"  Step {si}: Mine built at gap={gap}, factory E={energy:.0f}")

        # Track jump (position jumps 2+ rows north in 1 step)
        if si > 0:
            prev_obs = steps[si-1][0].observation
            prev_factory = next((v for v in prev_obs.get("robots", {}).values() if v[4] == 0 and v[0] == 0), None)
            if prev_factory and r - prev_factory[2] >= 2:
                jump_steps.append(si)
                events.append(f"  Step {si}: JUMP from ({prev_factory[1]},{prev_factory[2]}) to ({c},{r}), gap={gap}")

        # Track south movement
        if si > 0 and prev_factory and r < prev_factory[2]:
            south_steps.append(si)

    if in_stuck:
        stuck_periods.append((stuck_start, factory_death, factory_death - stuck_start))

    # Energy analysis
    if factory_energies:
        final_e = factory_energies[-1][1]
        min_e = min(e for _, e in factory_energies)
        print(f"\n  Energy: start={factory_energies[0][1]:.0f} min={min_e:.0f} final={final_e:.0f}")

    print(f"  Max gap: {max_gap}")
    print(f"  Worker alive until: step {worker_alive_until}" if worker_alive_until else "  No worker")
    print(f"  Mine built: step {mine_built_step}" if mine_built_step else "  No mine built")

    # Stuck periods
    if stuck_periods:
        print(f"\n  Stuck periods (no northward progress):")
        for start, end, dur in stuck_periods:
            if dur >= 4:
                gap_at_start = None
                gap_at_end = None
                for si2 in range(start, min(end + 1, factory_death)):
                    obs2 = steps[si2][0].observation
                    f2 = next((v for v in obs2.robots.values() if v[4] == 0 and v[0] == 0), None)
                    if f2:
                        g = f2[2] - obs2.southBound
                        if gap_at_start is None:
                            gap_at_start = g
                        gap_at_end = g
                # Get position range
                obs_s = steps[start][0].observation
                f_s = next((v for v in obs_s.robots.values() if v[4] == 0 and v[0] == 0), None)
                obs_e = steps[min(end, factory_death-1)][0].observation
                f_e = next((v for v in obs_e.robots.values() if v[4] == 0 and v[0] == 0), None)
                pos_s = f"({f_s[1]},{f_s[2]})" if f_s else "?"
                pos_e = f"({f_e[1]},{f_e[2]})" if f_e else "?"
                print(f"    Step {start}-{end} ({dur}t): gap {gap_at_start}->{gap_at_end} | pos {pos_s}->{pos_e}")

    # Key events
    if events:
        print(f"\n  Key events:")
        for e in events:
            print(e)

    # Jumps
    print(f"\n  Jumps: {len(jump_steps)} total at steps {jump_steps}")
    print(f"  South movements: {len(south_steps)}")

    # Last 20 steps detail
    print(f"\n  Last 20 steps before death:")
    start = max(0, factory_death - 20)
    for si in range(start, factory_death):
        obs = steps[si][0].observation
        robots = obs.get("robots", {})
        factory = next((v for v in robots.values() if v[4] == 0 and v[0] == 0), None)
        if factory is None:
            break
        c, r, energy = factory[1], factory[2], factory[3]
        gap = r - obs.southBound

        n_wall = "W" if not agent_v1.can_go(obs, config, c, r, "NORTH") else "_"
        e_wall = "W" if not agent_v1.can_go(obs, config, c, r, "EAST") else "_"
        w_wall = "W" if not agent_v1.can_go(obs, config, c, r, "WEST") else "_"
        s_wall = "W" if not agent_v1.can_go(obs, config, c, r, "SOUTH") else "_"
        walls = f"N{n_wall}E{e_wall}W{w_wall}S{s_wall}"

        stuck = agent_v1.STATE.get("factory_stuck", 0) if si == factory_death - 1 else "?"
        print(f"    {si:3d}: ({c:2d},{r:3d}) gap={gap:2d} E={energy:6.0f} {walls} stuck={stuck}")


for seed in SEEDS:
    analyze_loss(seed)
