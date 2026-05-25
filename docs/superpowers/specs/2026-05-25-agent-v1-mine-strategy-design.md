# Agent V1 Mine Strategy Redesign

## Overview

Redesign agent_v1's factory decision logic to leverage the mining economy. The core idea: use scouts to discover mining nodes, invest in miners when ROI is positive, and collect mine energy to fund continued survival.

## Strategy Phases

### Phase 1: Scout Exploration + Mine Discovery
- Scouts explore ahead of factory (existing behavior)
- `update_state` already records mining nodes to `STATE["nodes"]`
- No changes to scout behavior

### Phase 2: Mine Investment (when ROI > 700 energy)
- Factory BFS target includes the nearest reachable mining node
- Factory builds Miner when it reaches the node
- Miner walks to node and TRANSFORMs (takes ~3 turns: build + move + transform)
- Factory stands on mine to collect energy (50/turn, unlimited while gap safe)
- Leaves only when gap <= 2 (southern boundary catching up)

### Phase 3: Pure Survival (when ROI insufficient)
- No mine investment
- Keep 1 Worker for wall clearing
- Factory moves north at full speed

## ROI Calculation

```
scroll_interval = max(1, 4 - 3 * step / 400)
dist = BFS_distance(factory, mine_node)
turns_to_reach = dist * 2  # factory moves every 2 turns
gap_at_arrival = current_gap - ceil(turns_to_reach / scroll_interval)
stay_turns = gap_at_arrival - 2  # 2-row safety margin
# Account for 3-turn setup overhead (build + move + TRANSFORM)
effective_stay = max(0, stay_turns - 3)
expected_output = effective_stay * 50
ROI_OK = expected_output >= 700

# Filter out nodes that have scrolled behind the factory
candidate_nodes = [n for n in STATE["nodes"] if n[1] >= factory_row and in_bounds(n[0], n[1])]
```

If `ROI_OK`, the nearest mining node becomes a BFS priority target. If no node passes ROI, factory defaults to pure northward movement.

Note: 700 is the investment threshold, not a departure limit. Once a mine is built, factory stays until gap forces it to leave, collecting up to the mine's 1000 cap.

## BFS Target Fusion (Approach A)

Instead of a state machine, the factory's BFS goals naturally include mining nodes:

```python
goals = [(c, row + 2) for c in range(width)]  # existing northward targets
if mine_target and roi_ok(mine_target, step, gap):
    goals = [mine_target] + goals  # mine node takes priority
```

Factory naturally drifts toward the mine while still heading north. When mine ROI drops below threshold, factory seamlessly reverts to pure northward movement.

## Build Decision Logic

```
if build_cd == 0 and gap >= 2 and spawn cell clear:
    if own mine nearby and gap > 2:
        # Don't build, stay and collect energy
        pass
    elif mine_target reached and no miner en route:
        BUILD_MINER  (cost 300)
    elif worker_count < 1 and energy >= 250:
        BUILD_WORKER  (cost 200 + 50 reserve)
```

Key changes from current logic:
- Adds BUILD_MINER (currently missing)
- Adds BUILD_SCOUT is NOT included (rely on initial scouts)
- Removes the `energy >= 300` worker threshold, uses `energy >= 250` instead
- Mine investment takes priority over Worker when at a valid mine

## Factory at Mine: Wait and Collect

```
my_mines_nearby = [m for m in my_mines if manhattan(factory, m) <= 1]
if my_mines_nearby:
    mine = my_mines_nearby[0]
    if gap <= 2:
        resume northward movement
    else:
        move onto mine cell or IDLE to collect
```

Factory stays at the mine as long as gap > 2 (safe). No upper limit on energy collection — mine cap is 1000, everything beyond the 700 ROI threshold is pure profit. The only reason to leave is when the southern boundary catches up.

When factory leaves (gap <= 2), set `STATE["mine_invested"] = None`.

## Miner Behavior (unchanged)

Existing `miner_action` already handles:
1. Stand on mining node + enough energy → TRANSFORM
2. BFS to nearest mining node
3. Follow factory as fallback

No changes needed. Timing is correct: TRANSFORM (step 4) happens before movement (step 5), so miner must be on the node from a previous turn.

## Worker Behavior (minimal change)

- Keep existing wall-clearing priority
- Keep existing follow-factory behavior
- Only change: factory build threshold from `energy >= 300` to `energy >= 250`

## Scout Behavior (no change)

- Continue exploring ahead of factory
- Collect crystals opportunistically
- No mine-related changes needed

## Execution Order (no change)

Scout → Worker → Miner → Factory (non-factory units first for collision avoidance)

## Scroll Mechanics Reference

- Start: every 4 turns
- Ramp: linearly over 400 steps
- End: every turn from step 400 onward
- Formula: `scroll_interval = max(1, 4 - 3 * step / 400)`
- Game ends at step 500

## New State Fields

```python
STATE = {
    "turn": 0,
    "nodes": set(),            # known mining nodes (existing)
    "last_factory_pos": None,  # existing
    "factory_stuck": 0,        # existing
    "walls": {},               # existing
    "mine_invested": None,     # (col, row) of mine target; set when factory commits to a mine node, cleared when leaving (gap <= 2)
}
```

## Summary of Changes to agent_v1.py

1. **factory_action**: Add mine ROI calculation, BFS target fusion, BUILD_MINER, mine collection wait logic
2. **worker_action**: No functional changes (build threshold change is in factory_action)
3. **scout_action**: No changes
4. **miner_action**: No changes
5. **STATE**: Add `mine_invested` field
6. **Remove**: Dead code (`has_nodes` variable, duplicate IDLE in worker_action lines 339-343)
