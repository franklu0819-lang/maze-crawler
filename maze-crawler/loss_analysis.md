# Loss Analysis Report — agent_v1 vs v50 (Updated)

**Date**: 2025-05-25 (updated)
**Baseline**: agent_v1 (optimized mine logic, worker forward wall clear, pessimistic BFS)
**Results**: 96W-4L-0D vs v50 (100 games)

---

## 1. Changes Since Previous Analysis

### Applied Optimizations
1. **Optimized mine approach**: Factory navigates to `(mc, mr-1)` before building miner, so miner spawns directly ON the mining node and TRANSFORMs in 1 turn
2. **Removed mine-idle scout build**: Mechanically impossible — miner's `build_cd=10` covers all BUILD turn windows while factory collects mine energy
3. **Worker wall clear range**: NORTH expanded to 4 rows ahead of factory
4. **Worker stuck-assist**: Worker navigates to factory's north cell when factory stuck ≥ 2
5. **Pessimistic BFS**: Factory uses pessimistic pathfinding (unknown = wall) when stuck ≥ 2

### Impact Summary

| Seed | Before | After | Change |
|------|--------|-------|--------|
| 1412 | E=119, life=439, mine=wasted | E=**408**, life=**446**, mine=collected | +7 steps, mine now works |
| 6070 | life=456 | life=456 | No change |
| 11687 | life=436, mine=wasted | life=**458**, no mine | +22 steps, skipped useless mine |
| 13194 | life=448, mine=wasted | life=448, no mine | Skipped useless mine |

---

## 2. Loss Game Summary (Current)

### Seed 1412 (mine works, energy rich, still dies)
- **Death**: step 445, gap=0, E=408
- **Worker**: dies at step 83
- **Mine**: built at (4,22) t=77, factory ON mine 12 turns, collected +539 energy
- **Stuck periods** (≥4t): 7 periods
  - t=79-89 (11t): factory collecting mine energy
  - t=121-232: 5 more periods of 4t each
- **Root cause**: After mine collection, factory has plenty of energy but still gets trapped by walls. Worker died at t=83 (early), no wall-clearing help for remaining 362 steps.

### Seed 6070 (no mine, low energy)
- **Death**: step 455, gap=0, E=86
- **Worker**: dies at step 3 (!) — almost immediately
- **Mine**: none
- **Stuck periods**: only 2 (t=140-143, t=182-185)
- **Root cause**: Worker dies at step 3, leaving factory alone for 452 steps with no wall-clearing. Despite long survival, cumulative wall traps eventually catch up. Low energy at death (86) suggests factory barely sustains itself.

### Seed 11687 (no mine, late stuck)
- **Death**: step 457, gap=0, E=92
- **Worker**: dies at step 0 — never existed or died before tracking
- **Mine**: none (optimized logic correctly skipped it)
- **Stuck periods**: 3, all late (t=305-321)
- **Root cause**: Factory survives remarkably long (457 steps) with no worker or mine. Late-game stuck periods (t=305+) with faster scroll speed prove fatal. No wall-clearing capability at all.

### Seed 13194 (no mine, many stuck periods)
- **Death**: step 447, gap=0, E=120
- **Worker**: dies at step 0 — never existed or died before tracking
- **Mine**: none (optimized logic correctly skipped it)
- **Stuck periods**: 12 periods — highest of all losses
  - Early (t=5-8), mid (t=118-263), late (t=285-362)
- **Root cause**: Factory stuck frequently throughout the game. 12 stuck periods indicate factory is navigating through particularly maze-dense terrain. No wall-clearing help.

---

## 3. Pattern Classification (Updated)

### Pattern A: Worker Dies Extremely Early (3/4 losses: seeds 6070, 11687, 13194)
- Worker dies at step 0-3, factory alone for 447-455 steps
- No wall-clearing for entire game
- Factory relies entirely on BFS pathfinding through unknown maze

### Pattern B: Energy-Rich but Stuck (1/4 losses: seed 1412)
- Factory has 408 energy at death — more than enough to build units
- But `build_cd` and `move_cd` timing prevents building during mine collection
- After leaving mine, factory gets stuck repeatedly without worker (died at t=83)

### Key Insight: Worker Longevity is the Primary Factor
- In 3/4 losses, worker dies at step 0-3 (effectively never exists)
- In 1/4 loss, worker dies at step 83 (early-mid game)
- **Factory never has wall-clearing help for 80%+ of the game**

---

## 4. Scout Analysis

**Conclusion: Scout is not viable in current architecture.**

1. Mine-idle scout build: Mechanically impossible. Miner's `build_cd=10` covers all BUILD turn windows while factory collects mine energy. Once factory starts IDLE collection, `move_cd` stays at 0 forever — no BUILD turns.

2. Early scout build: Tested at turn ≥ 30 (after worker). Causes regression to 90W — 10-turn build CD delays critical builds.

3. Scout's value (terrain revelation) doesn't justify the build CD cost.

---

## 5. Actionable Improvement Directions

### Direction 1: Worker Survival (Highest Priority)
- **Problem**: Worker dies at step 0-3 in 3/4 losses
- **Idea**: Investigate WHY worker dies so early — enemy attack? Scroll boundary?
- **Idea**: Keep worker closer to factory (within 2-3 rows instead of exploring ahead)
- **Idea**: If worker dies, rebuild ASAP (lower energy threshold for rebuild when no worker exists)

### Direction 2: Factory Stuck Recovery
- **Problem**: Factory gets stuck in 4-12 periods per loss game
- **Idea**: When stuck ≥ 3 and no worker, accept south movement earlier (gap ≥ 2 instead of ≥ 3)
- **Idea**: When stuck, prioritize lateral movement over waiting for north opening
- **Idea**: Track recently visited cells to avoid oscillation patterns

### Direction 3: Dead-End Detection
- **Problem**: Factory enters dead-ends, gets stuck for 4+ turns
- **Idea**: Before moving, check if destination cell has at least 2 known exits
- **Idea**: Mark cells as "dead-end" after visiting and finding ≤ 1 exit
- **Idea**: Use wall memory to build a graph of known passable cells for smarter routing

### Direction 4: Worker Rebuild After Death
- **Problem**: After worker dies at step 0-3, factory has no wall-clearing for 440+ steps
- **Idea**: When worker_count == 0, lower worker build threshold (energy ≥ 300 instead of ≥ 400)
- **Idea**: Prioritize worker rebuild over mine investment when no worker exists
- **Risk**: Building worker at lower energy might leave factory energy-poor

---

## 6. Energy Distribution at Death

| Seed | Energy | Worker Alive | Mine | Stuck Periods | Key Issue |
|------|--------|-------------|------|---------------|-----------|
| 1412 | 408 | NO (t=83) | YES | 7 | Energy rich, no worker |
| 6070 | 86 | NO (t=3) | NO | 2 | Low energy, no worker |
| 11687 | 92 | NO (t=0) | NO | 3 | No worker ever |
| 13194 | 120 | NO (t=0) | NO | 12 | Many stuck, no worker |

**Key insight**: All 4 losses have NO worker at death. 3/4 had no worker for essentially the entire game. The worker is the critical unit for survival.

---

## 7. Summary

**The fundamental problem remains wall trapping, but the root cause is worker death.** In 3/4 loss games, the worker dies at step 0-3, leaving the factory to navigate the maze alone for 440+ steps. Without wall-clearing, the factory eventually gets trapped.

**The mine optimization successfully fixed seed 1412's energy collection** (E went from 119 → 408), confirming the optimized approach logic works. However, energy alone doesn't prevent wall trapping.

**Most promising improvements** (ranked by expected impact):
1. Investigate and fix early worker death (Direction 1)
2. Lower worker rebuild threshold when no worker exists (Direction 4)
3. Better stuck recovery without worker (Direction 2)
4. Dead-end avoidance (Direction 3)
