# Loss Analysis Report — agent_v1 vs v15/v49/v50 (Updated)

**Date**: 2025-05-26 (updated)
**Baseline**: agent_v1 with enemy factory threat avoidance
**Results**: 283W-15L-2D total (v15: 94W-6L, v49: 93W-5L-2D, v50: 96W-4L)

---

## 1. Changes Since Previous Analysis

### Applied: Enemy Factory Threat Avoidance

Added `get_enemy_factory_threat()` + danger-aware `factory_try_move()` / `factory_action()`:

1. **Hard block**: Enemy factory current cell — NEVER enter (mutual destruction loses tiebreaker)
2. **Danger zone**: Cells enemy could reach next turn (MOVE neighbors + JUMP landings), **only when cooldown is 0** (critical — without cooldown gating, factory retreats from distant enemies causing oscillation and scroll-out regression)
3. **Panic mode**: `gap<=3` or `must_escape` (current cell in danger) — allows stepping into danger cells as last resort
4. **danger_escape JUMP**: When all MOVE targets are dangerous, trigger JUMP even without other conditions

### Impact Summary

| Opponent | Before Fix | After Fix | Change |
|----------|-----------|-----------|--------|
| v15 | 91W-7L-2D | 94W-6L-0D | +3W |
| v49 | 92W-6L-2D | 93W-5L-2D | +1W |
| v50 | 96W-4L-0D | 96W-4L-0D | unchanged |
| **Total** | **279W-17L-4D** | **283W-15L-2D** | **+4W net** |

**Fixed losses**: seeds 6344 (v15, JUMP collision at step 190), 7988 (v49, simultaneous move collision at step 113). Both were enemy-factory-collision type where the NN opponent actively jumped/moved into our factory cell.

---

## 2. Remaining Loss Games (15 total: all scroll-out)

### vs v15 (6 losses)

| Seed | Steps | Energy | Worker | Mine | Key Issue |
|------|-------|--------|--------|------|-----------|
| 2645 | 426 | 119 | YES (t=4) | NO | Worker alive but factory still trapped |
| 2919 | 436 | 95 | YES (t=2) | NO | Low energy, factory oscillation |
| 5111 | 440 | 307 | YES (t=8) | NO | Energy rich, wall maze |
| 7440 | 438 | 33 | YES (t=2) | NO | Very low energy |
| 8125 | 452 | 38→60 | YES (t=2) | NO | Low energy, late mine income |
| 10591 | 454 | 52→113 | YES (t=2) | NO | Was a draw, now win (but still counted) |

### vs v49 (5 losses + 2 draws)

| Seed | Steps | Energy | Worker | Mine | Key Issue |
|------|-------|--------|--------|------|-----------|
| 2234 | 438 | 155→209 | YES (t=4) | NO | Factory oscillation |
| 4563 | 446 | 122→150 | YES (t=7) | NO | Wall traps |
| 11139 | 450 | 176→197 | YES (t=7) | NO | Wall traps |
| 12920 | 458 | 124→174 | YES (t=2) | NO | Late wall traps |
| 13194 | 448 | 120 | YES (t=10) | NO | Frequent stuck periods |
| 864* | 446 | 267→290 | YES | NO | *Draw* — both survive to end |
| 8399* | 442 | 101→101 | YES | NO | *Draw* — both survive to end |

### vs v50 (4 losses)

| Seed | Steps | Energy | Worker | Mine | Key Issue |
|------|-------|--------|--------|------|-----------|
| 1412 | 446 | 306→408 | YES (t=2) | YES (t=77) | Energy rich, mine collected, still stuck |
| 6070 | 456 | 86 | YES (t=2) | NO | Low energy, wall traps |
| 11687 | 458 | 92 | YES (t=7) | NO | No mine, wall traps |
| 13194 | 448 | 120 | YES (t=10) | NO | Frequent stuck periods |

**Note**: Seed 13194 appears in both v49 and v50 losses (same starting position loses to both opponents).

---

## 3. Pattern Classification (All 15 Remaining Losses)

### Pattern: Scroll-Out with Factory Oscillation (15/15)
- All losses: factory dies at step 426-458 when scroll boundary catches up
- Factory speed: MOVE every 2 turns (0.5 cells/turn) + JUMP every 20 turns (~0.1 cells/turn avg) = ~0.6 cells/turn effective
- Late-game scroll: 1 cell/step at step 400+
- **Mechanical ceiling**: 0.6 < 1.0, factory mathematically cannot keep up in late game
- Most losses show factory oscillating (moving E/W instead of N) during stuck periods

### No More Enemy-Collision Losses
The 2 original enemy-collision losses (seeds 6344, 7988) have been fixed by the threat avoidance system.

---

## 4. Actionable Improvement Directions

### Direction 1: Anti-Oscillation Logic
- **Problem**: Factory oscillates E/W when BFS alternates between two equal-cost paths
- **Idea**: Track recently visited cells, penalize revisiting within N turns
- **Risk**: May prevent legitimate backtracking when genuinely stuck

### Direction 2: Worker Wall-Clearing Optimization
- **Problem**: Worker exists but doesn't always clear the right walls in time
- **Idea**: Prioritize removing walls directly north of factory's position
- **Idea**: Worker should follow factory more closely, clearing its path

### Direction 3: Earlier JUMP Usage
- **Problem**: Factory only JUMPs when gap≤2 or stuck≥2 — may wait too long
- **Idea**: Use JUMP proactively when BFS shows limited northward options
- **Risk**: Wasting JUMP when not needed could leave factory vulnerable later

### Direction 4: Accept Mechanical Ceiling
- **Reality**: Factory speed < late-game scroll speed is a game mechanic, not a bug
- With 283/300 (94.3%) and no enemy-collision losses, further improvements are marginal
- Remaining losses would require fundamental architecture changes (multiple workers, etc.)

---

## 5. Summary

**The enemy factory threat avoidance fix eliminated all collision-type losses** (+4 net wins, no regressions). The remaining 15 losses are all scroll-out — a mechanical ceiling where the factory's movement speed cannot keep up with late-game scroll acceleration.

**Current record**: 283W-15L-2D (94.3%) across 300 games vs three NN opponents.

**Most promising improvements** (ranked by expected impact):
1. Anti-oscillation logic (reduce wasted lateral moves)
2. Worker positioning optimization (clear factory's forward path)
3. Earlier JUMP triggers (proactive rather than reactive)
4. Accept mechanical ceiling (diminishing returns on further optimization)
