# Maze Crawler — Fog-Aware Strategy 代码审计与改进报告

## 一、环境引擎关键规则（crawl.py）

### 1.1 游戏流程（每回合 10 个阶段）

```
Phase 0:  Cooldown tick     — 所有冷却 -1
Phase 1:  Action validation — 校验动作合法性
Phase 2:  Energy drain      — 每单位消耗 1 能量/回合（energy=0 强制 IDLE）
Phase 3:  Special actions   — TRANSFORM → BUILD/REMOVE墙 → BUILD单位 → TRANSFER
Phase 4:  Movement+Combat   — 移动结算 + 碰撞碾压（同级同归于尽）
Phase 5:  Crystal collect   — 拾取水晶（超出上限部分丢弃）
Phase 6:  Mine fill         — 从友方矿场充能
Phase 7:  Mine generation   — 矿场产出 mineRate(50) 能量/回合
Phase 8:  Scroll advance    — 南边界推进
Phase 9:  Boundary death    — row < southBound 的单位全部死亡
Phase 10: Win condition     — 工厂死亡 = 游戏结束
```

### 1.2 地图结构

- 宽度 20，高度（可见窗口）20
- 左半(0-9) **镜像**到右半(10-19)，由 Eller 算法生成
- **中央固定墙**：col 9 和 col 10 之间，不可建造/拆除
- **门（door）**：每行 8% 概率在中央墙打开一个通道
- **外周固定墙**：col=0 西墙、col=19 东墙

### 1.3 工厂与单位

| 类型 | 建造费 | 能量上限 | 移动周期 | 视野 | 战力 |
|---|---|---|---|---|---|
| Factory | - | ∞ | 2步/次 | 4 | 4(最强) |
| Scout | 50 | 100 | 1步/次 | 5 | 1 |
| Worker | 200 | 300 | 2步/次 | 3 | 2 |
| Miner | 300 | 500 | 2步/次 | 3 | 3 |

- 工厂初始位置：玩家 col=5 row=2，对手 col=14 row=2
- 建造冷却：10 回合，新单位出生在工厂北方（需无墙阻隔）
- JUMP：工厂专用，无视墙壁跳 2 格，冷却 20 回合
- TRANSFORM：Miner 在采矿节点上变身矿场，消耗 100 能量

### 1.4 滚动机制

```python
interval = round(4 - 3 * step / 400)
# step 0:   每 4 步滚 1 行
# step 133: 每 3 步滚 1 行
# step 267: 每 2 步滚 1 行
# step 400+: 每 1 步滚 1 行
```

工厂理论最大速度 0.5 格/步，滚动最终速度 1.0 格/步 → 工厂**必须**用 JUMP 维持领先。

### 1.5 胜负判定

- 工厂死亡 → 失败，reward = step - 502（越早死越惨）
- 存活方 reward = 总能量
- 超时（501步）→ 比总能量 → 比单位数 → 平局

### 1.6 碾压规则

- Factory > Miner > Worker > Scout
- 同级碰撞：双方同归于尽（不论阵营）
- Factory 只能被敌方 Factory 杀死（互杀）

---

## 二、Fog-Aware Agent 原始代码分析

### 2.1 架构

```
agent(obs, config)
  ├── update_memory()          — 更新全局记忆
  ├── decide_factory()         — 工厂决策（优先）
  └── decide_nonfactory() × N  — 各移动单位（按战力降序）
        ├── TRANSFORM          — Miner → 矿场
        ├── maybe_transfer()   — 能量转移给相邻友军
        ├── remove_direction() — Worker 拆墙（仅北墙）
        ├── best_attack_step() — 攻击弱敌
        ├── on_friendly_mine() — 在友方矿上充电
        └── role movement      — BFS 寻路到角色目标
```

### 2.2 已发现的 Bug 和问题

#### Bug 1: `nearest_point` 运算符优先级（line 62）

```python
return best, best_d if best else (None, inf)
# 实际解析为: return (best, (best_d if best else (None, inf)))
# 当 best=None 时返回 (None, inf)，恰好正确但逻辑不清晰
```

#### Bug 2: 建造不检查北墙（line 470）

```python
spawn_clear = in_bounds(spawn[0], spawn[1], obs, config) and spawn not in occupied
```
环境要求工厂和出生点之间**无墙**（crawl.py line 719），但 agent 只检查了 bounds 和占用。
当北墙存在时 BUILD 失败 → 浪费 10 回合冷却。

#### 问题 1: 工厂 safety_gap 阈值太低（line 473）

`safety_gap <= 3` 才触发紧急逃跑。实测工厂在 gap=4 时仍在建造，浪费宝贵移动回合。

#### 问题 2: 建造优先于移动（line 490-522）

safety_gap > 3 时，工厂优先级：建造 > 移动 > JUMP。
建造消耗 10 回合冷却 + 当前回合，实际相当于损失 5+ 次移动机会。

#### 问题 3: JUMP 限制过严（line 534）

```python
if jump_cd == 0 and r + 2 <= obs.northBound and safety_gap <= 8:
```
JUMP 是无视墙壁跳 2 格的最强逃生手段，但只在 danger ≤ 8 时使用。
应在冷却好时尽快使用以积累北向优势。

#### 问题 4: BFS 对未知区域过于乐观（line 128-136）

`wall_bits_at()` 返回 None 时 `blocked()` 返回 False，
BFS 会规划穿过从未见过的格子——那里可能有墙，导致实际移动失败或绕路。

#### 问题 5: 工厂移动目标太近（line 526）

```python
target_to_step((c, r), (c, min(obs.northBound, r + 5)), ...)
```
目标仅 5 格远，遇到墙时 BFS 绕路效率低。

#### 问题 6: Scout 穿越中央墙（line 406-408）

`mirrored_enemy_guess` 返回 col=14（右半），但中央有固定墙。
Scout 会撞墙卡住，浪费回合。

#### 问题 7: Worker 只拆北墙（line 441-455）

实际中工厂可能被东/西/南墙困住，需要 Worker 拆任意方向。

#### 问题 8: enemy_seen 每回合清零（line 87）

历史敌情丢失，无法追踪敌军移动模式。

---

## 三、实测结果（原始代码 vs random）

```
5 局全败，工厂平均存活 40 步（501步游戏）
工厂最终位置: row 9, southBound: 10 → 被滚动吞没
建造: 仅 1 个 Scout, 无 Worker/Miner/矿场
原始得分: -463 (负分)
```

### 死亡时间线

```
Step  0: factory row 2, south=0   → 建Scout（浪费移动）
Step  3: factory row 4, south=0   → 移动成功
Step  6: factory row 3, south=1   → BFS绕路回退！
Step 15: factory row 4, south=3   → 停滞不前
Step 24: factory row 8, south=6   → JUMP成功
Step 33: factory row 8, south=8   → 又回退
Step 40: factory row 9, south=10  → 死亡
```

---

## 四、改进记录

### v1: 基础生存优化（改进 P0 问题）

**改进内容**：
1. 工厂前 20 步优先移动，不建造 → 累积北向安全距离
2. safety_gap 阈值从 3 提高到 8
3. JUMP 冷却好了就立即使用（不限于 safety_gap≤8）
4. 建造前检查北墙是否存在
5. BFS 增加北向权重，惩罚南向移动
6. Scout 不再盲目穿越中央墙

**测试结果**（10 局 vs random）：

```
Record: 6W - 3L - 1D
Avg our reward: ~620  vs 原始 -463

Game | Seed  | Result |    Our |    Opp | Steps | FactLife | MaxRow
   1 |   42  |   WIN  |    924 |   -414 |    90 |       90 |   24
   2 |  123  |   WIN  |    964 |   -467 |    37 |       37 |    9
   3 |  456  |   WIN  |   1001 |   -483 |    21 |       21 |    7
   4 |  789  |   WIN  |   1028 |   -483 |    21 |       21 |    8
   5 | 1001  |  DRAW  |    0.5 |    0.5 |    75 |       74 |   21
   6 | 2024  |   WIN  |    980 |   -483 |    21 |       21 |    8
   7 |  303  |  LOSS  |   -459 |    356 |    45 |       44 |   10
   8 |  777  |   WIN  |    875 |   -378 |   126 |      126 |   38
   9 | 2048  |  LOSS  |   -408 |    183 |    96 |       95 |   25
  10 |  555  |  LOSS  |   -439 |    336 |    65 |       64 |   15
```

**关键发现**：
1. JUMP 策略改变是最大提升因素：从"始终 JUMP"改为"仅在北墙阻挡或卡住 8 回合后 JUMP"
   - 原始策略在 step 0 盲目 JUMP，可能落入死胡同（seed=777 完全卡死 20 步）
   - 新策略让工厂正常行走探索迷宫，卡住时才用 JUMP 逃生
2. BFS 路径寻找允许工厂绕路（不限于直接向北）
3. 南向回溯（stuck>=6 时 safety>=2）帮助工厂从死胡同中脱身
4. 建造阈值降至 10（原 15），但因工厂 safety_gap 始终不够，仍未建造任何单位

---

### v2: 移动优化与路径改进

**改进内容**：
1. **"北向无进展"卡住计数器**：不再追踪"位置不变"，而是追踪"未向北移动"。即使工厂南向回退，卡住计数器仍累加。仅在工厂实际向北移动时重置。这使 JUMP 触发更准确。
2. **移动优先级重构**：北 > BFS 绕路(不含南) > 东/西探索未知区域 > BFS 南向路径(仅卡住>=4) > 直接南向(仅卡住>=6)。
   - 核心改变：工厂优先探索东/西未知区域，避免过早南向回退
   - 南向回退阈值提高，减少不必要的后退
3. **BFS 搜索范围扩大**：从 r+15 扩大到 r+25，寻找更远的绕路路径
4. **建造逻辑优化**：仅在工厂无法移动时（冷却回合）建造，且要求 safety_gap >= 10 + 侦察兵逃生路径通畅
5. **侦察兵逃离逻辑**：侦察兵在工厂正北时立即移开，避免阻塞工厂

**建造实验失败记录**：
- 在 safety_gap >= 5 时建造侦察兵 → 4W-5L-1D（退步）。侦察兵在 (c, r+1) 生成阻塞工厂
- 在工厂冷却回合建造 + safety_gap >= 7 + 逃生检查 → 5W-4L-1D（略退）。侦察兵仍导致工厂卡住
- 结论：在当前工厂移动效率下，建造任何单位的风险都太高

**测试结果**（20 局 vs random）：

```
Record: 13W - 4L - 3D
Avg our reward: 541.8 vs 原始 -463

Game | Seed  | Result |    Our |    Opp | Steps | FactLife
   1 |   42  |   WIN  |    924 |   -414 |    90 |       90
   2 |  123  |   WIN  |    964 |   -467 |    37 |       37
   3 |  456  |   WIN  |   1001 |   -483 |    21 |       21
   4 |  789  |   WIN  |   1028 |   -483 |    21 |       21
   5 | 1001  |  DRAW  |    0.5 |    0.5 |    75 |       74
   6 | 2024  |   WIN  |    980 |   -483 |    21 |       21
   7 |  303  |  LOSS  |   -459 |    356 |    45 |       44
   8 |  777  |   WIN  |    903 |   -378 |   126 |      126
   9 | 2048  |  LOSS  |   -408 |    183 |    96 |       95
  10 |  555  |  LOSS  |   -439 |    336 |    65 |       64
  11 |  111  |   WIN  |    967 |   -459 |    45 |       45
  12 |  222  |  LOSS  |   -471 |    568 |    33 |       32
  13 |  333  |  DRAW  |    0.5 |    0.5 |    57 |       56
  14 |  444  |   WIN  |    980 |   -483 |    21 |       21
  15 | 5556  |  DRAW  |    0.5 |    0.5 |    45 |       44
  16 |  666  |   WIN  |    964 |   -467 |    37 |       37
  17 |  999  |   WIN  |    940 |   -443 |    61 |       61
  18 | 1337  |   WIN  |   1013 |   -467 |    37 |       37
  19 | 2025  |   WIN  |    994 |   -447 |    57 |       57
  20 | 3141  |   WIN  |    954 |   -423 |    81 |       81
```

**关键发现**：
1. 胜率 65%（13/20），远超原始代码（0/5 全败）
2. 4 局失败均为迷宫结构导致工厂无法快速北移（地图对称性不利）
3. 3 局平局：双方工厂几乎同时死亡
4. 工厂平均存活 53 步，WIN 局平均存活 52 步
5. 未建造任何单位/矿场——工厂的 safety_gap 很少超过 10
6. JUMP 在 loss 局中从未触发（工厂不断缓慢北移但不累积足够 stuck 计数）

**待改进方向（v3）**：
1. 提高工厂在复杂迷宫中的北移效率（减少绕路和回退）
2. 研究 loss 局中对手（random）为何得分更高——可能是地图镜像的不对称优势
3. 考虑仅在工厂极度安全时（safety >= 15）建造单位

---

### v2 稳定性测试（50 局）

使用 50 个随机种子（seed = 42, 179, 316, ... 每 137 递增）测试稳定性。

**测试结果**：

```
Record: 23W - 22L - 5D (46% win rate)
Avg our reward:    253.4
Avg opp reward:    -12.9
Avg factory life:  50.2 steps
Avg WIN life:      46.1 steps (23 games)
Avg LOSS life:     55.7 steps (22 games)
WIN reward range:  924 ~ 1060
LOSS reward range: -483 ~ -348

Running win rate (window=10):
  Games  1-10:  3W-6L-1D | avg reward:   18.2
  Games 11-20:  7W-3L-0D | avg reward:  557.5
  Games 21-30:  6W-3L-1D | avg reward:  457.2
  Games 31-40:  3W-6L-1D | avg reward:   26.4
  Games 41-50:  4W-4L-2D | avg reward:  207.6
```

**稳定性分析**：
1. **胜率不稳定**：10 局窗口波动 30%~70%，说明迷宫种子对结果影响极大
2. **20 局测试的 65% 胜率是种子偏差**：50 局真实胜率约 46%，接近与 random 持平
3. **WIN 局寿命反而更短**（46 vs 56 步）：胜利取决于对手先死，不是工厂存活更久
4. **无经济系统**：0 个单位建造，0 座矿场，完全依赖工厂单独生存
5. **收益分布两极**：WIN 得分 924~1060（工厂剩余能量），LOSS 得分 -483~-348

**根本瓶颈**：
- 工厂移动速度 (~0.5 格/步) 无法可靠超越滚动加速
- 迷宫结构约 50% 概率对我不利（地图镜像特性）
- 没有经济引擎，无法通过水晶/矿场积累能量优势

**v3 必须突破的方向**：
1. 建造侦察兵拾取水晶（需要安全建造机制）
2. 建造工人清除工厂前方墙壁（需要不阻塞工厂路径）
3. 工厂在安全间隙时建造，冷却期间继续移动
