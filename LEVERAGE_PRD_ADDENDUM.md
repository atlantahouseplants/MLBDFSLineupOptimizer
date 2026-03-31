# Leverage Strategy PRD — Addendum: Contest & Slate Awareness

**Supplements:** `LEVERAGE_STRATEGY_PRD.md`  
**Date:** March 31, 2026  
**Priority:** Implement AFTER or ALONGSIDE the main PRD sections.

---

## 0. Agent Instructions

This document modifies and extends `LEVERAGE_STRATEGY_PRD.md`. Read that document first, then apply these changes. Where this addendum contradicts the main PRD, **this addendum takes priority.**

---

## 1. Remove Cash Mode — We Only Play Tournaments

### 1.1 What Changes

The main PRD describes a "Cash/Safe" mode. **Delete it.** The user never plays 50/50, double-up, or cash games. Every contest is a tournament (GPP or single-entry).

Replace the three-mode Strategy selector (`GPP Leverage / Cash / Custom`) with a **tournament-specific** mode system based on two dimensions:

1. **Contest Size** — How many entries are in the field
2. **Slate Size** — How many games are on the slate

These two dimensions combine to create the strategy profile for any given night.

### 1.2 Contest Size Modes

| Mode | Field Size | Strategic Approach |
|------|-----------|-------------------|
| **Single Entry** | 1 entry per user, any field size | Maximum "correct" lineup. Still leverage-aware but prioritize your single best projected-ceiling lineup. Less need for portfolio diversity since you're submitting one lineup. Ownership penalty is moderate — you want some differentiation but can't afford to get too cute with only one bullet. |
| **Small Field GPP** | Multi-entry, under 500 total entries | Moderate leverage. With fewer opponents, there's less duplication in the field. You don't need to be as contrarian to be unique. Play more "obviously good" stacks but still avoid the single highest-owned stack. Portfolio of 10-20 lineups with moderate diversity. |
| **Large Field GPP** | Multi-entry, 1000+ total entries | Maximum leverage. This is where the full contrarian strategy shines. In a 5000-entry contest, the chalk is massively duplicated. You need to be aggressively different. Portfolio of 20-150+ lineups with high diversity, strong ownership penalties, and heavy preference for underowned stacks. |

### 1.3 Updated LeverageConfig Presets

Replace the presets in Section 10.2 of the main PRD with these:

**"Single Entry Tournament":**
```json
{
  "mode": "single_entry",
  "ceiling_weight": 0.30,
  "ownership_penalty": 0.25,
  "boom_weight": 0.15,
  "chalk_threshold": 0.30,
  "chalk_extra_penalty": 0.08,
  "within_stack_chalk_penalty": 0.05,
  "target_avg_lineup_ownership": 0.12,
  "max_stack_exposure_gpp": 1.0,
  "max_batter_exposure_gpp": 1.0,
  "max_pitcher_exposure_gpp": 1.0,
  "min_viable_projection_pct": 0.35,
  "default_num_lineups": 1,
  "portfolio_diversity_weight": 0.0
}
```

*Rationale:* With one lineup, you want the single highest-ceiling, moderately leveraged lineup. You still avoid the absolute chalk (ownership_penalty is non-zero) but you can't afford wild swings. Ceiling weight is high because you need your one lineup to boom. No exposure limits since it's one lineup.

**"Small Field GPP" (under 500 entries):**
```json
{
  "mode": "small_field_gpp",
  "ceiling_weight": 0.25,
  "ownership_penalty": 0.30,
  "boom_weight": 0.10,
  "chalk_threshold": 0.25,
  "chalk_extra_penalty": 0.10,
  "within_stack_chalk_penalty": 0.08,
  "target_avg_lineup_ownership": 0.11,
  "max_stack_exposure_gpp": 0.45,
  "max_batter_exposure_gpp": 0.40,
  "max_pitcher_exposure_gpp": 0.60,
  "min_viable_projection_pct": 0.35,
  "default_num_lineups": 20,
  "portfolio_diversity_weight": 0.25
}
```

*Rationale:* In a 200-entry contest, maybe 30-40 people stacked the popular team. That's not catastrophic duplication. You want to be somewhat different but you can also play some chalk stacks. Moderate penalties across the board. You're trying to be in the top 5%, not the top 0.1%.

**"Large Field GPP" (1000+ entries):**
```json
{
  "mode": "large_field_gpp",
  "ceiling_weight": 0.30,
  "ownership_penalty": 0.55,
  "boom_weight": 0.15,
  "chalk_threshold": 0.20,
  "chalk_extra_penalty": 0.20,
  "within_stack_chalk_penalty": 0.15,
  "target_avg_lineup_ownership": 0.08,
  "max_stack_exposure_gpp": 0.30,
  "max_batter_exposure_gpp": 0.30,
  "max_pitcher_exposure_gpp": 0.50,
  "min_viable_projection_pct": 0.40,
  "default_num_lineups": 150,
  "portfolio_diversity_weight": 0.40
}
```

*Rationale:* In a 5000-entry contest, the top stack might have 800+ duplicates. You need to aggressively fade chalk. Ownership penalty is nearly double the small-field preset. Higher diversity weight ensures your 150 lineups cover many different stack configurations. You're hunting for first place, not just a cash spot — the only thing that matters is hitting the top.

### 1.4 Config Changes

In `config.py`, update the `LeverageConfig` dataclass. Remove the `"cash"` option from the `mode` field. Replace with:

```python
@dataclass
class LeverageConfig:
    # ... existing fields from main PRD ...

    mode: str = "large_field_gpp"
    # Valid modes: "single_entry", "small_field_gpp", "large_field_gpp"

    # New field: number of lineups to generate (varies by contest type)
    default_num_lineups: int = 20

    # New field: portfolio diversity weight for lineup selector
    portfolio_diversity_weight: float = 0.30
```

### 1.5 Dashboard Changes

Replace the Strategy Mode selector from Section 8.1 of the main PRD with:

```
Contest Type: [Single Entry] [Small Field GPP] [Large Field GPP]
```

Display the contest type prominently at the top. When selected, it auto-populates all LeverageConfig parameters. A "Custom" expander below allows manual override of any parameter.

Also add a field for contest entry count — when the user enters the number of entries in their target contest, auto-suggest the appropriate mode:
- 1 entry per person → Single Entry
- Under 500 field → Small Field GPP
- 500-999 → Small Field GPP (but suggest bumping ownership_penalty slightly)
- 1000+ → Large Field GPP

---

## 2. Slate Size Awareness

### 2.1 Why Slate Size Matters

The number of games on the slate fundamentally changes the strategy:

**Small Slate (2-3 games):**
- Only 20-35 batters and 4-6 pitchers available
- Everyone is forced into the same tiny player pool
- Ownership concentrates heavily — the "best" players might be 35-50% owned
- Game selection doesn't exist — you're playing all the games
- **Stacking IS the strategy.** The only way to differentiate is stack construction: which 4 hitters from each team, and how you combine stacks across the 2-3 games
- Bring-backs are critical because every hitter is in the same 2-3 games, so game correlation is unavoidable
- The chalky pitcher might be 45% owned — you almost certainly want to fade and play the contrarian pitcher
- **Key insight:** On small slates, everyone knows the "right" plays. The only edge is in unique combinations of known players.

**Medium Slate (5-8 games):**
- 50-90 batters, 10-16 pitchers
- This is the standard slate type where the full leverage strategy applies
- Enough games for meaningful game selection (fade some games, target others)
- Standard stacking with 2-3 distinct stack targets across the portfolio
- Ownership spreads out enough that true leverage plays exist
- The main PRD strategy applies here with no modifications

**Large Slate (9+ games, full MLB slate):**
- 100-140 batters, 18-28 pitchers
- Maximum game selection advantage — you can completely ignore 4-5 games
- Ownership is very spread out; even the chalk plays might only be 15-20%
- The ownership penalty should be slightly less aggressive because natural ownership dilution already creates differentiation
- More room for "deep leverage" plays — obscure players in low-profile games
- Pitcher selection becomes more impactful because there are more viable options
- **Key insight:** On large slates, game selection is the primary differentiator. Pick the right 2-3 games to target and you're already ahead of most of the field.

### 2.2 Slate Size Detection

Add automatic slate size detection based on the number of games in the optimizer dataset. In `dataset.py` or a new utility:

```python
@dataclass
class SlateProfile:
    num_games: int
    num_batters: int
    num_pitchers: int
    slate_type: str          # "small", "medium", "large"
    recommended_stacks: int  # How many distinct team stacks to target
    stack_templates: list     # Recommended stack shapes

def detect_slate_profile(optimizer_df: pd.DataFrame) -> SlateProfile:
    games = optimizer_df["game_key"].nunique()
    batters = len(optimizer_df[optimizer_df["player_type"].str.lower() == "batter"])
    pitchers = len(optimizer_df[optimizer_df["player_type"].str.lower() == "pitcher"])

    if games <= 3:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="small",
            recommended_stacks=games,  # Stack every game
            stack_templates=[(4, 4)] if games == 2 else [(4, 3, 1), (4, 4)],
        )
    elif games <= 8:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="medium",
            recommended_stacks=min(4, games - 1),
            stack_templates=[(4, 3, 1), (4, 2, 2), (3, 3, 2)],
        )
    else:
        return SlateProfile(
            num_games=games,
            num_batters=batters,
            num_pitchers=pitchers,
            slate_type="large",
            recommended_stacks=min(5, games // 2),
            stack_templates=[(4, 2, 2), (4, 3, 1), (4, 2, 1, 1), (3, 3, 2)],
        )
```

### 2.3 Slate-Specific Strategy Adjustments

When the slate profile is detected, automatically adjust `LeverageConfig` parameters. These adjustments layer ON TOP of the contest-size presets:

#### Small Slate Adjustments

```python
def apply_small_slate_adjustments(config: LeverageConfig) -> LeverageConfig:
    # On small slates, ownership concentrates heavily
    # We need to be MORE aggressive about fading chalk
    config.ownership_penalty *= 1.3   # 30% more aggressive on ownership
    config.chalk_threshold *= 0.8     # Lower the chalk threshold (20% -> 16%)
    config.chalk_extra_penalty *= 1.5 # Bigger penalty above chalk threshold

    # Pitcher leverage is critical on small slates
    # The "obvious" pitcher might be 40%+ owned — big edge in fading
    config.pitcher_fade_bonus = 0.20  # NEW: bonus for using non-chalk pitcher

    # Everyone knows the "right" batters on small slates
    # Differentiate through WHICH batters on each team, not which teams
    config.within_stack_chalk_penalty *= 1.5  # Stronger preference for underowned teammates

    # Stack templates: on 2-gamers, you MUST go 4-4 or close to it
    # On 3-gamers, 4-3-1 or 4-4 with a one-off
    config.min_primary_stack_size = 4  # NEW: primary stack must be 4 batters minimum

    # Bring-backs are almost mandatory on small slates
    config.bring_back_enabled = True
    config.bring_back_count = 1

    # Fewer viable lineup combinations exist, so lower the diversity pressure
    config.portfolio_diversity_weight *= 0.7

    return config
```

#### Medium Slate Adjustments

```python
def apply_medium_slate_adjustments(config: LeverageConfig) -> LeverageConfig:
    # Medium slates are the "standard" — the main PRD defaults work well
    # Only minor tweaks needed

    # Standard bring-back preference (not mandatory)
    config.bring_back_enabled = True
    config.bring_back_count = 1

    return config
```

#### Large Slate Adjustments

```python
def apply_large_slate_adjustments(config: LeverageConfig) -> LeverageConfig:
    # On large slates, natural ownership dilution does some of the work for us
    # Ownership is more spread out, so we can be slightly less aggressive
    config.ownership_penalty *= 0.85  # 15% less aggressive on ownership
    config.chalk_threshold *= 1.15    # Raise the chalk threshold slightly (20% -> 23%)

    # Game selection is the key differentiator on large slates
    # Boost the stack leverage bonus to really reward picking the right games
    config.stack_leverage_bonus *= 1.4  # 40% more reward for leverage stacks

    # More room for deep leverage — lower the minimum viable projection cutoff
    # to allow some boom-or-bust cheap players in obscure games
    config.min_viable_projection_pct *= 0.85  # Allow slightly lower-projected players

    # More pitcher options = more room for contrarian pitcher plays
    config.pitcher_fade_bonus = 0.10  # Moderate bonus for non-chalk pitchers

    # Higher diversity across the portfolio since there are more stack options
    config.portfolio_diversity_weight *= 1.2

    # Can run more stack templates since more games are available
    config.max_stack_exposure_gpp *= 0.85  # Tighter per-stack limits = more variety

    return config
```

### 2.4 New Config Field: `pitcher_fade_bonus`

Add to `LeverageConfig`:

```python
# Bonus applied to non-chalk pitchers in the objective function
# On small slates, the "obvious" pitcher can be 40%+ owned — fading them is high leverage
# This adds a flat bonus to pitchers whose ownership is below the top-2 pitcher average
pitcher_fade_bonus: float = 0.0  # Set by slate adjustments, not directly by user

# Minimum primary stack size (used on small slates to enforce real stacks)
min_primary_stack_size: int = 3  # Overridden to 4 on small slates

# Force bring-backs (small slate setting)
bring_back_enabled: bool = True
bring_back_count: int = 1
```

### 2.5 Pitcher Fade Logic in Solver

On small slates especially, the top 1-2 pitchers absorb massive ownership. Add pitcher-specific leverage logic to the solver:

```python
# In _compute_gpp_score, add pitcher-specific handling:
if player_type == "pitcher":
    # Identify the top-2 pitchers by ownership
    pitcher_pool = pool[pool["player_type"].str.lower() == "pitcher"]
    top2_avg_own = pitcher_pool["proj_fd_ownership"].nlargest(2).mean()

    if ownership < top2_avg_own * 0.6:
        # This pitcher is meaningfully less owned than the chalk pitchers
        score += config.pitcher_fade_bonus * (top2_avg_own - ownership)
```

This rewards non-obvious pitcher selections, especially on small slates where the top pitcher might be 40% owned.

---

## 3. Combined Strategy Matrix

The final strategy is determined by crossing contest size with slate size:

| | Small Slate (2-3) | Medium Slate (5-8) | Large Slate (9+) |
|---|---|---|---|
| **Single Entry** | Maximum ceiling. One big correlated stack. Fade the chalk pitcher. Bring-back mandatory. | Standard ceiling play. Best leverage stack available. Moderate ownership awareness. | Game selection is key. Find the single best leverage game and stack it hard. |
| **Small Field GPP** | Moderate leverage on stacks. 2-3 distinct stack configurations across 10-20 lineups. Pitcher diversity across lineups. | Balanced leverage. 3-4 distinct stacks. Standard PRD approach. | Wide game selection. 4-5 distinct stack targets. Moderate ownership penalty. |
| **Large Field GPP** | Aggressive pitcher fades. Every possible stack combination covered. Within-stack differentiation is everything. | Full leverage. 4-5+ distinct stacks. Heavy ownership penalties. | Maximum game selection leverage. 5+ distinct stacks across obscure games. Deep leverage on cheap players. |

### 3.1 Auto-Strategy Selection

When the user uploads their data files, the system should:

1. **Detect slate size** automatically from the number of games in the dataset
2. **Ask for contest type** via the dashboard selector (Single Entry / Small Field / Large Field)
3. **Auto-configure** all LeverageConfig parameters based on the combined strategy matrix
4. **Display the strategy summary** prominently:

```
📊 Strategy: Large Field GPP × Medium Slate (7 games)
→ Full leverage mode. Targeting 4 distinct stacks.
→ Ownership penalty: aggressive. Chalk threshold: 20%.
→ Recommended lineups: 150. Stack exposure cap: 30%.
```

The user can always override any parameter via the Custom expander.

---

## 4. Smart Auto-Stack Updates for Slate Size

Update the smart auto-stack logic from Section 5.3 of the main PRD to account for slate size:

### Small Slate Auto-Stack

On 2-game slates, the template is always **4-4**. On 3-game slates:
- If one game has significantly higher total than the others → 4-3-1 (4 from the best game)
- If two games are similar and one is low → 4-4 from the two good games
- If all three are similar → rotate between 4-3-1, 3-3-2, and 4-4

The key on small slates: **don't rotate through too many templates.** There are only 2-3 possible stacks. Pick the best 2-3 configurations and build your whole portfolio around them.

### Medium Slate Auto-Stack

Use the logic from the main PRD Section 5.3 as-is. Score games by leverage, pick templates based on the number of prime/good games.

### Large Slate Auto-Stack

More aggressive template variety:
- Use 3-4 different templates across the portfolio
- Include at least one "wild" template (3-2-2-1 or 3-2-1-1-1) that accesses deep leverage games the field ignores entirely
- When building 150 lineups, rotate through templates more aggressively to maximize game-stack diversity:

```python
if slate_type == "large" and num_lineups >= 100:
    # Build a rotation that covers more template shapes
    rotation = []
    prime_templates = [(4, 3, 1), (4, 2, 2)]
    secondary_templates = [(3, 3, 2), (4, 2, 1, 1)]
    wild_templates = [(3, 2, 2, 1), (3, 2, 1, 1, 1)]

    # 50% prime, 35% secondary, 15% wild
    prime_count = int(num_lineups * 0.50)
    secondary_count = int(num_lineups * 0.35)
    wild_count = num_lineups - prime_count - secondary_count

    for t in prime_templates:
        rotation.extend([t] * (prime_count // len(prime_templates)))
    for t in secondary_templates:
        rotation.extend([t] * (secondary_count // len(secondary_templates)))
    for t in wild_templates:
        rotation.extend([t] * (wild_count // len(wild_templates)))
```

---

## 5. Dashboard UI Updates

### 5.1 Replace Strategy Selector

Remove the `[GPP Leverage] [Cash/Safe] [Custom]` selector from the main PRD.

Replace with:

```
┌─────────────────────────────────────────────────┐
│  Contest Type                                    │
│  ○ Single Entry  ○ Small Field GPP  ● Large GPP │
│                                                  │
│  Slate: 7 games detected (Medium)               │
│  Strategy: Large Field GPP × Medium Slate       │
│                                                  │
│  ▼ Override Settings (Advanced)                  │
│    Ceiling Weight: [====|=====] 0.30             │
│    Ownership Penalty: [======|===] 0.55          │
│    Boom Weight: [===|======] 0.15                │
│    ...                                           │
└─────────────────────────────────────────────────┘
```

### 5.2 Slate Awareness Banner

After the user uploads data files and before Step 2 (Projections), display:

```
🎯 Slate Analysis
├── Games: 7 (Medium slate)
├── Batters: 78  |  Pitchers: 14
├── Prime leverage games: 2 (CLE@DET 9.5 total, MIL@PIT 8.5 total)
├── Games to consider fading: NYY@BOS (highest aggregate ownership)
└── Recommended templates: 4-3-1, 4-2-2, 3-3-2
```

This gives the user immediate situational awareness before they even look at projections.

### 5.3 Remove All Cash/50-50 References

Scan `daily_workflow.py` and all related files for any references to "cash", "50/50", "double up", or "cash_preset" and either remove them or rebrand them. The only presets should be tournament-related. The `SimulationConfig.cash_preset()` classmethod should be removed or renamed to `single_entry_preset()`.

---

## 6. Implementation Notes

### 6.1 Order of Operations

These changes integrate with the main PRD implementation order:

1. After main PRD Step 3 (LeverageConfig), add the contest-size presets and slate-profile detection
2. After main PRD Step 4 (solver objective), add the pitcher fade logic and slate-specific adjustments
3. After main PRD Step 5 (smart auto-stack), add the slate-aware template selection
4. After main PRD Step 9 (dashboard), replace the strategy selector and add the slate analysis banner

### 6.2 Testing

Add tests for:
- Slate profile detection correctly identifies 2-game, 7-game, 12-game slates
- Small slate adjustments increase ownership penalty relative to base config
- Large slate adjustments decrease ownership penalty relative to base config
- Pitcher fade bonus is applied on small slates, not on medium/large
- Auto-stack produces 4-4 on 2-game slates (not 3-3-2 or hodgepodge)
- Contest presets are mutually exclusive (selecting one clears the others)
- Strategy summary text matches the selected combination

### 6.3 Backward Compatibility

The `generate_lineups()` function should still work with no `LeverageConfig` passed. In that case, default to `"large_field_gpp"` mode with medium slate assumptions. This ensures existing scripts and tests don't break.
