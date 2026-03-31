# URGENT FIX: Leverage Optimizer Producing Bad Lineups

Read this entire prompt before making changes. The leverage system has three critical bugs that are causing it to select relievers as pitchers, waste salary, and over-stack bad teams.

## Bug 1: No pitcher salary floor — relievers are being selected as starting pitchers

Robert Suarez ($5,500), Hayden Harris ($5,500), and Junior Fernandez ($5,500) are RELIEVERS. They should never be selected in the P slot. The system has no way to distinguish starters from relievers.

### Fix in `src/slate_optimizer/optimizer/solver.py`:

In the `generate_lineups()` function, AFTER the player floor filter block (around line 486-495 where `_gpp_mode` filters by `min_viable_projection_pct`) and BEFORE `usage_limits = _max_usage(df, num_lineups)`, add:

```python
    # ── Pitcher salary floor: exclude relievers from pitcher pool ────
    # On FanDuel, starting pitchers are generally $6,500+. Relievers at
    # $5,500-$6,000 should never be selected as the lineup's starting pitcher.
    # This applies in ALL modes, not just GPP.
    PITCHER_MIN_SALARY = 6500
    pitcher_mask_filter = (df["player_type"].str.lower() == "pitcher") & (df["salary"] < PITCHER_MIN_SALARY)
    if pitcher_mask_filter.any():
        df = df[~pitcher_mask_filter].reset_index(drop=True)
```

## Bug 2: No minimum salary usage — lineups waste $5,000-$6,000 of cap space

Average lineup salary is $29,295 out of $35,000. Leaving $5,700 on the table means the optimizer is choosing 9 cheap players when it could upgrade multiple positions.

### Fix in `src/slate_optimizer/optimizer/solver.py`:

In `_build_base_lp()`, add a minimum salary constraint right after the existing salary cap constraint (line ~172 `prob += lpSum(...) <= salary_cap`):

```python
    # Minimum salary usage — don't waste cap space
    # For a $35K cap, require at least $32K usage (91% utilization)
    min_salary = int(salary_cap * 0.91)
    prob += lpSum(pool.loc[idx, "salary"] * var for idx, var in decision_vars.items()) >= min_salary
```

## Bug 3: Ownership penalty is way too aggressive — makes good players look bad

With `ownership_penalty=0.55` and `chalk_extra_penalty=0.20`, an ace pitcher projected for 40 FD points at 25% estimated ownership loses 0.55*0.25 + 0.20*0.25 = 0.1875 points from the objective for ownership alone. Meanwhile a $2,300 batter projected for 8 points at 2% ownership only loses 0.011 points. The penalty doesn't scale with projection magnitude — it's an absolute penalty applied to a relative ownership percentage, which means it disproportionately hurts expensive players.

### Fix in `src/slate_optimizer/optimizer/solver.py`:

In `_compute_gpp_score()`, change the ownership penalty to be RELATIVE to the player's projection rather than absolute:

Replace the current ownership penalty lines:
```python
    score -= leverage_config.ownership_penalty * ownership
    ...
    if ownership > leverage_config.chalk_threshold:
        score -= leverage_config.chalk_extra_penalty * ownership
```

With this version that scales with the player's own projection:
```python
    # Ownership penalty scaled relative to the player's mean projection.
    # This ensures a 25%-owned ace still scores well, while a 25%-owned
    # mediocre player gets penalized more proportionally.
    if mean > 0:
        relative_ownership_cost = ownership / max(mean, 1.0)
        score -= leverage_config.ownership_penalty * relative_ownership_cost * mean * 0.5
    
    # Extra chalk penalty (also relative)
    if ownership > leverage_config.chalk_threshold and mean > 0:
        score -= leverage_config.chalk_extra_penalty * relative_ownership_cost * mean * 0.3
```

## Bug 4: Default LeverageConfig parameters are too aggressive

### Fix in `src/slate_optimizer/optimizer/config.py`:

Update the `LeverageConfig` defaults AND all three presets:

Default / `large_field_gpp_preset`:
- `ownership_penalty`: 0.55 → **0.30**
- `chalk_extra_penalty`: 0.20 → **0.08**
- `ceiling_weight`: 0.30 → **0.20**
- `boom_weight`: 0.15 → **0.08**
- `within_stack_chalk_penalty`: 0.15 → **0.06**

`small_field_gpp_preset`:
- `ownership_penalty`: 0.30 → **0.20**
- `chalk_extra_penalty`: 0.10 → **0.05**
- `ceiling_weight`: 0.25 → **0.18**

`single_entry_preset`:
- `ownership_penalty`: 0.25 → **0.15**
- `chalk_extra_penalty`: 0.08 → **0.04**
- `ceiling_weight`: 0.30 → **0.22**

Also in `_apply_small_slate()`:
- Change `cfg.ownership_penalty *= 1.3` to `cfg.ownership_penalty *= 1.15`
- Change `cfg.chalk_extra_penalty *= 1.5` to `cfg.chalk_extra_penalty *= 1.2`

## Bug 5: Ownership model salary exponent creates extreme ownership estimates

### Fix in `src/slate_optimizer/projection/ownership.py`:

In the `OwnershipModelConfig` dataclass, change:
- `salary_exponent`: 1.8 → **1.4**
- `max_pitcher_ownership`: 0.35 → **0.30**
- `pitcher_steepness`: 4.0 → **3.0**

## After all fixes:

1. Run `python -m pytest tests/ -v` to make sure nothing is broken
2. Commit with message: "Fix leverage optimizer: add pitcher salary floor, min salary usage, scale ownership penalty relative to projection, reduce penalty aggressiveness"  
3. Push to main
