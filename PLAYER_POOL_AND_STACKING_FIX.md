# CRITICAL FIX: Player Pool, Pitcher Validation, and Optimizer Logic

Read this entire document before making any changes. These fixes address the core problems causing the optimizer to select relievers, waste salary, and build uncorrelated lineups.

## Problem Summary

When the user pastes confirmed lineups into the dashboard, the system should ONLY use those players. Currently:
1. The player pool filter falls back to the full FanDuel pool when matching is imperfect
2. Relievers slip into the pitcher slot because there's no validation that a pitcher is actually a confirmed starter
3. The "Auto" stack mode doesn't enforce enough correlation — baseball lineups MUST be stacked
4. The optimizer doesn't have a true expected value calculation that accounts for correlation

---

## Fix 1: HARD player pool filter — no fallback to full pool

**File: `dashboard/daily_workflow.py`**

The current code at line ~936-960 checks if the filtered pool is "too small" and falls back to the full player list. This fallback is the root cause of relievers and bench players appearing in lineups.

**Change the fallback behavior.** When lineup paste data is provided, NEVER fall back to the full pool. Instead, if the match rate is low, show a warning but still use only the matched players. The only situation where we should stop is if we literally can't fill a valid lineup (fewer than 9 matched players covering all positions).

Find the block starting around line 936:
```python
            pool_too_small = (
                len(combined) < 40
                or bool(_missing)
                or match_rate < 0.5
            )
```

Replace the threshold. Change `len(combined) < 40` to `len(combined) < 15`. A 10-game slate with confirmed lineups gives ~180 starters. Even a 2-game slate gives ~40. If we have fewer than 15 matched players, something is genuinely wrong with the paste. But 40 is way too aggressive — it triggers the fallback on short slates.

Also, in the fallback block that follows (where it restores the full pool), add a hard override: if lineup paste was provided AND at least 15 players matched AND all positions are covered, NEVER restore the full pool. Instead just warn about unmatched players.

Find the section where `combined` gets reset to `full_pool_backup` (around line 960-990). Wrap that restoration in a stricter condition:

```python
            if pool_too_small:
                # Only fall back if positions are literally missing or match rate is catastrophic
                if bool(_missing) or len(combined) < 10:
                    # Genuine problem — restore full pool but warn loudly
                    combined = full_pool_backup
                    optional_messages.append(
                        "WARNING: Lineup paste matching failed. Using full player pool. "
                        "Check that pasted names match the FanDuel player list."
                    )
                else:
                    # Pool is small but usable — keep the filtered pool, just warn
                    optional_messages.append(
                        f"Note: Only {len(combined)} players matched from paste. "
                        f"Proceeding with confirmed starters only."
                    )
```

## Fix 2: Pitcher must be a confirmed starter — validate in solver

**File: `src/slate_optimizer/optimizer/solver.py`**

The existing pitcher salary floor fix (from URGENT_LEVERAGE_FIX.md) helps, but the real solution is: if the dataset has an `is_confirmed_lineup` column, ONLY allow pitchers who are confirmed starters.

In the `generate_lineups()` function, right after the pitcher salary floor filter, add:

```python
    # ── Pitcher must be a confirmed starter when lineup data is available ──
    if "is_confirmed_lineup" in df.columns:
        # For pitchers, require is_confirmed_lineup=True
        # This prevents relievers from being selected even if they pass the salary floor
        unconfirmed_pitcher = (
            (df["player_type"].str.lower() == "pitcher")
            & (~df["is_confirmed_lineup"].astype(bool))
        )
        if unconfirmed_pitcher.any() and df["is_confirmed_lineup"].astype(bool).any():
            # Only filter if we have SOME confirmed data (avoid filtering everything)
            confirmed_pitchers = df[
                (df["player_type"].str.lower() == "pitcher") 
                & (df["is_confirmed_lineup"].astype(bool))
            ]
            if len(confirmed_pitchers) >= 2:
                # We have enough confirmed pitchers — remove unconfirmed ones
                df = df[~unconfirmed_pitcher].reset_index(drop=True)
```

## Fix 3: Auto-stack must ALWAYS enforce real stacks in GPP mode

**File: `src/slate_optimizer/optimizer/solver.py`**

The current auto-stack logic in `_select_auto_stack_template` is good, but there's a bigger issue: the system allows the solver to produce lineups with NO stacking when the stack constraints are infeasible. The fallback on line ~582 drops all stack constraints:

```python
        if status != LpStatusOptimal and used_stacks:
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns",
                leverage_config=leverage_config,
            )
```

This "no stacks" fallback should be a LAST resort and should still enforce a minimum 3-batter team grouping. Replace the no-stack fallback with a relaxed stack fallback:

```python
        # Fallback: if stacks made it infeasible, try a simpler stack first
        if status != LpStatusOptimal and used_stacks:
            # Try a relaxed 3-3-2 before giving up on stacks entirely
            warnings.warn(
                f"Lineup {lineup_index}: template {current_template} infeasible, trying 3-3-2.",
                stacklevel=2,
            )
            prob, decision_vars = _build_base_lp(
                pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_relax",
                leverage_config=leverage_config,
            )
            relaxed_info = _add_stack_constraints(
                prob, pool, decision_vars, (3, 3, 2),
                min_game_total=None,  # Drop game total filter for fallback
            )
            if relaxed_info:
                if _gpp_mode and leverage_config is not None:
                    _add_stack_leverage_bonus(prob, pool, relaxed_info, leverage_config.stack_leverage_bonus)
            status = prob.solve(PULP_CBC_CMD(msg=False))
            
            # Only if 3-3-2 also fails, go truly unconstrained
            if status != LpStatusOptimal:
                warnings.warn(
                    f"Lineup {lineup_index}: all stacks infeasible, solving without stacks.",
                    stacklevel=2,
                )
                prob, decision_vars = _build_base_lp(
                    pool, lineup_index, salary_cap, max_lineup_ownership, previous_lineups, tag="_ns",
                    leverage_config=leverage_config,
                )
                status = prob.solve(PULP_CBC_CMD(msg=False))
```

## Fix 4: Add `leverage_adjusted_top1` to the simulation selection metric dropdown

**File: `dashboard/daily_workflow.py`**

Find the `metric_options` list (around line 3621):
```python
        metric_options = ["top_1pct_rate", "win_rate", "cash_rate", "expected_roi", "p99_score"]
```

Add the leverage-adjusted metric:
```python
        metric_options = ["top_1pct_rate", "leverage_adjusted_top1", "win_rate", "expected_roi", "p99_score"]
```

Remove `"cash_rate"` since we only play tournaments.

## Fix 5: Add portfolio size input to Step 4

**File: `dashboard/daily_workflow.py`**

This is the missing "how many lineups to select from simulation" input that was identified earlier. In the Step 4 UI (around line 3576), add before the "Run Simulation" button:

In the left column of sim settings, add:
```python
        sim_state["num_portfolio_lineups"] = st.number_input(
            "Portfolio size (lineups to select)",
            min_value=1,
            max_value=2000,
            value=int(sim_state.get("num_portfolio_lineups", 150) or 150),
            step=10,
            help="How many final lineups to select from your candidate pool after simulation.",
        )
```

Add `"num_portfolio_lineups": 150` to the `_get_sim_config_state()` default dict.

In `_build_simulation_config()`, add:
```python
    config.num_candidates = int(state.get("num_portfolio_lineups", 150))
```

## Fix 6: Default stacking should be MORE aggressive for baseball

**File: `src/slate_optimizer/optimizer/solver.py`**

In `_select_auto_stack_template`, the medium slate default when there are fewer than 2 "prime" games falls back to (3, 3, 2). For baseball, 4-hitter stacks are almost always preferred because run-scoring is so correlated. Change the medium slate defaults:

```python
    # ── Medium slate (default) ───────────────────────────────────────
    if prime_count >= 2:
        return (4, 4)
    if prime_count >= 1:
        return (4, 3, 1)
    if good_count >= 2:
        return (4, 3, 1)  # was (3, 3, 2) — 4-batter primary stack is always better in baseball
    if good_count >= 1:
        return (4, 2, 2)
    return (3, 3, 2)  # absolute fallback only
```

For large slates, the rotation should also favor 4-batter primaries more:

In `_build_large_slate_rotation`, change the split to 60% prime / 30% secondary / 10% wild:
```python
    prime_count = int(num_lineups * 0.60)
    secondary_count = int(num_lineups * 0.30)
    wild_count = num_lineups - prime_count - secondary_count
```

---

## After all fixes:

1. Run `python -m pytest tests/ -v` — fix any test failures
2. Commit with message: "Fix player pool filtering, enforce confirmed starters for pitchers, improve stacking defaults, add portfolio size input"
3. Push to main
