# METHODS.md — Catan Monte Carlo engine

## What the engine does
4-player Catan, full games, 4 identical heuristic bots. The experimental
variable is **Player 0's opening STRATEGY** (`p0_strategy`, ids 0-4),
used for both of Player 0's snake-draft picks (picks 1 and 8 of the
1-2-3-4-4-3-2-1 draft). Players 1-3 place by the baseline heuristic.
Board (terrain + number tokens) is re-randomized every game. Winner =
first to 10 VP (settlements 1, cities 2, longest road 2, largest army 2,
VP cards).

## Opening strategies (exact definitions)
- **0 baseline** (control): identical to the heuristic used by players
  1-3. Score each legal intersection as (sum of adjacent dice pips) +
  1.5 x (distinct adjacent resource types, desert excluded); pick the
  highest. Ties -> lowest vertex id (deterministic).
- **1 max_pips**: legal intersection with the highest sum of adjacent
  dice pips (desert excluded, no diversity bonus). Applied independently
  to both picks. Ties -> lowest vertex id.
- **2 diversity**: first pick = legal intersection adjacent to the most
  distinct resource types (desert excluded); ties -> higher pip sum,
  then lowest vertex id. Second pick = legal intersection maximizing
  distinct resource types in the UNION of both picks' adjacent hexes;
  ties -> higher pip sum of the second vertex, then lowest vertex id.
- **3 port_seeking**: prefer intersections touching a port (any of the 9
  fixed ports: 4x 3:1, 5x 2:1 wood/brick/sheep/wheat/ore). Among legal
  intersections, pick a port vertex first; among port vertices (or when
  none is legal) pick the highest adjacent pip sum. Ties -> lowest
  vertex id.
- **4 wheat_ore**: prefer city-building resources. Score each legal
  intersection by the sum of adjacent dice pips on WHEAT and ORE hexes
  only; ties -> higher total adjacent pip sum (all resources), then
  lowest vertex id.

All five strategies are deterministic given the board. Road placement
after each settlement (`best_opening_road`) is unchanged for all
players.

## Rules implemented faithfully
- 19-hex board, 54 intersections, 72 edges; distance rule for settlements.
- Terrain: 4 wood, 4 brick, 4 sheep, 4 wheat, 3 ore, 1 desert; desert starts
  with the robber and no number token.
- Number tokens 2-12 excluding 7 (18 tokens), dice = 2d6.
- Snake-draft opening: settlement + 1 road each pick; 2nd settlement grants
  starting resources (1 per adjacent non-desert hex).
- Piece limits: 5 settlements, 4 cities, 15 roads per player.
- Dev cards: 25 (14 knight, 5 VP, 2 road building, 2 year of plenty,
  2 monopoly), drawn without replacement.
- Longest road: 2 VP, needs 5+ segments, computed by exact DFS longest-path
  on the player's road graph; opponent settlements block continuity
  (you may not build *through* them, standard rule).
- Largest army: 2 VP, needs 3+ knights, ties keep the current holder
  (must strictly exceed).
- Robber: on a 7, targets the current VP leader (excluding the roller),
  blocks the leader's highest-probability adjacent hex, steals 1 random
  card from a random eligible victim.
- Discard rule (real Catan): on a 7, every player holding MORE THAN 7
  resource cards discards half, rounded down (most-abundant type first).
- Bank trading 4:1, 3:1 and 2:1 with matching ports. Bank has infinite
  resources/cards.
- Win check includes unrevealed VP dev cards (standard: you may reveal to win).

## Simplifications vs. real Catan

> **BIGGEST REALISM GAP — no player-to-player trading (v1).** Bots trade
> only with the bank, and only when the trade directly completes their
> next build. This removes negotiation, which in real Catan is a major
> skill axis. Affects: resource velocity (games are slower and more
> luck-driven than real Catan), robber politics, and monopoly-card value.
> All players are affected symmetrically, so strategy *rankings* should
> still be meaningful, but absolute win rates and game lengths do not
> match real table play. Any video using these numbers must disclose this.

1. **Fully randomized board.** Terrain and numbers are independently
   shuffled each game. Real casual play often does this, but the official
   setup constrains 6/8 adjacency; we do not. Consequence (important):
   with full randomization, **every intersection is statistically
   identical except for its position relative to the fixed ports**.
2. **Ports in a fixed, non-canonical layout.** 9 ports (4x 3:1, 5x 2:1
   wood/brick/sheep/wheat/ore) placed evenly around the coast. Real Catan
   has a specific port arrangement; ours is a reasonable fixed pattern.
3. **Dev-card timing simplified.** Knights are played automatically at the
   start of the turn after the turn they are drawn (max 1/turn, before the
   roll). Road building / year of plenty / monopoly resolve **immediately**
   when drawn (real Catan: not on the draw turn, max 1 card/turn).
4. **Greedy, identical bots.** Fixed build priority
   settlement > city > road > dev card, always building the highest-scoring
   legal option. No blocking opponents, no strategic robber beyond
   leader-targeting, no port-collection planning, no card holding, no
   looking ahead. All four bots run the *exact* same code; the only
   asymmetry in an experiment is Player 0's opening strategy.
5. **Placement heuristic** (players 1-3, and Player 0's baseline):
   score = sum of dice pips on adjacent hexes + 1.5 x (distinct resource
   types adjacent). Ties -> lowest vertex id (deterministic).
6. **City upgrade** always upgrades the bot's highest-scoring settlement.
7. **Year of plenty** takes 2 of the resource with the largest deficit
   toward a settlement (ties -> ore > wheat > wood > brick > sheep).
8. **Monopoly** takes the resource opponents collectively hold the most of
   (ties -> ore first).
9. **Discard choice:** discards the most-abundant resource type first.
10. **Turn cap** of 1200 total turns (300 rounds); if hit, highest VP wins.
    Never triggered in validation (avg game ~108 turns).
11. **No even-money / no surrender** (not modeled; not in base Catan
    tournament rules we target).

## Experimental design note (why strategies, not fixed spots)
An earlier design tested **fixed pairs of intersections** as Player 0's
opening. Validation revealed a structural fact: because terrain and
numbers are fully randomized each game, all 54 intersections are
symmetric in expectation **except for port proximity**. A fixed vertex
pair is therefore almost always *worse* than the adaptive heuristic
(measured: fixed openings won 1-7% for Player 0 vs ~21% when Player 0
also used the heuristic), because the heuristic picks the best spot *for
that board* while a fixed pair is stuck. Fixed 2nd-pick vertices were
also frequently taken or blocked by the 8th draft pick (27-69% fallback
to heuristic), making fixed-pair experiments fragile.

The design was therefore switched to compare opening **strategies**
(`p0_strategy` 0-4), which answers the question Catan players actually
ask ("what opening strategy is best") and is the better video.

## Insert experiments (2026-10-22)

Five follow-up experiments for the video's optional inserts, implemented in
`inserts.py` (which reuses `engine.py`'s validated game loop unchanged) and
run by `run_inserts.py` -> `insert_results.json`. 575,000 additional games.
The experimental player is a parameter `exp` (0..3); everyone else plays
the baseline heuristic.

- **random_control** (50k): two settlements at uniformly random legal
  intersections, then baseline play. 1.76% ± 0.12.
- **seat rotation** (4 x 50k): baseline strategy from each seat of the
  1-2-3-4-4-3-2-1 snake draft. 22.85% → 24.07% → 26.21% → 27.59%
  (±0.37–0.39, all CIs non-overlapping). Establishes the seat-order effect
  behind the control's 22.81%.
- **smart_port** (50k): port vertices whose port matches a produced resource
  (2:1 matched to the same resource on the vertex; 3:1 matched to any
  produced resource), highest pips wins; falls back to plain port-seeking if
  none is legal. 6.28% ± 0.21 — worse than plain port rush.
- **opening stats** (5 x 25k, one per production strategy): per game, sum of
  adjacent pips across both opening settlements, distinct resources covered
  (union), and the global turn of the experimental player's first city.
- **diversity weight sweep** (3 x 50k): score = pips + w × distinct(joint
  union), w ∈ {0.0, 1.5, 3.0}. 20.84% → 27.07% → 27.12%. w=0.0 reproduces
  max_pips (21.18% ± 0.36) within noise, confirming the implementation.

Validation bar (same as production): every game has a winner, no negative
resources, created − destroyed == held. Turn-cap fallback (winner = VP
leader at 1200 turns) is part of the engine; at 50k scale it triggers in
~0.002% of games and is recorded per experiment (`cap_hits`) rather than
treated as a failure. `run_inserts.py` checkpoints each experiment to
`insert_results_partial.json` so a failure never loses completed work.

## Performance
~14 ms/game (numba, compiled, 2-core VM) -> 50,000 games ~12 minutes.
5 strategies x 50k = 250,000 games ~1 hour.

## Files
- `board.py` -- static graph (54v/72e), ports, opening validation helpers.
- `engine.py` -- numba game simulation; `simulate_many(strat, n_games,
  seed, wins, stat_sum)`; `STRATEGIES` and `STRATEGY_DEFS`.
- `validate.py` -- validation suite; prints the validation report.
- `run_production.py` -- production run: 50k games x 5 strategies ->
  `results.json`.
- `results.json` -- production results (written by run_production.py).
- `inserts.py` -- insert strategies + instrumented runners (numba, reuses
  engine.py's validated game loop).
- `run_inserts.py` -- insert experiments -> `insert_results.json`
  (checkpointed/resumable).
- `insert_results.json` -- insert results (written by run_inserts.py).
