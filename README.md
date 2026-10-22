# Catan Opening Simulator — 825,000 Games

Monte Carlo simulation backing the YouTube video **"The Best Catan Opening,
Tested on 825,000 Games."** Every number in the video comes from this code.
Run it yourself.

## Results (50,000 games per strategy; 95% CIs non-overlapping)

| Strategy | Win rate | 95% CI |
|---|---|---|
| Resource diversity | 26.74% | ±0.39 |
| Balanced baseline (control) | 22.81% | ±0.37 |
| Max pips | 21.18% | ±0.36 |
| Wheat-and-ore rush | 13.28% | ±0.30 |
| Port rush | 6.91% | ±0.22 |

Follow-up experiments (575,000 more games): random-placement control 1.76%;
seat rotation 22.85% → 24.07% → 26.21% → 27.59% (same strategy, seats 1–4);
"smart" port rush 6.28% (worse than the plain version); diversity weight
sweep 20.84% → 27.07% → 27.12% (saturates).

## Rules simulated

- 4-player Catan, first to 10 victory points
- Snake-draft opening (1-2-3-4-4-3-2-1); Player 1's two picks use the tested
  strategy, all other players use the baseline heuristic
- Board (terrain + number tokens) re-randomized every game
- Real Catan discard rule: on a 7, every player holding more than 7 cards
  discards half, rounded down
- Settlements, cities, roads, development cards, robber, longest road,
  largest army
- **No player-to-player trading** (the big limitation — stated in the video
  and here; diversity's edge is probably smaller at real tables)

## The five strategies

1. **Resource diversity** — first pick: most distinct adjacent resources;
   second pick: maximizes distinct resources covered by the *union* of both
   picks. Ties broken by higher pips.
2. **Baseline** — (sum of adjacent pips) + 1.5 × (distinct resources); the
   control, also played by all three opponents.
3. **Max pips** — highest sum of adjacent dice pips, no diversity bonus.
4. **Wheat-and-ore rush** — wheat+ore pips first, then total pips.
5. **Port rush** — any port vertex first (3:1 or 2:1), then highest pips
   among port vertices; does NOT match the port to a produced resource.

## Files

- `engine.py` — the simulation engine (numba-accelerated); `STRATEGIES` and
  exact `STRATEGY_DEFS`
- `board.py` — static board graph (54 intersections, 72 edges, 9 fixed ports)
- `validate.py` — validation suite (1,000-game baseline run, per-strategy
  checks, resource-economy conservation)
- `run_production.py` — production run: 50k games x 5 strategies
- `inserts.py` — follow-up experiment strategies (random control, smart
  port, diversity weight sweep, seat rotation, opening-stats logging)
- `run_inserts.py` — insert experiments (checkpointed/resumable)
- `results.json` — production results; `insert_results.json` — insert results
- `METHODS.md` — full methodology, simplifications, and definitions

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install numba numpy
python validate.py        # validation suite
python run_production.py  # 50k games x 5 strategies (~8 min)
python run_inserts.py     # follow-up experiments (~8 min)
```

## Validation

`validate.py` runs 1,000 baseline games (every game has a winner, no
turn-cap hits, resource conservation holds exactly, no negative
resources), checks the first-player win rate is plausible, and runs 200
games per strategy with zero errors. The weight-0 diversity sweep point
reproduces the max-pips result within noise, and the seat-0 rotation
reproduces the production control within noise — two independent
cross-checks of the implementation.
