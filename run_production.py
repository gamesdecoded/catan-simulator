"""Production run: 50,000 games per opening strategy x 5 strategies.

Saves ~/workspace/sim-lab/catan/results.json with per-strategy
{games, wins, win_rate, avg_game_turns} plus the exact strategy
definitions used. Run in the background (~1 hour).
"""
import sys
import time
import json
import datetime
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/sim-lab/catan")
from engine import simulate_many, STRATEGIES, STRATEGY_DEFS

N = 50_000
OUT = "/home/hatch/workspace/sim-lab/catan/results.json"

results = {
    "meta": {
        "date": datetime.date.today().isoformat(),
        "games_per_strategy": N,
        "n_strategies": len(STRATEGIES),
        "engine": "4-player Catan, identical heuristic bots; Player 0's "
                  "opening strategy is the only asymmetry; board "
                  "re-randomized every game. See METHODS.md for full "
                  "rules, simplifications (no player-to-player trading), "
                  "and strategy definitions.",
        "strategy_definitions": STRATEGY_DEFS,
    },
    "strategies": {},
}

t_all = time.time()
for sid, name in STRATEGIES.items():
    wins = np.zeros(4, dtype=np.int64)
    stat = np.zeros(12, dtype=np.int64)
    t0 = time.time()
    simulate_many(sid, N, 9000 + sid, wins, stat)
    dt = time.time() - t0
    p0_wins = int(wins[0])
    results["strategies"][name] = {
        "strategy_id": sid,
        "games": N,
        "wins": p0_wins,                       # Player-0 wins
        "wins_by_player": [int(w) for w in wins],
        "win_rate": p0_wins / N,
        "avg_game_turns": float(stat[0] / N),
        "run_seconds": dt,
    }
    print(f"[{time.strftime('%H:%M:%S')}] {name}: "
          f"P0 {p0_wins}/{N} = {p0_wins / N * 100:.2f}%  "
          f"avg {stat[0] / N:.1f} turns  ({dt / 60:.1f} min)", flush=True)

results["meta"]["total_run_seconds"] = time.time() - t_all
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {OUT}")
