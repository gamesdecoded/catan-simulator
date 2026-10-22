"""Validation for the Catan engine (opening-strategy experiment).
Run: python validate.py

Checks:
 1. 1,000 baseline games run with no crashes; every game has a winner.
 2. First-player win rate (plausible small edge, nothing absurd).
 3. Resource economy: no negative resources; created-destroyed == held.
 4. Each of the 5 opening strategies completes games with no errors.
 5. Timing + extrapolation to the 250,000-game production run.
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/sim-lab/catan")
import board
from board import PORT_AT_VERT
import engine
from engine import STRATEGIES, STRATEGY_DEFS

STAT_NAMES = ["turns", "capped", "unused", "created", "destroyed",
              "held", "min_ok", "conserve_ok", "winner_vp", "lr_awarded",
              "army_awarded", "devs_bought"]


def run(strat, n, seed):
    wins = np.zeros(4, dtype=np.int64)
    stat = np.zeros(12, dtype=np.int64)
    t0 = time.time()
    engine.simulate_many(strat, n, seed, wins, stat)
    return wins, stat, time.time() - t0


def report(label, wins, stat, dt, n):
    print(f"\n=== {label} ({n} games, {dt:.1f}s) ===")
    print(f"wins by player: {wins.tolist()}  (sum={wins.sum()})")
    for i, p in enumerate(wins):
        print(f"  P{i}: {p / n * 100:.1f}%")
    print(f"avg turns/game: {stat[0] / n:.1f}")
    print(f"turn-cap hits: {stat[1]}")
    print(f"conservation violations: {n - stat[7]} | "
          f"negative-resource games: {n - stat[6]}")
    print(f"avg winner VP: {stat[8] / n:.2f}")
    print(f"longest-road awarded: {stat[9] / n * 100:.1f}% of games | "
          f"largest-army awarded: {stat[10] / n * 100:.1f}% of games")
    print(f"avg dev cards bought/game: {stat[11] / n:.1f}")


print("board: 54 vertices, 72 edges, "
      f"{(PORT_AT_VERT > 0).sum()} port vertices")
print("strategies under test:")
for sid, name in STRATEGIES.items():
    print(f"  {sid} = {name}")

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        ok = False


# ---- 1+2+3: baseline, 1000 games ----
wins, stat, dt = run(0, 1000, 42)
report("BASELINE (strategy 0, all-heuristic-equivalent)", wins, stat, dt, 1000)

print("\n--- validation checks ---")
check("1000 games, every game has a winner", wins.sum() == 1000)
check("no turn-cap hits (all games ended naturally)", stat[1] == 0)
p0 = wins[0] / 1000
check(f"first-player win rate plausible ({p0 * 100:.1f}%, expect ~20-35%)",
      0.20 <= p0 <= 0.35)
check("no conservation violations", stat[7] == 1000)
check("no negative resources", stat[6] == 1000)
check("winners reach 10 VP on average", 9.9 <= stat[8] / 1000 <= 10.5)

# ---- 4: each strategy completes games cleanly ----
print("\n--- per-strategy validation (200 games each) ---")
strat_p0_rates = {}
for sid, name in STRATEGIES.items():
    w2, s2, d2 = run(sid, 200, 1000 + sid)
    p0r = w2[0] / 200
    strat_p0_rates[name] = p0r
    print(f"strategy {sid} ({name}): P0 win rate {p0r * 100:.1f}% "
          f"({d2:.1f}s, avg {s2[0] / 200:.0f} turns)")
    check(f"strategy '{name}': 200 games, all have winners", w2.sum() == 200)
    check(f"strategy '{name}': no cap hits", s2[1] == 0)
    check(f"strategy '{name}': conservation holds", s2[7] == 200)
    check(f"strategy '{name}': no negative resources", s2[6] == 200)

print("\n--- P0 win rate by strategy (validation sample, n=200 each; "
      "wide CIs, production run will tighten) ---")
for name, r in strat_p0_rates.items():
    print(f"  {name:12s}: {r * 100:5.1f}%")

# ---- 5: timing extrapolation ----
per_game = dt / 1000
print(f"\n--- timing ---")
print(f"measured: {per_game * 1000:.2f} ms/game (steady state, compiled)")
print(f"extrapolated 50,000 games x 5 strategies (250,000 games): "
      f"{per_game * 250000 / 3600:.2f} hours")
check("250k games feasible (< 2 hours)", per_game * 250000 < 7200)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
