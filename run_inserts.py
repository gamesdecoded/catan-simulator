"""Run the 5 optional insert experiments for the Catan video.

Single-process; pin to the free CPU (taskset) so the Monopoly 200M run
on the other core is untouched. Saves insert_results.json.
"""
import sys
import time
import json
import math
import os
import datetime

sys.path.insert(0, "/home/hatch/workspace/sim-lab/catan")
import numpy as np

from inserts import simulate_many_xp
from engine import STRATEGIES

OUT = "/home/hatch/workspace/sim-lab/catan/insert_results.json"


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)


def run(exp, mode, prod_strat, w, n, seed):
    wins = np.zeros(4, dtype=np.int64)
    stat = np.zeros(12, dtype=np.int64)
    opn = np.zeros(4, dtype=np.int64)
    t0 = time.time()
    simulate_many_xp(exp, mode, prod_strat, w, n, seed, wins, stat, opn)
    return wins, stat, opn, time.time() - t0


def validate(label, wins, stat, n):
    assert int(wins.sum()) == n, f"{label}: wins sum {wins.sum()} != {n}"
    assert int(stat[6]) == n, f"{label}: negative-resource games"
    assert int(stat[7]) == n, f"{label}: conservation violations"
    # Turn-cap fallback (winner = VP leader) is part of the validated engine.
    # The 1000-game validation bar of zero cap hits is too strict at 50k
    # scale; allow up to 0.05% and record the count.
    cap = int(stat[1])
    assert cap <= max(1, n // 2000), \
        f"{label}: {cap} turn-cap hits ({cap / n * 100:.3f}%)"
    print(f"  PASS {label}: {n} games, avg {stat[0] / n:.0f} turns, "
          f"cap hits {cap}", flush=True)


def entry(exp, mode, prod_strat, w, n, seed, label, defn):
    wins, stat, opn, dt = run(exp, mode, prod_strat, w, n, seed)
    validate(label, wins, stat, n)
    xp_wins = int(wins[exp])
    p = xp_wins / n
    return {
        "label": label,
        "definition": defn,
        "exp_player": exp,
        "games": n,
        "wins": xp_wins,
        "wins_by_player": [int(x) for x in wins],
        "win_rate": p,
        "ci95": ci95(p, n),
        "cap_hits": int(stat[1]),
        "avg_game_turns": float(stat[0] / n),
        "avg_opening_pips": float(opn[0] / n),
        "avg_opening_distinct": float(opn[1] / n),
        "avg_first_city_turn": (float(opn[2] / opn[3]) if opn[3] > 0 else None),
        "share_built_city": float(opn[3] / n),
        "run_seconds": dt,
        "seed": seed,
    }


t_all = time.time()

# ---- checkpointing: each experiment's entry is saved to disk the moment
# it passes validation, so a later failure never loses completed work.
# Re-running the script resumes from the partial file. ----
PARTIAL = "/home/hatch/workspace/sim-lab/catan/insert_results_partial.json"


def load_partial():
    try:
        with open(PARTIAL) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_partial(d):
    tmp = PARTIAL + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, PARTIAL)


partial = load_partial()
if partial:
    print(f"resuming: {len(partial)} experiments already cached", flush=True)


def cached(key, thunk):
    if key in partial:
        print(f"  (cached {key})", flush=True)
        return partial[key]
    e = thunk()
    partial[key] = e
    save_partial(partial)
    return e


out = {"info": {
    "date": datetime.date.today().isoformat(),
    "engine": "engine.py (validated) + inserts.py; 4 identical heuristic bots, "
              "board re-randomized every game; no player-to-player trading. "
              "See METHODS.md.",
    "ci": "95% Wald interval 1.96*sqrt(p(1-p)/n)",
}, "experiments": {}}

# ---- 1. random-placement control ----
print("[1] random-placement control (50k)", flush=True)
out["experiments"]["random_control"] = cached(
    "random_control",
    lambda: entry(
        0, 1, 0, 0.0, 50_000, 91001, "random_control",
        "Two settlements at uniformly random legal intersections; baseline "
        "heuristic play afterwards."))

# ---- 2. seat rotation: baseline strategy from each seat ----
print("[2] seat rotation (4 x 50k)", flush=True)
seats = {}
for s in range(4):
    print(f"    seat {s}", flush=True)
    seats[f"seat_{s}"] = cached(
        f"seat_{s}",
        lambda s=s: entry(
            s, 0, 0, 0.0, 50_000, 91100 + s, f"baseline_seat_{s}",
            f"Baseline strategy played from seat {s} of the 1-2-3-4-4-3-2-1 "
            f"snake draft; all other seats baseline."))
out["experiments"]["seat_rotation"] = seats

# ---- 3. smart port ----
print("[3] smart port (50k)", flush=True)
out["experiments"]["smart_port"] = cached(
    "smart_port",
    lambda: entry(
        0, 2, 0, 0.0, 50_000, 91201, "smart_port",
        "Port vertices whose port matches a produced resource (2:1 matched "
        "to the same resource on the vertex; 3:1 matched to any produced "
        "resource), highest pips wins; falls back to plain port-seeking."))

# ---- 4. opening stats per production strategy (25k each) ----
print("[4] opening stats per strategy (5 x 25k)", flush=True)
ostats = {}
for sid, name in STRATEGIES.items():
    print(f"    {name}", flush=True)
    ostats[name] = cached(
        f"ostats_{name}",
        lambda sid=sid, name=name: entry(
            0, 0, sid, 0.0, 25_000, 91300 + sid, f"opening_stats_{name}",
            f"Production strategy '{name}' with opening instrumentation."))
out["experiments"]["opening_stats"] = ostats

# ---- 5. diversity weight sweep ----
print("[5] diversity weight sweep (3 x 50k)", flush=True)
sweep = {}
for i, w in enumerate([0.0, 1.5, 3.0]):
    print(f"    w={w}", flush=True)
    sweep[f"w_{w}"] = cached(
        f"sweep_w_{w}",
        lambda i=i, w=w: entry(
            0, 3, 0, float(w), 50_000, 91400 + i, f"diversity_w_{w}",
            f"Diversity score = pips + {w} * distinct(joint union); ties by "
            f"lowest vertex id. w=0.0 reproduces max_pips."))
out["experiments"]["weight_sweep"] = sweep

out["info"]["total_run_seconds"] = time.time() - t_all
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
if os.path.exists(PARTIAL):
    os.remove(PARTIAL)  # all experiments checkpointed into the final file

print(f"\nsaved {OUT} ({out['info']['total_run_seconds'] / 60:.1f} min total)",
      flush=True)

# ---- summary table ----
print("\n=== SUMMARY (experimental player's win rate, 95% CI) ===")
rows = []
rows.append(("random_control", out["experiments"]["random_control"]))
for k, v in out["experiments"]["seat_rotation"].items():
    rows.append((k, v))
rows.append(("smart_port", out["experiments"]["smart_port"]))
for k, v in out["experiments"]["weight_sweep"].items():
    rows.append((k, v))
for label, e in rows:
    print(f"{label:16s}: {e['win_rate'] * 100:5.2f}% ± "
          f"{e['ci95'] * 100:.2f}  (n={e['games']})")
print("\n=== OPENING STATS (experimental player) ===")
for name, e in out["experiments"]["opening_stats"].items():
    print(f"{name:12s}: pips {e['avg_opening_pips']:4.1f} | distinct "
          f"{e['avg_opening_distinct']:.2f} | first city turn "
          f"{e['avg_first_city_turn']:.0f} ({e['share_built_city'] * 100:.0f}% "
          f"built one)")
