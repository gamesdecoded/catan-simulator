"""Optional insert experiments for the Catan video (script-v2.md).

Builds on engine.py (validated, untouched). The experimental player is now
a parameter `exp` (0..3); everyone else plays the baseline heuristic. The
experimental variable for `exp` is `mode`:

  mode 0: production opening strategy `prod_strat` (0..4). Used for seat
          rotation (prod_strat=0) and for opening-stats logging per strategy.
  mode 1: random_control. Two settlements at uniformly random legal
          intersections, then baseline heuristic play for the rest of the game.
  mode 2: smart_port. Prefers port vertices whose port matches a resource
          the vertex actually produces: a 2:1 port counts as matched only if
          an adjacent hex produces that port's resource; a 3:1 port counts
          as matched if the vertex touches any produced resource. Among
          matched port vertices, highest adjacent pip sum wins. Falls back
          to plain port-seeking if no matched port vertex is legal.
  mode 3: weighted diversity. score = pips + w * distinct(joint union),
          ties broken by lowest vertex id. w=0.0 reproduces max_pips
          (identical tie-breaking); the existing lexicographic "diversity"
          strategy (prod_strat=2) is the diversity-first sweep point.

Opening instrumentation per game (experimental player only):
  opening[0] = sum of adjacent pips across both opening settlements
  opening[1] = distinct resource types covered by both openings (union)
  opening[2] = global turn index of the experimental player's first city
               (-1 if they never built one)
  opening[3] = 1 if they built at least one city, else 0
"""

import numba
import numpy as np

from engine import (
    can_settle, vertex_pips, vertex_mask, popcount,
    best_spot, best_opening_road, setup_board, do_turn, victory_points,
    add_port, pick_opening,
    DRAFT, N_HEX, N_VERT, N_EDGE, N_P, MAX_TURNS, DEV_DECK0,
)
from board import (
    HEX_VERTS, VERT_HEXES, VERT_ADJ, VERT_EDGES, EDGE_VERTS, PORT_AT_VERT,
)


@numba.njit
def pick_smart_port(vert_owner, terrain, number):
    """Pass 1: matched port vertices (port matches a produced resource),
    highest pips wins. Pass 2 (fallback): plain port_seeking."""
    best = -1
    bp = -1
    found = False
    for v in range(N_VERT):
        if not can_settle(v, vert_owner):
            continue
        pid = PORT_AT_VERT[v]
        if pid == 0:
            continue
        mask = vertex_mask(v, terrain)
        if pid == 1:
            ok = mask != 0
        else:
            ok = ((mask >> np.int64(pid - 2)) & np.int64(1)) == 1
        if not ok:
            continue
        p = vertex_pips(v, terrain, number)
        if p > bp:
            bp = p
            best = v
            found = True
    if found:
        return best
    # fallback: plain port_seeking (port vertex first, then pips)
    best = -1
    b1 = -1
    b2 = -1
    for v in range(N_VERT):
        if not can_settle(v, vert_owner):
            continue
        k1 = 1 if PORT_AT_VERT[v] > 0 else 0
        k2 = vertex_pips(v, terrain, number)
        if k1 > b1 or (k1 == b1 and k2 > b2):
            b1 = k1
            b2 = k2
            best = v
    return best


@numba.njit
def pick_xp(mode, prod_strat, w, exp_first, vert_owner, edge_owner,
            terrain, number):
    if mode == 0:
        return pick_opening(prod_strat, exp_first, vert_owner, edge_owner,
                            terrain, number)
    if mode == 1:
        # random_control: uniformly random legal intersection
        cnt = 0
        for v in range(N_VERT):
            if can_settle(v, vert_owner):
                cnt += 1
        r = np.random.randint(0, cnt)
        k = 0
        for v in range(N_VERT):
            if can_settle(v, vert_owner):
                if k == r:
                    return v
                k += 1
        return -1  # unreachable: opening board always has legal vertices
    if mode == 2:
        return pick_smart_port(vert_owner, terrain, number)
    # mode 3: weighted diversity; joint union on the second pick
    first_mask = vertex_mask(exp_first, terrain) if exp_first >= 0 else 0
    best = -1
    bs = -1.0
    for v in range(N_VERT):
        if not can_settle(v, vert_owner):
            continue
        mask = vertex_mask(v, terrain)
        if exp_first >= 0:
            mask = first_mask | mask
        s = float(vertex_pips(v, terrain, number)) + w * float(popcount(mask))
        if s > bs:
            bs = s
            best = v
    return best


@numba.njit
def place_opening_xp(exp, mode, prod_strat, w, terrain, number, vert_owner,
                     edge_owner, res, ports, n_sett, sett_verts, n_road,
                     created):
    """Snake-draft opening (1-2-3-4-4-3-2-1). Player `exp` uses the insert
    strategy; players 1-3 (other than exp) use the baseline heuristic."""
    exp_first = -1
    second = np.zeros(4, dtype=np.int64)
    for idx in range(8):
        p = DRAFT[idx]
        if p == exp:
            v = pick_xp(mode, prod_strat, w, exp_first, vert_owner,
                        edge_owner, terrain, number)
            if exp_first < 0:
                exp_first = v
        else:
            v = best_spot(p, vert_owner, edge_owner, terrain, number, False)
        vert_owner[v] = p
        sett_verts[p, n_sett[p]] = v
        n_sett[p] += 1
        add_port(p, v, ports)
        e = best_opening_road(v, edge_owner, terrain, number)
        edge_owner[e] = p
        n_road[p] += 1
        if idx >= 4:
            second[p] = v
    for p in range(N_P):
        v = second[p]
        for k in range(3):
            h = VERT_HEXES[v, k]
            if h >= 0 and terrain[h] > 0:
                t = terrain[h] - 1
                res[p, t] += 1
                created[0] += 1


@numba.njit
def simulate_one_xp(exp, mode, prod_strat, w, stats, opening):
    """Same game loop as engine.simulate_one, plus opening instrumentation
    for the experimental player. Returns winner 0..3."""
    terrain = np.zeros(N_HEX, dtype=np.int64)
    number = np.zeros(N_HEX, dtype=np.int64)
    res = np.zeros((N_P, 5), dtype=np.int64)
    vert_owner = np.full(N_VERT, -1, dtype=np.int64)
    is_city = np.zeros(N_VERT, dtype=np.int64)
    edge_owner = np.full(N_EDGE, -1, dtype=np.int64)
    ports = np.zeros(N_P, dtype=np.int64)
    n_sett = np.zeros(N_P, dtype=np.int64)
    sett_verts = np.zeros((N_P, 5), dtype=np.int64)
    n_city = np.zeros(N_P, dtype=np.int64)
    city_verts = np.zeros((N_P, 4), dtype=np.int64)
    n_road = np.zeros(N_P, dtype=np.int64)
    knights_old = np.zeros(N_P, dtype=np.int64)
    knights_new = np.zeros(N_P, dtype=np.int64)
    knights_played = np.zeros(N_P, dtype=np.int64)
    vp_cards = np.zeros(N_P, dtype=np.int64)
    deck = DEV_DECK0.copy()
    robber = np.zeros(1, dtype=np.int64)
    lr_holder = np.full(1, -1, dtype=np.int64)
    lr_len = np.zeros(1, dtype=np.int64)
    army_holder = np.full(1, -1, dtype=np.int64)
    army_count = np.zeros(1, dtype=np.int64)
    created = np.zeros(1, dtype=np.int64)
    destroyed = np.zeros(1, dtype=np.int64)

    robber[0] = setup_board(terrain, number)
    place_opening_xp(exp, mode, prod_strat, w, terrain, number, vert_owner,
                     edge_owner, res, ports, n_sett, sett_verts, n_road,
                     created)

    v1 = sett_verts[exp, 0]
    v2 = sett_verts[exp, 1]
    opening[0] = vertex_pips(v1, terrain, number) + vertex_pips(v2, terrain,
                                                               number)
    opening[1] = popcount(vertex_mask(v1, terrain) | vertex_mask(v2, terrain))

    fct = -1
    nc0 = 0
    winner = -1
    turn = 0
    while turn < MAX_TURNS:
        p = turn % N_P
        won = do_turn(p, terrain, number, vert_owner, is_city, edge_owner,
                      res, ports, n_sett, sett_verts, n_city, city_verts,
                      n_road, knights_old, knights_new, knights_played,
                      vp_cards, deck, robber, lr_holder, lr_len, army_holder,
                      army_count, created, destroyed)
        if n_city[exp] > nc0:
            nc0 = n_city[exp]
            if fct < 0:
                fct = turn
        if won:
            winner = p
            break
        turn += 1

    capped = 0
    if winner == -1:
        capped = 1
        best = 0
        best_vp = -1
        for q in range(N_P):
            vp = victory_points(q, n_sett, n_city, vp_cards, lr_holder,
                                army_holder)
            if vp > best_vp:
                best_vp = vp
                best = q
        winner = best

    held = 0
    min_ok = 1
    for q in range(N_P):
        for t in range(5):
            held += res[q, t]
            if res[q, t] < 0:
                min_ok = 0
    conserve_ok = 1 if created[0] - destroyed[0] == held else 0

    stats[0] = turn + 1
    stats[1] = capped
    stats[2] = 0
    stats[3] = created[0]
    stats[4] = destroyed[0]
    stats[5] = held
    stats[6] = min_ok
    stats[7] = conserve_ok
    stats[8] = victory_points(winner, n_sett, n_city, vp_cards, lr_holder,
                              army_holder)
    stats[9] = 1 if lr_holder[0] != -1 else 0
    stats[10] = 1 if army_holder[0] != -1 else 0
    devs_left = 0
    for i in range(5):
        devs_left += deck[i]
    stats[11] = 25 - devs_left

    opening[2] = fct
    opening[3] = 1 if fct >= 0 else 0
    return winner


@numba.njit
def simulate_many_xp(exp, mode, prod_strat, w, n_games, seed, wins, stat_sum,
                     open_sum):
    np.random.seed(seed)
    for g in range(n_games):
        stats = np.zeros(12, dtype=np.int64)
        opening = np.zeros(4, dtype=np.int64)
        winner = simulate_one_xp(exp, mode, prod_strat, w, stats, opening)
        wins[winner] += 1
        for i in range(12):
            stat_sum[i] += stats[i]
        open_sum[0] += opening[0]
        open_sum[1] += opening[1]
        if opening[3] == 1:
            open_sum[2] += opening[2]
            open_sum[3] += 1
