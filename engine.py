"""
4-player Catan Monte Carlo engine (numba-accelerated).

Experimental design: 4 identical heuristic bots play full games. The
experimental variable is Player 0's opening STRATEGY (parameter
`strat`), used for both of Player 0's snake-draft picks (picks 1 and 8
of the 1-2-3-4-4-3-2-1 draft). Players 1-3 always use the baseline
placement heuristic. Board (terrain + number tokens) is re-randomized
every game. Winner = first to 10 VP.

Strategy ids:
  0 baseline      current greedy heuristic (control)
  1 max_pips      highest expected resource pips
  2 diversity     most distinct resource types covered (joint across picks)
  3 port_seeking  port vertices first, then pips
  4 wheat_ore     wheat+ore pips first, then total pips

See METHODS.md for every simplification vs. real Catan.
"""

import numba
import numpy as np

from board import (HEX_VERTS, VERT_HEXES, VERT_ADJ, VERT_EDGES, EDGE_VERTS,
                   PORT_AT_VERT)

# ---- static data (frozen as numba constants) ----
PIP = np.zeros(13, dtype=np.int64)
for _n, _p in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5),
               (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)):
    PIP[_n] = _p

TERRAIN_MULTISET = np.array(
    [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
    dtype=np.int64)  # 0=desert,1=wood,2=brick,3=sheep,4=wheat,5=ore
NUMBER_MULTISET = np.array(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    dtype=np.int64)
DEV_DECK0 = np.array([14, 5, 2, 2, 2], dtype=np.int64)  # knight,vp,road,plenty,monopoly
# build costs: settlement, city, road, devcard; resources wood,brick,sheep,wheat,ore
COSTS = np.array([[1, 1, 1, 1, 0],
                  [0, 0, 0, 2, 3],
                  [1, 1, 0, 0, 0],
                  [1, 0, 1, 1, 1]], dtype=np.int64)
DRAFT = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int64)

MAX_TURNS = 1200
N_HEX, N_VERT, N_EDGE, N_P = 19, 54, 72, 4

# ---- opening strategies (Player 0's experimental variable) ----
STRATEGIES = {
    0: "baseline",
    1: "max_pips",
    2: "diversity",
    3: "port_seeking",
    4: "wheat_ore",
}

# Exact, auditable definitions of what each strategy does (also saved
# into results.json with every production run).
STRATEGY_DEFS = {
    "baseline": (
        "Control. Identical to the heuristic used by players 1-3: score each "
        "legal intersection as (sum of adjacent dice pips) + 1.5 * (distinct "
        "adjacent resource types, desert excluded); pick the highest score. "
        "Ties broken by lowest vertex id (deterministic)."
    ),
    "max_pips": (
        "Pick the legal intersection with the highest sum of adjacent dice "
        "pips (desert excluded, no diversity bonus). Applied independently "
        "to both draft picks. Ties broken by lowest vertex id."
    ),
    "diversity": (
        "First pick: the legal intersection adjacent to the most distinct "
        "resource types (desert excluded); ties broken by higher pip sum, "
        "then lowest vertex id. Second pick: the legal intersection "
        "maximizing the number of distinct resource types covered by the "
        "UNION of both picks' adjacent hexes; ties broken by higher pip sum "
        "of the second vertex, then lowest vertex id."
    ),
    "port_seeking": (
        "Prefer intersections touching a port (any of the 9 fixed ports: "
        "4x 3:1, 5x 2:1): among legal intersections, pick a port vertex "
        "first; among port vertices (or if none are legal) pick the highest "
        "adjacent pip sum. Ties broken by lowest vertex id."
    ),
    "wheat_ore": (
        "Prefer city-building resources: score each legal intersection by "
        "the sum of adjacent dice pips on WHEAT and ORE hexes only; ties "
        "broken by higher total adjacent pip sum (all resources), then "
        "lowest vertex id."
    ),
}


@numba.njit
def shuffle(a, n):
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp


@numba.njit
def setup_board(terrain, number):
    t = TERRAIN_MULTISET.copy()
    shuffle(t, 19)
    desert = 0
    for h in range(19):
        terrain[h] = t[h]
        if t[h] == 0:
            desert = h
    nums = NUMBER_MULTISET.copy()
    shuffle(nums, 18)
    k = 0
    for h in range(19):
        if h == desert:
            number[h] = 0
        else:
            number[h] = nums[k]
            k += 1
    return desert


@numba.njit
def vertex_score(v, terrain, number):
    pips = 0
    mask = 0
    for k in range(3):
        h = VERT_HEXES[v, k]
        if h < 0:
            continue
        t = terrain[h]
        if t == 0:
            continue
        pips += PIP[number[h]]
        mask |= (1 << (t - 1))
    distinct = 0
    m = mask
    while m:
        distinct += m & 1
        m >>= 1
    return pips + 1.5 * distinct


@numba.njit
def can_settle(v, vert_owner):
    if vert_owner[v] != -1:
        return False
    for k in range(3):
        u = VERT_ADJ[v, k]
        if u >= 0 and vert_owner[u] != -1:
            return False
    return True


@numba.njit
def net_vertex(v, p, vert_owner, edge_owner):
    if vert_owner[v] == p:
        return True
    for k in range(3):
        e = VERT_EDGES[v, k]
        if e >= 0 and edge_owner[e] == p:
            return True
    return False


@numba.njit
def best_spot(p, vert_owner, edge_owner, terrain, number, need_road):
    best = -1
    best_s = -1.0
    for v in range(N_VERT):
        if not can_settle(v, vert_owner):
            continue
        if need_road and not net_vertex(v, p, vert_owner, edge_owner):
            continue
        s = vertex_score(v, terrain, number)
        if s > best_s:
            best_s = s
            best = v
    return best


@numba.njit
def vertex_pips(v, terrain, number):
    """Sum of adjacent dice pips (desert excluded)."""
    pips = 0
    for k in range(3):
        h = VERT_HEXES[v, k]
        if h < 0:
            continue
        if terrain[h] == 0:
            continue
        pips += PIP[number[h]]
    return pips


@numba.njit
def vertex_mask(v, terrain):
    """Bitmask of adjacent resource types (bit t-1 for terrain t; desert excluded)."""
    mask = 0
    for k in range(3):
        h = VERT_HEXES[v, k]
        if h < 0:
            continue
        t = terrain[h]
        if t == 0:
            continue
        mask |= (1 << (t - 1))
    return mask


@numba.njit
def popcount(m):
    c = 0
    while m:
        c += m & 1
        m >>= 1
    return c


@numba.njit
def wheat_ore_pips(v, terrain, number):
    """Sum of adjacent dice pips on wheat (4) and ore (5) hexes only."""
    pips = 0
    for k in range(3):
        h = VERT_HEXES[v, k]
        if h < 0:
            continue
        t = terrain[h]
        if t == 4 or t == 5:
            pips += PIP[number[h]]
    return pips


@numba.njit
def pick_opening(strat, p0_first, vert_owner, edge_owner, terrain, number):
    """Player 0's snake-draft pick under opening strategy `strat`.

    p0_first: Player 0's first-pick vertex, or -1 when making the first pick
    (used by the diversity strategy to maximize joint coverage).
    Deterministic given the board: vertices scanned in id order with strict
    `>` comparisons, so the lowest vertex id wins every tie.
    """
    best = -1
    b1 = -1
    b2 = -1
    first_mask = vertex_mask(p0_first, terrain) if p0_first >= 0 else 0
    for v in range(N_VERT):
        if not can_settle(v, vert_owner):
            continue
        # opening picks never need a road connection (matches old behavior)
        if strat == 0:
            # baseline: pips + 1.5 * distinct  <=>  2*pips + 3*distinct
            # (exact integer ordering, no float rounding)
            s1 = vertex_pips(v, terrain, number)
            s2 = popcount(vertex_mask(v, terrain))
            k1 = 2 * s1 + 3 * s2
            k2 = 0
        elif strat == 1:
            # max_pips
            k1 = vertex_pips(v, terrain, number)
            k2 = 0
        elif strat == 2:
            # diversity: distinct types (joint across both picks on 2nd pick)
            if p0_first < 0:
                k1 = popcount(vertex_mask(v, terrain))
            else:
                k1 = popcount(first_mask | vertex_mask(v, terrain))
            k2 = vertex_pips(v, terrain, number)
        elif strat == 3:
            # port_seeking: port vertex first, then pips
            k1 = 1 if PORT_AT_VERT[v] > 0 else 0
            k2 = vertex_pips(v, terrain, number)
        else:
            # wheat_ore: wheat+ore pips first, then total pips
            k1 = wheat_ore_pips(v, terrain, number)
            k2 = vertex_pips(v, terrain, number)
        if k1 > b1 or (k1 == b1 and k2 > b2):
            b1 = k1
            b2 = k2
            best = v
    return best


@numba.njit
def best_opening_road(v, edge_owner, terrain, number):
    best = -1
    best_s = -1.0
    for k in range(3):
        e = VERT_EDGES[v, k]
        if e < 0 or edge_owner[e] != -1:
            continue
        a = EDGE_VERTS[e, 0]
        b = EDGE_VERTS[e, 1]
        u = b if a == v else a
        s = vertex_score(u, terrain, number)
        if s > best_s:
            best_s = s
            best = e
    return best


@numba.njit
def best_road_spot(p, vert_owner, edge_owner, terrain, number):
    best = -1
    best_s = -1.0
    for e in range(N_EDGE):
        if edge_owner[e] != -1:
            continue
        a = EDGE_VERTS[e, 0]
        b = EDGE_VERTS[e, 1]
        na = net_vertex(a, p, vert_owner, edge_owner)
        nb = net_vertex(b, p, vert_owner, edge_owner)
        if not (na or nb):
            continue
        if na and not nb:
            s = vertex_score(b, terrain, number)
        elif nb and not na:
            s = vertex_score(a, terrain, number)
        else:
            s = 0.0
        if s > best_s:
            best_s = s
            best = e
    return best


@numba.njit
def dfs_lr(v, adj_a, adj_b, m, blocked, vis, depth):
    best = depth
    for i in range(m):
        if (vis >> np.int64(i)) & np.int64(1):
            continue
        a = adj_a[i]
        b = adj_b[i]
        if a == v:
            u = b
        elif b == v:
            u = a
        else:
            continue
        if blocked[u]:
            if depth + 1 > best:
                best = depth + 1
            continue
        r = dfs_lr(u, adj_a, adj_b, m, blocked,
                   vis | (np.int64(1) << np.int64(i)), depth + 1)
        if r > best:
            best = r
    return best


@numba.njit
def longest_road_len(p, edge_owner, vert_owner):
    adj_a = np.zeros(16, dtype=np.int64)
    adj_b = np.zeros(16, dtype=np.int64)
    starts = np.zeros(N_VERT, dtype=np.int64)
    m = 0
    for e in range(N_EDGE):
        if edge_owner[e] == p:
            a = EDGE_VERTS[e, 0]
            b = EDGE_VERTS[e, 1]
            adj_a[m] = a
            adj_b[m] = b
            m += 1
            starts[a] = 1
            starts[b] = 1
    if m == 0:
        return 0
    blocked = np.zeros(N_VERT, dtype=np.int64)
    for v in range(N_VERT):
        o = vert_owner[v]
        if o != -1 and o != p:
            blocked[v] = 1
    best = 0
    for s in range(N_VERT):
        if starts[s]:
            r = dfs_lr(s, adj_a, adj_b, m, blocked, np.int64(0), 0)
            if r > best:
                best = r
    return best


@numba.njit
def victory_points(p, n_sett, n_city, vp_cards, lr_holder, army_holder):
    vp = n_sett[p] + 2 * n_city[p] + vp_cards[p]
    if lr_holder[0] == p:
        vp += 2
    if army_holder[0] == p:
        vp += 2
    return vp


@numba.njit
def vp_leader_excluding(p, n_sett, n_city, vp_cards, lr_holder, army_holder):
    best = -1
    best_vp = -1
    for q in range(N_P):
        if q == p:
            continue
        vp = victory_points(q, n_sett, n_city, vp_cards, lr_holder, army_holder)
        if vp > best_vp:
            best_vp = vp
            best = q
    return best


@numba.njit
def move_robber(p, terrain, number, vert_owner, robber_hex, leader):
    best_h = -1
    best_s = -1.0
    for h in range(N_HEX):
        if h == robber_hex:
            continue
        opp = False
        lead = False
        for k in range(6):
            v = HEX_VERTS[h, k]
            o = vert_owner[v]
            if o != -1 and o != p:
                opp = True
                if o == leader:
                    lead = True
        if not opp:
            continue
        s = float(PIP[number[h]]) * (2.0 if lead else 1.0)
        if s > best_s:
            best_s = s
            best_h = h
    if best_h == -1:
        for h in range(N_HEX):
            if h == robber_hex:
                continue
            s = float(PIP[number[h]])
            if s > best_s:
                best_s = s
                best_h = h
    return best_h


@numba.njit
def steal_from_hex(p, h, vert_owner, res):
    cands = np.zeros(4, dtype=np.int64)
    nc = 0
    for k in range(6):
        v = HEX_VERTS[h, k]
        q = vert_owner[v]
        if q != -1 and q != p:
            already = False
            for j in range(nc):
                if cands[j] == q:
                    already = True
                    break
            if not already:
                tot = 0
                for t in range(5):
                    tot += res[q, t]
                if tot > 0:
                    cands[nc] = q
                    nc += 1
    if nc == 0:
        return
    q = cands[np.random.randint(0, nc)]
    tot = 0
    for t in range(5):
        tot += res[q, t]
    r = np.random.randint(0, tot)
    for t in range(5):
        if r < res[q, t]:
            res[q, t] -= 1
            res[p, t] += 1
            return
        r -= res[q, t]


@numba.njit
def discard_half(p, res, destroyed):
    """Real Catan: on a 7, every player holding MORE THAN 7 resource cards
    discards half of them, rounded down. Discards the most-abundant
    resource type first. Returns cards discarded."""
    tot = 0
    for t in range(5):
        tot += res[p, t]
    if tot <= 7:
        return 0
    disc = tot // 2
    for _ in range(disc):
        bt = 0
        bc = -1
        for t in range(5):
            if res[p, t] > bc:
                bc = res[p, t]
                bt = t
        res[p, bt] -= 1
    destroyed[0] += disc
    return disc


@numba.njit
def distribute(roll, terrain, number, robber_hex, vert_owner, is_city, res,
               created):
    for h in range(N_HEX):
        if number[h] != roll or h == robber_hex:
            continue
        t = terrain[h] - 1
        if t < 0:
            continue
        for k in range(6):
            v = HEX_VERTS[h, k]
            o = vert_owner[v]
            if o == -1:
                continue
            n = 2 if is_city[v] else 1
            res[o, t] += n
            created[0] += n


@numba.njit
def trade_rate(p, t, ports):
    pm = ports[p]
    if pm & (1 << (t + 1)):
        return 2
    if pm & 1:
        return 3
    return 4


@numba.njit
def bank_trades_for(p, res, ports, cost, destroyed):
    """Trade with the bank (4:1, 3:1/2:1 with ports) to afford `cost`.
    Only trades surplus resources; never breaks the build's own cost."""
    for _ in range(8):
        d = -1
        for t in range(5):
            if res[p, t] < cost[t]:
                d = t
                break
        if d == -1:
            return True
        done = False
        for s in range(5):
            if s == d:
                continue
            rate = trade_rate(p, s, ports)
            if res[p, s] - cost[s] >= rate:
                res[p, s] -= rate
                res[p, d] += 1
                destroyed[0] += rate - 1
                done = True
                break
        if not done:
            return False
    for t in range(5):
        if res[p, t] < cost[t]:
            return False
    return True


@numba.njit
def pay_cost(p, res, cost, destroyed):
    s = 0
    for t in range(5):
        res[p, t] -= cost[t]
        s += cost[t]
    destroyed[0] += s


@numba.njit
def add_port(p, v, ports):
    pid = PORT_AT_VERT[v]
    if pid == 1:
        ports[p] |= 1
    elif pid > 1:
        ports[p] |= (1 << (pid - 1))


@numba.njit
def draw_dev(deck):
    tot = 0
    for i in range(5):
        tot += deck[i]
    r = np.random.randint(0, tot)
    for i in range(5):
        if r < deck[i]:
            deck[i] -= 1
            return i
        r -= deck[i]
    return 0


@numba.njit
def do_plenty(p, res, created):
    # 2 of the resource with the largest deficit toward a settlement;
    # ties broken ore > wheat > wood > brick > sheep
    best_t = 4
    best_d = -1
    for t in (4, 3, 0, 1, 2):
        d = COSTS[0, t] - res[p, t]
        if d < 0:
            d = 0
        if d > best_d:
            best_d = d
            best_t = t
    res[p, best_t] += 2
    created[0] += 2


@numba.njit
def do_monopoly(p, res):
    best_t = 4
    best_g = -1
    for t in (4, 3, 2, 1, 0):
        g = 0
        for q in range(N_P):
            if q != p:
                g += res[q, t]
        if g > best_g:
            best_g = g
            best_t = t
    for q in range(N_P):
        if q != p:
            res[p, best_t] += res[q, best_t]
            res[q, best_t] = 0


@numba.njit
def update_longest_road(p, edge_owner, vert_owner, lr_holder, lr_len):
    L = longest_road_len(p, edge_owner, vert_owner)
    if L >= 5 and L > lr_len[0]:
        lr_holder[0] = p
        lr_len[0] = L


@numba.njit
def try_build(p, btype, terrain, number, vert_owner, is_city, edge_owner, res,
              ports, n_sett, sett_verts, n_city, city_verts, n_road,
              knights_new, vp_cards, deck, lr_holder, lr_len, created,
              destroyed):
    """Build priority handled by caller order. Returns True if built."""
    if btype == 0:  # settlement
        if n_sett[p] >= 5:
            return False
        v = best_spot(p, vert_owner, edge_owner, terrain, number, True)
        if v < 0:
            return False
        if not bank_trades_for(p, res, ports, COSTS[0], destroyed):
            return False
        pay_cost(p, res, COSTS[0], destroyed)
        vert_owner[v] = p
        sett_verts[p, n_sett[p]] = v
        n_sett[p] += 1
        add_port(p, v, ports)
        return True
    if btype == 1:  # city: upgrade best settlement
        if n_city[p] >= 4 or n_sett[p] == 0:
            return False
        bv = -1
        bi = -1
        bs = -1.0
        for i in range(n_sett[p]):
            v = sett_verts[p, i]
            s = vertex_score(v, terrain, number)
            if s > bs:
                bs = s
                bv = v
                bi = i
        if not bank_trades_for(p, res, ports, COSTS[1], destroyed):
            return False
        pay_cost(p, res, COSTS[1], destroyed)
        n_sett[p] -= 1
        sett_verts[p, bi] = sett_verts[p, n_sett[p]]
        city_verts[p, n_city[p]] = bv
        n_city[p] += 1
        is_city[bv] = 1
        return True
    if btype == 2:  # road
        if n_road[p] >= 15:
            return False
        e = best_road_spot(p, vert_owner, edge_owner, terrain, number)
        if e < 0:
            return False
        if not bank_trades_for(p, res, ports, COSTS[2], destroyed):
            return False
        pay_cost(p, res, COSTS[2], destroyed)
        edge_owner[e] = p
        n_road[p] += 1
        update_longest_road(p, edge_owner, vert_owner, lr_holder, lr_len)
        return True
    # btype == 3: dev card
    deck_tot = 0
    for _i in range(5):
        deck_tot += deck[_i]
    if deck_tot == 0:
        return False
    if not bank_trades_for(p, res, ports, COSTS[3], destroyed):
        return False
    pay_cost(p, res, COSTS[3], destroyed)
    d = draw_dev(deck)
    if d == 0:
        knights_new[p] += 1
    elif d == 1:
        vp_cards[p] += 1
    elif d == 2:
        for _ in range(2):
            e = best_road_spot(p, vert_owner, edge_owner, terrain, number)
            if e < 0 or n_road[p] >= 15:
                break
            edge_owner[e] = p
            n_road[p] += 1
            update_longest_road(p, edge_owner, vert_owner, lr_holder, lr_len)
    elif d == 3:
        do_plenty(p, res, created)
    else:
        do_monopoly(p, res)
    return True


@numba.njit
def do_turn(p, terrain, number, vert_owner, is_city, edge_owner, res, ports,
            n_sett, sett_verts, n_city, city_verts, n_road, knights_old,
            knights_new, knights_played, vp_cards, deck, robber, lr_holder,
            lr_len, army_holder, army_count, created, destroyed):
    # merge newly drawn knights; play one knight if held
    knights_old[p] += knights_new[p]
    knights_new[p] = 0
    if knights_old[p] > 0:
        knights_old[p] -= 1
        knights_played[p] += 1
        leader = vp_leader_excluding(p, n_sett, n_city, vp_cards, lr_holder,
                                     army_holder)
        h = move_robber(p, terrain, number, vert_owner, robber[0], leader)
        robber[0] = h
        steal_from_hex(p, h, vert_owner, res)
        kp = knights_played[p]
        if kp >= 3 and kp > army_count[0]:
            army_holder[0] = p
            army_count[0] = kp
        if victory_points(p, n_sett, n_city, vp_cards, lr_holder,
                           army_holder) >= 10:
            return True
    roll = np.random.randint(1, 7) + np.random.randint(1, 7)
    if roll == 7:
        for q in range(N_P):
            discard_half(q, res, destroyed)
        leader = vp_leader_excluding(p, n_sett, n_city, vp_cards, lr_holder,
                                     army_holder)
        h = move_robber(p, terrain, number, vert_owner, robber[0], leader)
        robber[0] = h
        steal_from_hex(p, h, vert_owner, res)
    else:
        distribute(roll, terrain, number, robber[0], vert_owner, is_city, res,
                   created)
    # build in priority order: settlement > city > road > dev card
    for _ in range(12):
        acted = False
        if try_build(p, 0, terrain, number, vert_owner, is_city, edge_owner,
                     res, ports, n_sett, sett_verts, n_city, city_verts,
                     n_road, knights_new, vp_cards, deck, lr_holder, lr_len,
                     created, destroyed):
            acted = True
        elif try_build(p, 1, terrain, number, vert_owner, is_city, edge_owner,
                       res, ports, n_sett, sett_verts, n_city, city_verts,
                       n_road, knights_new, vp_cards, deck, lr_holder, lr_len,
                       created, destroyed):
            acted = True
        elif try_build(p, 2, terrain, number, vert_owner, is_city, edge_owner,
                       res, ports, n_sett, sett_verts, n_city, city_verts,
                       n_road, knights_new, vp_cards, deck, lr_holder, lr_len,
                       created, destroyed):
            acted = True
        elif try_build(p, 3, terrain, number, vert_owner, is_city, edge_owner,
                       res, ports, n_sett, sett_verts, n_city, city_verts,
                       n_road, knights_new, vp_cards, deck, lr_holder, lr_len,
                       created, destroyed):
            acted = True
        if victory_points(p, n_sett, n_city, vp_cards, lr_holder,
                           army_holder) >= 10:
            return True
        if not acted:
            break
    return victory_points(p, n_sett, n_city, vp_cards, lr_holder,
                           army_holder) >= 10


@numba.njit
def place_opening(strat, terrain, number, vert_owner, edge_owner, res, ports,
                  n_sett, sett_verts, n_road, created):
    """Snake-draft opening (1-2-3-4-4-3-2-1). Player 0 uses opening strategy
    `strat` for both picks; players 1-3 use the baseline placement heuristic.
    Each pick also places one road (best_opening_road); the second pick
    grants starting resources (1 per adjacent non-desert hex)."""
    p0_first = -1
    second = np.zeros(4, dtype=np.int64)
    for idx in range(8):
        p = DRAFT[idx]
        if p == 0:
            v = pick_opening(strat, p0_first, vert_owner, edge_owner,
                             terrain, number)
            if idx == 0:
                p0_first = v
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
def simulate_one(strat, stats):
    """Play one full game. stats[12]: turns, capped, unused(0), created,
    destroyed, held, min_ok, conserve_ok, winner_vp, lr_awarded,
    army_awarded, devs_bought. Returns winner 0..3."""
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
    place_opening(strat, terrain, number, vert_owner, edge_owner, res, ports,
                  n_sett, sett_verts, n_road, created)

    winner = -1
    turn = 0
    while turn < MAX_TURNS:
        p = turn % N_P
        if do_turn(p, terrain, number, vert_owner, is_city, edge_owner, res,
                   ports, n_sett, sett_verts, n_city, city_verts, n_road,
                   knights_old, knights_new, knights_played, vp_cards, deck,
                   robber, lr_holder, lr_len, army_holder, army_count,
                   created, destroyed):
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
    stats[2] = 0  # unused (was v2-fallback in the fixed-opening design)
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
    return winner


@numba.njit
def simulate_many(strat, n_games, seed, wins, stat_sum):
    np.random.seed(seed)
    for g in range(n_games):
        stats = np.zeros(12, dtype=np.int64)
        w = simulate_one(strat, stats)
        wins[w] += 1
        for i in range(12):
            stat_sum[i] += stats[i]
