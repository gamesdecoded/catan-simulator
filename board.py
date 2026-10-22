"""
Static Catan board geometry (pure Python, computed once at import).

Builds the 19-hex / 54-intersection / 72-edge graph from axial coordinates,
plus a fixed 9-port layout on coastal edges.

Exports (numpy int64 arrays):
  HEX_VERTS   [19, 6]  vertex ids around each hex
  VERT_HEXES  [54, 3]  hex ids adjacent to each vertex (-1 pad)
  VERT_ADJ    [54, 3]  neighboring vertex ids (-1 pad)
  VERT_EDGES  [54, 3]  incident edge ids (-1 pad)
  EDGE_VERTS  [72, 2]  vertex ids of each edge
  PORT_AT_VERT[54]     0=none, 1=3:1, 2..6=2:1 wood/brick/sheep/wheat/ore
"""

import math
import numpy as np

# axial coords (q, r), radius 2 -> 19 hexes
HEXES = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]
assert len(HEXES) == 19

_CUBE_DIRS = [(1, -1, 0), (1, 0, -1), (0, 1, -1),
              (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]


def _cube(q, r):
    return (q, r, -q - r)


def _build():
    vmap = {}
    hex_verts = []
    for (q, r) in HEXES:
        c = _cube(q, r)
        vs = []
        for i in range(6):
            d1 = _CUBE_DIRS[i]
            d2 = _CUBE_DIRS[(i + 1) % 6]
            key = tuple(sorted([
                c,
                (c[0] + d1[0], c[1] + d1[1], c[2] + d1[2]),
                (c[0] + d2[0], c[1] + d2[1], c[2] + d2[2]),
            ]))
            if key not in vmap:
                vmap[key] = len(vmap)
            vs.append(vmap[key])
        hex_verts.append(vs)

    emap = {}
    for vs in hex_verts:
        for i in range(6):
            a, b = vs[i], vs[(i + 1) % 6]
            key = (a, b) if a < b else (b, a)
            if key not in emap:
                emap[key] = len(emap)

    n_vert = len(vmap)
    n_edge = len(emap)
    assert n_vert == 54, f"expected 54 vertices, got {n_vert}"
    assert n_edge == 72, f"expected 72 edges, got {n_edge}"

    HEX_VERTS = np.array(hex_verts, dtype=np.int64)

    VERT_HEXES = np.full((54, 3), -1, dtype=np.int64)
    for h, vs in enumerate(hex_verts):
        for v in vs:
            row = VERT_HEXES[v]
            row[np.where(row == -1)[0][0]] = h

    EDGE_VERTS = np.zeros((72, 2), dtype=np.int64)
    for (a, b), e in emap.items():
        EDGE_VERTS[e] = (a, b)

    VERT_ADJ = np.full((54, 3), -1, dtype=np.int64)
    VERT_EDGES = np.full((54, 3), -1, dtype=np.int64)
    for e in range(72):
        a, b = EDGE_VERTS[e]
        for v, u in ((a, b), (b, a)):
            row = VERT_ADJ[v]
            row[np.where(row == -1)[0][0]] = u
            erow = VERT_EDGES[v]
            erow[np.where(erow == -1)[0][0]] = e

    return HEX_VERTS, VERT_HEXES, VERT_ADJ, VERT_EDGES, EDGE_VERTS


HEX_VERTS, VERT_HEXES, VERT_ADJ, VERT_EDGES, EDGE_VERTS = _build()

# ---- ports: 9 fixed ports on coastal edges, evenly spaced ----
# port ids: 0=none, 1=3:1, 2=wood 2:1, 3=brick 2:1, 4=sheep 2:1, 5=wheat 2:1, 6=ore 2:1
PORT_TYPES = [1, 2, 1, 3, 1, 4, 1, 5, 6]  # 4x 3:1 + 5x 2:1


def _build_ports():
    coastal = (VERT_HEXES >= 0).sum(axis=1) < 3
    cedges = []
    for e in range(72):
        a, b = EDGE_VERTS[e]
        if coastal[a] and coastal[b]:
            # 2D position of edge midpoint (pointy-top axial layout)
            pts = []
            for v in (a, b):
                xs, ys, n = 0.0, 0.0, 0
                for k in range(3):
                    h = VERT_HEXES[v, k]
                    if h < 0:
                        continue
                    q, r = HEXES[h]
                    xs += math.sqrt(3) * (q + r / 2.0)
                    ys += 1.5 * r
                    n += 1
                pts.append((xs / n, ys / n))
            mx = (pts[0][0] + pts[1][0]) / 2.0
            my = (pts[0][1] + pts[1][1]) / 2.0
            cedges.append((math.atan2(my, mx), e))
    cedges.sort()
    n = len(cedges)
    chosen = []
    for i in range(9):
        idx = int(round(i * n / 9.0)) % n
        chosen.append(cedges[idx][1])
    # ensure no two chosen edges share a vertex (ports are spaced in real Catan)
    used = set()
    for e in chosen:
        a, b = EDGE_VERTS[e]
        assert a not in used and b not in used, "ports too close together"
        used.add(a)
        used.add(b)
    PORT_AT_VERT = np.zeros(54, dtype=np.int64)
    for e, pid in zip(chosen, PORT_TYPES):
        a, b = EDGE_VERTS[e]
        PORT_AT_VERT[a] = pid
        PORT_AT_VERT[b] = pid
    return PORT_AT_VERT


PORT_AT_VERT = _build_ports()


# ---- opening helpers ----

def validate_opening(v1, v2):
    """Check a candidate opening pair (v1 first pick, v2 second pick)."""
    assert 0 <= v1 < 54 and 0 <= v2 < 54, "vertex id out of range"
    assert v1 != v2, "openings must be two distinct intersections"
    assert v2 not in VERT_ADJ[v1], "opening settlements violate the distance rule"
    return True


def coastal_vertices():
    return [v for v in range(54) if (VERT_HEXES[v] >= 0).sum() < 3]


def describe_vertex(v):
    """Human-readable geometry description (board-independent)."""
    nh = int((VERT_HEXES[v] >= 0).sum())
    port = int(PORT_AT_VERT[v])
    port_names = {0: "no port", 1: "3:1 port", 2: "wood 2:1", 3: "brick 2:1",
                  4: "sheep 2:1", 5: "wheat 2:1", 6: "ore 2:1"}
    return (f"vertex {v}: {nh} adjacent hexes, "
            f"{'coastal' if nh < 3 else 'interior'}, {port_names[port]}")
