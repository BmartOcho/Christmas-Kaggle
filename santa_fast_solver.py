import math
import random
import csv
from typing import Dict, List, Tuple

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#  BASIC CONFIG
# =========================

MAX_N = 200
FAST_DEEP_LIMIT = 15   # number of Ns to refine
GLOBAL_SEED = 42

# =========================
#  TREE SHAPE
# =========================

TREE_POLYGON = Polygon([
    (0.0, 0.8),
    (0.25/2, 0.5),
    (0.25/4, 0.5),
    (0.4/2, 0.25),
    (0.4/4, 0.25),
    (0.7/2, 0.0),
    (0.15/2, 0.0),
    (0.15/2, -0.2),
    (-0.15/2, -0.2),
    (-0.15/2, 0.0),
    (-0.7/2, 0.0),
    (-0.4/4, 0.25),
    (-0.4/2, 0.25),
    (-0.25/4, 0.5),
    (-0.25/2, 0.5)
])

def place_tree(x, y, deg):
    p = rotate(TREE_POLYGON, deg, origin=(0, 0))
    return translate(p, xoff=x, yoff=y)

def layout_to_polys(layout):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]

def bounding_side(polys):
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)

def has_overlap(poly, polys, skip):
    for j, other in enumerate(polys):
        if j == skip:
            continue
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False

def group_contrib(side, n):
    return (side*side)/n

# =========================
#  BASIC NESTED LAYOUT
# =========================

def nested_layout(n, spacing=0.82):
    """Very fast heuristic grid-based placement."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    layout = []
    idx = 0

    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            deg = 0 if (r + c) % 2 == 0 else 180
            x = c * spacing
            y = r * spacing
            if r % 2 == 1:
                x += spacing*0.45
            layout.append((x, y, deg))
            idx += 1

    xs = [x for x,_,_ in layout]
    ys = [y for _,y,_ in layout]
    cx = (max(xs)+min(xs))/2
    cy = (max(ys)+min(ys))/2
    return [(x-cx, y-cy, deg) for (x,y,deg) in layout]

# =========================
#  FAST COLLISION CLEANUP
# =========================

def cleanup(layout, iters=30, push=0.015):
    layout = layout[:]
    polys = layout_to_polys(layout)

    for _ in range(iters):
        moved = False
        for i in range(len(layout)):
            for j in range(i+1, len(layout)):
                p1 = polys[i]
                p2 = polys[j]
                if p1.intersects(p2) and not p1.touches(p2):
                    x1,y1,d1 = layout[i]
                    x2,y2,d2 = layout[j]
                    dx = x1-x2
                    dy = y1-y2
                    dist = math.hypot(dx,dy) or 1
                    push_amt = push/dist
                    x1 += dx*push_amt
                    y1 += dy*push_amt
                    x2 -= dx*push_amt
                    y2 -= dy*push_amt
                    layout[i] = (x1,y1,d1)
                    layout[j] = (x2,y2,d2)
                    polys[i] = place_tree(x1,y1,d1)
                    polys[j] = place_tree(x2,y2,d2)
                    moved = True
        if not moved:
            break

    return layout

# =========================
#  VERY LIGHT ANNEALING
# =========================

def light_anneal(layout, steps=800):
    layout = layout[:]
    polys = layout_to_polys(layout)
    curr = bounding_side(polys)
    best_layout = layout[:]
    best = curr

    for _ in range(steps):
        i = random.randint(0, len(layout)-1)
        x,y,deg = layout[i]
        nx = x + random.uniform(-0.01, 0.01)
        ny = y + random.uniform(-0.01, 0.01)
        ndeg = (deg + random.uniform(-2,2)) % 360
        npoly = place_tree(nx,ny,ndeg)

        if has_overlap(npoly, polys, i):
            continue

        old_poly = polys[i]
        polys[i] = npoly
        new_side = bounding_side(polys)

        if new_side <= curr:
            layout[i] = (nx, ny, ndeg)
            curr = new_side
            if new_side < best:
                best = new_side
                best_layout = layout[:]
        else:
            polys[i] = old_poly

    return best_layout, best

# =========================
#  PHASE 1: SLOPPY PASS
# =========================

def sloppy_pass():
    result = {}
    for n in range(1, MAX_N+1):
        lay = nested_layout(n)
        lay = cleanup(lay, iters=20, push=0.015)
        side = bounding_side(layout_to_polys(lay))
        contrib = group_contrib(side,n)
        result[n] = (lay, side, contrib)
    return result

# =========================
#  SELECT WORST NS
# =========================

def pick_worst(sloppy, k=FAST_DEEP_LIMIT):
    items = [(n, sloppy[n][2]) for n in sloppy]
    items.sort(key=lambda x: x[1], reverse=True)
    top = [n for n,_ in items[:k]]
    top.sort()
    return top

# =========================
#  FAST DEEP REFINEMENT
# =========================

def fast_refine(n, base):
    """shallow local optimization"""
    layout = base[:]
    layout = cleanup(layout, iters=20, push=0.02)
    refined, _ = light_anneal(layout, steps=900)
    return refined

# =========================
#  BUILD FINAL LAYOUTS
# =========================

def build_final(sloppy, refined_ns, refined_sets):
    final = {}
    deep_ns = sorted(refined_ns)

    def nearest(n):
        for dn in deep_ns:
            if n <= dn:
                return dn
        return deep_ns[-1]

    for n in range(1, MAX_N+1):
        if n in refined_sets:
            final[n] = refined_sets[n]
        else:
            parent = nearest(n)
            base = refined_sets[parent][:n]
            base = [(x*1.01, y*1.01, deg) for (x,y,deg) in base]
            base = cleanup(base, iters=15, push=0.01)
            out, _ = light_anneal(base, steps=400)
            final[n] = out

    return final

# =========================
#  WRITE SUBMISSION
# =========================

def write_sub(final, filename="submission.csv"):
    with open(filename,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","x","y","deg"])
        for n in range(1, MAX_N+1):
            layout = final[n]
            for i,(x,y,deg) in enumerate(layout):
                w.writerow([f"{n:03d}_{i}",f"s{x}",f"s{y}",f"s{deg}"])

# =========================
#  MAIN
# =========================

def main():
    random.seed(GLOBAL_SEED)

    print("PHASE 1: sloppy pass...")
    sloppy = sloppy_pass()

    print("Selecting worst offenders...")
    worst = pick_worst(sloppy)

    print("PHASE 2: refining...")
    refined_sets = {}
    for n in worst:
        print(f"  refining n={n}")
        base,_,_ = sloppy[n]
        refined_sets[n] = fast_refine(n, base)

    print("PHASE 3: building final layouts...")
    final = build_final(sloppy, worst, refined_sets)

    write_sub(final)
    print("Done. Wrote submission.csv")

if __name__ == "__main__":
    main()
