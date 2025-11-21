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
FAST_DEEP_LIMIT = 15   # how many N to refine more carefully
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

def place_tree(x: float, y: float, deg: float):
    p = rotate(TREE_POLYGON, deg, origin=(0, 0))
    return translate(p, xoff=x, yoff=y)

def layout_to_polys(layout: List[Tuple[float, float, float]]):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]

def bounding_side(polys) -> float:
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)

def boxes_overlap(b1, b2) -> bool:
    # b = (minx, miny, maxx, maxy)
    return not (b1[2] < b2[0] or
                b2[2] < b1[0] or
                b1[3] < b2[1] or
                b2[3] < b1[1])

def has_overlap(poly, polys, skip: int) -> bool:
    b1 = poly.bounds
    for j, other in enumerate(polys):
        if j == skip:
            continue
        b2 = other.bounds
        if not boxes_overlap(b1, b2):
            continue
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False

def group_contrib(side: float, n: int) -> float:
    return (side * side) / n

# =========================
#  GRID NESTED LAYOUT
# =========================

def nested_layout(n: int, spacing: float = 0.82) -> List[Tuple[float, float, float]]:
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    layout: List[Tuple[float, float, float]] = []
    idx = 0

    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            deg = 0.0 if (r + c) % 2 == 0 else 180.0
            x = c * spacing
            y = r * spacing
            if r % 2 == 1:
                x += spacing * 0.45
            layout.append((x, y, deg))
            idx += 1

    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs)+min(xs))/2.0
    cy = (max(ys)+min(ys))/2.0
    return [(x-cx, y-cy, deg) for (x,y,deg) in layout]

# =========================
#  OCTAGON-RING LAYOUT
# =========================

def octagon_ring_layout(n: int, base_spacing: float = 0.70) -> List[Tuple[float, float, float]]:
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    positions = [(0.0, 0.0)]
    ring = 1

    while len(positions) < n:
        count = 8 * ring
        radius = base_spacing * (0.85 + 0.15 * ring)
        angle_offset = 45.0 if (ring % 2 == 1) else 0.0
        step = 360.0 / count

        for i in range(count):
            ang = math.radians(angle_offset + i * step)
            positions.append((radius * math.cos(ang), radius * math.sin(ang)))

        ring += 1

    positions = positions[:n]
    layout: List[Tuple[float, float, float]] = []
    for i, (x, y) in enumerate(positions):
        deg = 0.0 if (i % 2 == 0) else 180.0
        x += random.uniform(-0.002, 0.002)
        y += random.uniform(-0.002, 0.002)
        layout.append((x, y, deg))

    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs)+min(xs))/2.0
    cy = (max(ys)+min(ys))/2.0
    return [(x-cx, y-cy, deg) for (x,y,deg) in layout]

# =========================
#  SPIRAL (PHYLLOTAXIS) LAYOUT
# =========================

def spiral_layout(n: int, base_spacing: float = 0.55) -> List[Tuple[float, float, float]]:
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    golden_angle = math.radians(137.50776405003785)
    layout: List[Tuple[float, float, float]] = []

    for k in range(n):
        r = base_spacing * math.sqrt(k)
        theta = k * golden_angle
        x = r * math.cos(theta) + random.uniform(-0.002, 0.002)
        y = r * math.sin(theta) + random.uniform(-0.002, 0.002)
        deg = 0.0 if (k % 2 == 0) else 180.0
        layout.append((x, y, deg))

    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs)+min(xs))/2.0
    cy = (max(ys)+min(ys))/2.0
    return [(x-cx, y-cy, deg) for (x,y,deg) in layout]

# =========================
#  MINI COMPACTOR
# =========================

def mini_compact(layout: List[Tuple[float, float, float]],
                 scales=(0.985, 0.97)) -> List[Tuple[float, float, float]]:
    curr = layout[:]
    for s in scales:
        scaled = [(x * s, y * s, deg) for (x, y, deg) in curr]
        scaled = cleanup(scaled, iters=15, push=0.012)
        curr = scaled
    return curr

# =========================
#  CLEANUP (with bounds pre-check)
# =========================

def cleanup(layout: List[Tuple[float, float, float]],
            iters: int = 30,
            push: float = 0.015) -> List[Tuple[float, float, float]]:
    layout = layout[:]
    polys = layout_to_polys(layout)

    for _ in range(iters):
        moved = False
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                p1 = polys[i]
                p2 = polys[j]
                b1 = p1.bounds
                b2 = p2.bounds
                if not boxes_overlap(b1, b2):
                    continue
                if p1.intersects(p2) and not p1.touches(p2):
                    x1, y1, d1 = layout[i]
                    x2, y2, d2 = layout[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = math.hypot(dx, dy) or 1.0
                    push_amt = push / dist
                    x1 += dx * push_amt
                    y1 += dy * push_amt
                    x2 -= dx * push_amt
                    y2 -= dy * push_amt
                    layout[i] = (x1, y1, d1)
                    layout[j] = (x2, y2, d2)
                    polys[i] = place_tree(x1, y1, d1)
                    polys[j] = place_tree(x2, y2, d2)
                    moved = True
        if not moved:
            break

    return layout

# =========================
#  SAFETY: ENSURE NO OVERLAPS
# =========================

def ensure_no_overlap(layout: List[Tuple[float, float, float]],
                      safe: bool = False,
                      max_tries: int = 3) -> List[Tuple[float, float, float]]:
    curr = layout[:]
    for _ in range(max_tries):
        polys = layout_to_polys(curr)
        n = len(curr)
        overlapped = False
        for i in range(n):
            for j in range(i + 1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    overlapped = True
                    break
            if overlapped:
                break
        if not overlapped:
            return curr

        # If overlaps remain, expand a bit and re-clean
        scale = 1.08 if safe else 1.02
        curr = [(x * scale, y * scale, deg) for (x, y, deg) in curr]
        curr = cleanup(curr,
                       iters=60 if safe else 40,
                       push=0.02 if safe else 0.018)

    return curr  # final attempt, should be very safe

# =========================
#  LIGHT ANNEALER
# =========================

def light_anneal(layout: List[Tuple[float, float, float]],
                 steps: int = 800):
    layout = layout[:]
    polys = layout_to_polys(layout)
    curr = bounding_side(polys)
    best_layout = layout[:]
    best = curr

    for _ in range(steps):
        i = random.randint(0, len(layout) - 1)
        x, y, d = layout[i]
        nx = x + random.uniform(-0.01, 0.01)
        ny = y + random.uniform(-0.01, 0.01)
        nd = (d + random.uniform(-2.0, 2.0)) % 360.0
        npoly = place_tree(nx, ny, nd)

        if has_overlap(npoly, polys, i):
            continue

        old = polys[i]
        polys[i] = npoly
        side = bounding_side(polys)

        if side <= curr:
            layout[i] = (nx, ny, nd)
            curr = side
            if side < best:
                best = side
                best_layout = layout[:]
        else:
            polys[i] = old

    return best_layout, best

# =========================
#  PHASE 1: ENSEMBLE SLOPPY PASS (HYBRID)
# =========================

def sloppy_pass():
    result: Dict[int, Tuple[List[Tuple[float, float, float]], float, float]] = {}
    for n in range(1, MAX_N + 1):
        # HYBRID: safe small-N mode
        if n <= 20:
            lay = spiral_layout(n, base_spacing=0.60)
            lay = cleanup(lay, iters=40, push=0.015)
            lay = ensure_no_overlap(lay, safe=True)
        else:
            # Aggressive ensemble for larger N
            lay1 = nested_layout(n)
            side1 = bounding_side(layout_to_polys(lay1))

            lay2 = octagon_ring_layout(n)
            side2 = bounding_side(layout_to_polys(lay2))

            lay3 = spiral_layout(n, base_spacing=0.55)
            side3 = bounding_side(layout_to_polys(lay3))

            best_side, best_lay = side1, lay1
            if side2 < best_side:
                best_side, best_lay = side2, lay2
            if side3 < best_side:
                best_side, best_lay = side3, lay3

            lay = mini_compact(best_lay, scales=(0.985, 0.97))
            lay = cleanup(lay, iters=20, push=0.012)
            lay = ensure_no_overlap(lay, safe=False)

        polys = layout_to_polys(lay)
        side = bounding_side(polys)
        contrib = group_contrib(side, n)
        result[n] = (lay, side, contrib)

        if n % 50 == 0:
            print(f"[SLOPPY] n={n}, side≈{side:.4f}, contrib≈{contrib:.6f}")

    return result

# =========================
#  SELECT WORST (SMALL-N BIAS)
# =========================

def pick_worst(sloppy, k: int = FAST_DEEP_LIMIT):
    items = []
    for n in sloppy:
        _, _, contrib = sloppy[n]
        weight = contrib * (2.0 if n <= 20 else 1.2 if n <= 80 else 1.0)
        items.append((n, weight))
    items.sort(key=lambda x: x[1], reverse=True)
    top = [n for n, _ in items[:k]]
    top.sort()
    print(f"[SELECT] Worst Ns: {top}")
    return top

# =========================
#  PHASE 2: FAST REFINEMENT
# =========================

def fast_refine(n: int, base_layout: List[Tuple[float, float, float]]):
    if n <= 20:
        steps = 1600
        safe_flag = True
    elif n <= 80:
        steps = 1000
        safe_flag = False
    else:
        steps = 700
        safe_flag = False

    layout = cleanup(base_layout, iters=25, push=0.02)
    layout, _ = light_anneal(layout, steps=steps)
    layout = cleanup(layout, iters=25, push=0.015)
    layout = ensure_no_overlap(layout, safe=safe_flag)
    return layout

# =========================
#  PHASE 3: BUILD FINAL LAYOUTS
# =========================

def build_final(sloppy, refined_ns, refined_sets):
    final: Dict[int, List[Tuple[float, float, float]]] = {}
    deep_ns = sorted(refined_ns)

    def nearest_refined(n: int) -> int:
        if not deep_ns:
            return -1
        candidates = [dn for dn in deep_ns if dn >= n]
        if candidates:
            return min(candidates)
        return deep_ns[-1]

    for n in range(1, MAX_N + 1):
        if n in refined_sets:
            layout = refined_sets[n]
        else:
            parent = nearest_refined(n)
            if parent == -1:
                layout = sloppy[n][0]
            else:
                parent_layout = refined_sets[parent]
                if len(parent_layout) < n:
                    # For small N, use safe spiral; for larger N, normal spiral
                    if n <= 20:
                        layout = spiral_layout(n, base_spacing=0.60)
                        layout = cleanup(layout, iters=40, push=0.015)
                        layout = ensure_no_overlap(layout, safe=True)
                    else:
                        print(f"[WARN] parent={parent} has {len(parent_layout)} < n={n}, using fresh spiral")
                        layout = spiral_layout(n, base_spacing=0.55)
                        layout = cleanup(layout, iters=20, push=0.012)
                        layout = ensure_no_overlap(layout, safe=False)
                else:
                    layout = parent_layout[:n]
                    layout = [(x * 1.01, y * 1.01, deg) for (x, y, deg) in layout]
                    layout = cleanup(layout, iters=15, push=0.01)
                    refine_steps = 600 if n <= 50 else 400
                    layout, _ = light_anneal(layout, steps=refine_steps)
                    layout = cleanup(layout, iters=15, push=0.01)
                    layout = ensure_no_overlap(layout, safe=(n <= 20))

        final[n] = layout

    return final

# =========================
#  VALIDATION
# =========================

def validate_final(final: Dict[int, List[Tuple[float, float, float]]]):
    for n in range(1, MAX_N + 1):
        if n not in final:
            print(f"[FIX] Missing layout for n={n}, regenerating.")
            if n <= 20:
                lay = spiral_layout(n, base_spacing=0.60)
                lay = cleanup(lay, iters=40, push=0.015)
                lay = ensure_no_overlap(lay, safe=True)
            else:
                lay = spiral_layout(n, base_spacing=0.55)
                lay = cleanup(lay, iters=25, push=0.02)
                lay = ensure_no_overlap(lay, safe=False)
            final[n] = lay
        else:
            layout = final[n]
            if len(layout) != n:
                print(f"[FIX] Layout for n={n} has {len(layout)} trees, expected {n}. Regenerating.")
                if n <= 20:
                    lay = spiral_layout(n, base_spacing=0.60)
                    lay = cleanup(lay, iters=40, push=0.015)
                    lay = ensure_no_overlap(lay, safe=True)
                else:
                    lay = spiral_layout(n, base_spacing=0.55)
                    lay = cleanup(lay, iters=25, push=0.02)
                    lay = ensure_no_overlap(lay, safe=False)
                final[n] = lay

# =========================
#  WRITE SUBMISSION
# =========================

def write_submission(final: Dict[int, List[Tuple[float, float, float]]],
                     filename: str = "submission.csv"):
    total_rows = 0
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, MAX_N + 1):
            layout = final[n]
            if len(layout) != n:
                raise RuntimeError(f"Final layout for n={n} has {len(layout)} trees, expected {n}")
            for i, (x, y, deg) in enumerate(layout):
                w.writerow([f"{n:03d}_{i}", f"s{x}", f"s{y}", f"s{deg}"])
                total_rows += 1
    print(f"[WRITE] Wrote {total_rows} rows (expected 20100).")

# =========================
#  MAIN
# =========================

def main():
    random.seed(GLOBAL_SEED)

    print("=== PHASE 1: SLOPPY PASS (HYBRID) ===")
    sloppy = sloppy_pass()
    sloppy_total = sum(group_contrib(sloppy[n][1], n) for n in range(1, MAX_N + 1))
    print(f"[SLOPPY] Estimated score ≈ {sloppy_total:.6f}")

    print("=== SELECT WORST ===")
    worst = pick_worst(sloppy, k=FAST_DEEP_LIMIT)

    print("=== PHASE 2: FAST REFINEMENT ===")
    refined_sets: Dict[int, List[Tuple[float, float, float]]] = {}
    for n in worst:
        base, _, _ = sloppy[n]
        print(f"  refining n={n}")
        refined_sets[n] = fast_refine(n, base)

    print("=== PHASE 3: BUILD FINAL ===")
    final = build_final(sloppy, worst, refined_sets)

    print("=== VALIDATE ===")
    validate_final(final)

    print("=== WRITE SUBMISSION ===")
    write_submission(final)

    print("Done. submission.csv is ready.")

if __name__ == "__main__":
    main()
