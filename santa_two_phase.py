import math
import random
import csv
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#  CONFIG
# =========================

MAX_N = 200

# How many Ns to send to deep optimization (phase 2)
MAX_DEEP_NS = 25

# Whether to use multiprocessing for deep optimization
USE_PARALLEL = True
N_PROCESSES = max(1, mp.cpu_count() - 1)

# Random seed for reproducibility
GLOBAL_SEED = 42

# =========================
#  TREE SHAPE (exact match)
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
    p = translate(p, xoff=x, yoff=y)
    return p


# =========================
#  HELPERS: GEOMETRY & SCORE
# =========================

def layout_to_polys(layout):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]


def bounding_side(polys):
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)


def has_overlap(poly, polys, skip_idx):
    for j, other in enumerate(polys):
        if j == skip_idx:
            continue
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False


def group_metric_contrib(side: float, n: int) -> float:
    """Kaggle metric contribution for one group: side^2 / n."""
    return (side * side) / n


# =========================
#  VISUALIZATION
# =========================

def plot_layout(layout, title="layout"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    polys = layout_to_polys(layout)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for p in polys:
        xs, ys = p.exterior.xy
        ax.plot(xs, ys)
    ax.set_aspect("equal", "box")
    plt.title(title)
    plt.show()


def plot_score_comparison(sloppy_scores: Dict[int, float],
                          final_scores: Dict[int, float]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    ns = sorted(sloppy_scores.keys())
    sloppy = [sloppy_scores[n] for n in ns]
    final = [final_scores.get(n, sloppy_scores[n]) for n in ns]

    plt.figure(figsize=(10, 4))
    plt.plot(ns, sloppy, label="sloppy side", alpha=0.7)
    plt.plot(ns, final, label="final side", alpha=0.7)
    plt.xlabel("n")
    plt.ylabel("bounding side")
    plt.legend()
    plt.title("Sloppy vs final bounding sides")
    plt.grid(True)
    plt.show()


# =========================
#  NESTED INITIAL LAYOUT
# =========================

def nested_row_layout(n, spacing=0.8):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    layout = []

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                break
            deg = 0.0 if (r + c) % 2 == 0 else 180.0
            x = c * spacing
            y = r * spacing
            if r % 2 == 1:
                x += spacing * 0.4
            layout.append((x, y, deg))
            i += 1

    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    return [(x - cx, y - cy, deg) for (x, y, deg) in layout]


# =========================
#  COLLISION CLEANUP
# =========================

def collision_cleanup(layout, iterations=50, push_amount=0.02):
    layout = layout[:]  # copy
    polys = layout_to_polys(layout)

    for _ in range(iterations):
        moved = False
        for i in range(len(layout)):
            for j in range(i+1, len(layout)):
                p1 = polys[i]
                p2 = polys[j]
                if p1.intersects(p2) and not p1.touches(p2):
                    x1, y1, d1 = layout[i]
                    x2, y2, d2 = layout[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = math.hypot(dx, dy) or 1.0
                    push = push_amount / dist
                    x1 += dx * push
                    y1 += dy * push
                    x2 -= dx * push
                    y2 -= dy * push
                    layout[i] = (x1, y1, d1)
                    layout[j] = (x2, y2, d2)
                    polys[i] = place_tree(x1, y1, d1)
                    polys[j] = place_tree(x2, y2, d2)
                    moved = True
        if not moved:
            break

    return layout


# =========================
#  ANNEALING LOCAL SEARCH
# =========================

def anneal_layout(initial_layout,
                  steps=4000,
                  initial_step=0.015, final_step=0.003,
                  initial_rot=4.0, final_rot=1.0,
                  initial_T=0.015, final_T=0.0005):

    layout = initial_layout[:]
    polys = layout_to_polys(layout)

    curr_score = bounding_side(polys)
    best_score = curr_score
    best_layout = layout[:]

    for step in range(steps):
        t = step / steps
        step_size = initial_step * (1 - t) + final_step * t
        rot_size = initial_rot * (1 - t) + final_rot * t
        T = initial_T * (1 - t) + final_T * t

        i = random.randint(0, len(layout)-1)
        x, y, deg = layout[i]

        nx = x + random.uniform(-step_size, step_size)
        ny = y + random.uniform(-step_size, step_size)
        ndeg = (deg + random.uniform(-rot_size, rot_size)) % 360.0

        new_poly = place_tree(nx, ny, ndeg)
        if has_overlap(new_poly, polys, i):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bounding_side(polys)
        delta = new_score - curr_score

        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            layout[i] = (nx, ny, ndeg)
            curr_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_layout = layout[:]
        else:
            polys[i] = old_poly

    return best_layout, best_score


# =========================
#  SHRINK-AND-PACK
# =========================

def shrink_and_pack(initial_layout,
                    rounds=20,
                    shrink_factor=0.985,
                    anneal_steps=2500):

    layout = initial_layout[:]
    polys = layout_to_polys(layout)
    best_layout = layout[:]
    best_score = bounding_side(polys)
    no_improve = 0

    for r in range(rounds):
        xs = [x for x, _, _ in layout]
        ys = [y for _, y, _ in layout]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2

        shrunk = []
        for (x, y, deg) in layout:
            sx = cx + (x - cx) * shrink_factor
            sy = cy + (y - cy) * shrink_factor
            shrunk.append((sx, sy, deg))

        shrunk = collision_cleanup(shrunk, iterations=40, push_amount=0.01)
        shrunk, score = anneal_layout(
            shrunk,
            steps=anneal_steps,
            initial_step=0.012,
            final_step=0.003,
            initial_rot=4.0,
            final_rot=1.0,
            initial_T=0.012,
            final_T=0.0004
        )

        if score < best_score - 1e-4:
            best_score = score
            best_layout = shrunk[:]
            layout = shrunk
            no_improve = 0
        else:
            layout = shrunk
            no_improve += 1
            if no_improve >= 6:
                break

    best_layout = collision_cleanup(best_layout, iterations=60, push_amount=0.01)
    return best_layout, best_score


# =========================
#  PHASE 1: FAST SLOPPY PASS
# =========================

@dataclass
class LayoutInfo:
    layout: List[Tuple[float, float, float]]
    side: float
    contrib: float


def fast_pass(max_n=MAX_N) -> Dict[int, LayoutInfo]:
    """
    Quick & dirty layouts for all n = 1..max_n:
    - nested heuristic only
    - small cleanup
    """
    result: Dict[int, LayoutInfo] = {}
    for n in range(1, max_n+1):
        base = nested_row_layout(n, spacing=0.85)
        base = collision_cleanup(base, iterations=20, push_amount=0.02)
        polys = layout_to_polys(base)
        side = bounding_side(polys)
        contrib = group_metric_contrib(side, n)
        result[n] = LayoutInfo(layout=base, side=side, contrib=contrib)
        if n % 20 == 0:
            print(f"[FAST] n={n}, side≈{side:.4f}, contrib≈{contrib:.6f}")
    return result


def write_sloppy_scores(sloppy: Dict[int, LayoutInfo],
                        filename="sloppy_scores.csv"):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "side", "contrib"])
        for n in sorted(sloppy.keys()):
            info = sloppy[n]
            w.writerow([n, info.side, info.contrib])


# =========================
#  SELECT NS FOR DEEP PASS
# =========================

def select_ns_for_deep(sloppy: Dict[int, LayoutInfo],
                       max_deep=MAX_DEEP_NS) -> List[int]:
    """
    Choose Ns with the largest metric contribution from sloppy pass.
    """
    items = [(n, info.contrib) for n, info in sloppy.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    chosen = [n for (n, _) in items[:max_deep]]

    # Make sure large Ns are represented
    large_candidates = [n for n in range(MAX_N-20, MAX_N+1)]
    for n in large_candidates:
        if n not in chosen:
            chosen.append(n)
    chosen = sorted(set(chosen))
    print(f"[SELECT] Deep optimize Ns: {chosen}")
    return chosen


# =========================
#  PHASE 2: DEEP PASS
# =========================

def deep_optimize_single(args):
    """
    Worker function for deep optimization of a single n.
    Used for multiprocessing or sequential.
    """
    n, base_layout, seed = args
    random.seed(seed)
    print(f"[DEEP] n={n} starting...")

    # Multi-start inside: try a couple of jittered bases
    n_starts = 3 if n < 100 else 2
    best_layout = None
    best_side = float("inf")

    for s in range(n_starts):
        jittered = []
        for (x, y, deg) in base_layout:
            # small random jiggle to escape symmetry
            jx = x + random.uniform(-0.05, 0.05)
            jy = y + random.uniform(-0.05, 0.05)
            jdeg = (deg + random.uniform(-10, 10)) % 360
            jittered.append((jx, jy, jdeg))

        jittered = collision_cleanup(jittered, iterations=40, push_amount=0.02)

        rounds = 18 if n < 100 else 14
        steps = 2200 if n < 100 else 2600

        packed, side = shrink_and_pack(
            jittered,
            rounds=rounds,
            shrink_factor=0.986,
            anneal_steps=steps,
        )
        if side < best_side:
            best_side = side
            best_layout = packed[:]

    print(f"[DEEP] n={n} finished, best side≈{best_side:.4f}")
    return n, best_layout, best_side


def deep_pass(sloppy: Dict[int, LayoutInfo],
              ns_to_optimize: List[int]) -> Dict[int, LayoutInfo]:
    """
    Deep optimize selected Ns using shrink-and-pack, parallel or sequential.
    """
    tasks = []
    for i, n in enumerate(ns_to_optimize):
        base_layout = sloppy[n].layout
        tasks.append((n, base_layout, GLOBAL_SEED + i*17))

    results: Dict[int, LayoutInfo] = {}

    if USE_PARALLEL and len(tasks) > 1:
        print(f"[DEEP] Running in parallel with {N_PROCESSES} processes")
        with mp.Pool(N_PROCESSES) as pool:
            for n, layout, side in pool.map(deep_optimize_single, tasks):
                contrib = group_metric_contrib(side, n)
                results[n] = LayoutInfo(layout=layout, side=side, contrib=contrib)
    else:
        print("[DEEP] Running sequentially")
        for t in tasks:
            n, layout, side = deep_optimize_single(t)
            contrib = group_metric_contrib(side, n)
            results[n] = LayoutInfo(layout=layout, side=side, contrib=contrib)

    return results


# =========================
#  INHERIT & FINAL LAYOUTS
# =========================

def nearest_deep_n(n: int, deep_ns: List[int]) -> int:
    # choose the smallest deep_n >= n, else the largest one
    candidates = [dn for dn in deep_ns if dn >= n]
    if candidates:
        return min(candidates)
    return max(deep_ns)


def build_final_layouts(sloppy: Dict[int, LayoutInfo],
                        deep: Dict[int, LayoutInfo]) -> Dict[int, LayoutInfo]:
    """
    For each n:
      - use deep layout if n in deep
      - else inherit from nearest deep-n and lightly refine
    """
    deep_ns = sorted(deep.keys())
    final: Dict[int, LayoutInfo] = {}

    for n in range(1, MAX_N+1):
        if n in deep:
            final[n] = deep[n]
        else:
            parent_n = nearest_deep_n(n, deep_ns)
            parent_layout = deep[parent_n].layout

            base = parent_layout[:n]
            # slight expansion to reduce collisions
            base = [(x * 1.01, y * 1.01, deg) for (x, y, deg) in base]

            # gentle local refine
            steps = 1500 if n <= 80 else 2000
            refined, side = anneal_layout(
                base,
                steps=steps,
                initial_step=0.01,
                final_step=0.002,
                initial_rot=3.0,
                final_rot=1.0,
                initial_T=0.01,
                final_T=0.0004,
            )
            refined = collision_cleanup(refined, iterations=40, push_amount=0.01)
            polys = layout_to_polys(refined)
            side = bounding_side(polys)
            contrib = group_metric_contrib(side, n)
            final[n] = LayoutInfo(layout=refined, side=side, contrib=contrib)

    return final


def write_final_scores(final: Dict[int, LayoutInfo],
                       filename="final_scores.csv"):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "side", "contrib"])
        for n in sorted(final.keys()):
            info = final[n]
            w.writerow([n, info.side, info.contrib])


# =========================
#  SUBMISSION
# =========================

def write_submission(final: Dict[int, LayoutInfo],
                     filename="submission.csv"):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, MAX_N+1):
            layout = final[n].layout
            for idx, (x, y, deg) in enumerate(layout):
                w.writerow([
                    f"{n:03d}_{idx}",
                    f"s{x}",
                    f"s{y}",
                    f"s{deg}"
                ])


# =========================
#  MAIN
# =========================

def main():
    random.seed(GLOBAL_SEED)

    print("=== PHASE 1: FAST SLOPPY PASS ===")
    sloppy = fast_pass(MAX_N)
    write_sloppy_scores(sloppy)
    sloppy_total = sum(info.contrib for info in sloppy.values())
    print(f"[FAST] Total sloppy metric ≈ {sloppy_total:.6f}")

    print("\n=== PHASE 2: SELECT NS FOR DEEP OPT ===")
    ns_deep = select_ns_for_deep(sloppy, max_deep=MAX_DEEP_NS)

    print("\n=== PHASE 3: DEEP OPTIMIZATION ===")
    deep = deep_pass(sloppy, ns_deep)
    deep_total = sum(info.contrib for info in deep.values())
    print(f"[DEEP] Sum contrib of deep-only Ns ≈ {deep_total:.6f}")

    print("\n=== PHASE 4: BUILD FINAL LAYOUTS ===")
    final = build_final_layouts(sloppy, deep)
    final_total = sum(info.contrib for info in final.values())
    print(f"[FINAL] Estimated total metric ≈ {final_total:.6f}")

    write_final_scores(final)
    write_submission(final)
    print("Wrote submission.csv, sloppy_scores.csv, final_scores.csv")

    # Optional: quick comparison plot
    # plot_score_comparison(
    #     {n: info.side for n, info in sloppy.items()},
    #     {n: info.side for n, info in final.items()},
    # )


if __name__ == "__main__":
    main()
