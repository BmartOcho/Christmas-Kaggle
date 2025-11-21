import math
import random
import csv

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

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
    """Create shapely polygon for a tree at (x,y) rotated by deg degrees."""
    p = rotate(TREE_POLYGON, deg, origin=(0, 0))
    p = translate(p, xoff=x, yoff=y)
    return p


# =========================
#  BASIC GEOMETRY HELPERS
# =========================

def bounding_side(polys):
    """Side of minimal bounding square that encloses all polygons."""
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)


def has_overlap(poly, polys, skip_idx):
    """Check overlap with touching allowed."""
    for j, other in enumerate(polys):
        if j == skip_idx:
            continue
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False


def layout_to_polys(layout):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]


# =========================
#  INITIAL LAYOUT (HEURISTIC)
# =========================

def nested_row_layout(n, spacing=0.75):
    """
    Basic heuristic:
    - grid with rows/cols
    - checkerboard 0/180 degrees
    - horizontal staggering for odd rows
    """
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

def collision_cleanup(layout, iterations=60, push_amount=0.02):
    """
    Deterministic-ish pass to resolve any remaining overlaps by slightly
    pushing pairs apart along their connecting line.
    """
    layout = layout[:]  # copy
    polys = layout_to_polys(layout)

    for _ in range(iterations):
        moved = False
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
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
                  steps=6000,
                  initial_step=0.02, final_step=0.003,
                  initial_rot=5.0, final_rot=1.0,
                  initial_T=0.02, final_T=0.0005):
    """
    Simulated annealing:
    - random tree perturbation per step
    - reject moves that cause overlap
    - accept worse moves with probability based on temperature
    """

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

        i = random.randint(0, len(layout) - 1)
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
            # accept
            layout[i] = (nx, ny, ndeg)
            curr_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_layout = layout[:]
        else:
            # revert
            polys[i] = old_poly

    return best_layout, best_score


# =========================
#  SHRINK-AND-PACK LOOP
# =========================

def shrink_and_pack(initial_layout,
                    rounds=25,
                    shrink_factor=0.985,
                    anneal_steps=2500):
    """
    Iteratively:
    - shrink positions toward center
    - cleanup collisions
    - run annealing
    Track best layout by bounding side.
    """

    layout = initial_layout[:]
    polys = layout_to_polys(layout)
    best_layout = layout[:]
    best_score = bounding_side(polys)
    no_improve_rounds = 0

    for r in range(rounds):
        # shrink around center
        xs = [x for x, _, _ in layout]
        ys = [y for _, y, _ in layout]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2

        shrunk = []
        for (x, y, deg) in layout:
            sx = cx + (x - cx) * shrink_factor
            sy = cy + (y - cy) * shrink_factor
            shrunk.append((sx, sy, deg))

        # Cleanup overlaps caused by shrink
        shrunk = collision_cleanup(shrunk, iterations=40, push_amount=0.01)

        # Anneal locally
        shrunk, score = anneal_layout(
            shrunk,
            steps=anneal_steps,
            initial_step=0.015,
            final_step=0.003,
            initial_rot=4.0,
            final_rot=1.0,
            initial_T=0.015,
            final_T=0.0005,
        )

        if score < best_score - 1e-4:
            best_score = score
            best_layout = shrunk[:]
            layout = shrunk
            no_improve_rounds = 0
        else:
            layout = shrunk
            no_improve_rounds += 1
            if no_improve_rounds >= 6:
                # Plateau, bail out
                break

    # final cleanup
    best_layout = collision_cleanup(best_layout, iterations=80, push_amount=0.01)
    return best_layout, best_score


# =========================
#  KEY SIZES + MULTI-START
# =========================

KEY_NS = [10, 20, 40, 80, 120, 160, 200]


def nearest_key(n):
    for k in KEY_NS:
        if n <= k:
            return k
    return KEY_NS[-1]


def build_key_layouts():
    """
    For each key N:
    - multi-start: try several different initial layouts
    - shrink-and-pack each
    - keep best
    """
    key_layouts = {}
    for k in KEY_NS:
        print(f"[KEY {k}] building compact layout...")

        best_layout = None
        best_score = float("inf")

        # multi-start
        starts = 4 if k < 100 else 3
        for s in range(starts):
            print(f"  start {s+1}/{starts}")
            spacing = 0.72 + 0.08 * random.random()
            base = nested_row_layout(k, spacing=spacing)

            # initial gentle cleanup
            base = collision_cleanup(base, iterations=40, push_amount=0.015)

            # shrink & pack
            rounds = 22 if k < 100 else 18
            anneal_steps = 2200 if k < 100 else 2600

            packed, score = shrink_and_pack(
                base,
                rounds=rounds,
                shrink_factor=0.986,
                anneal_steps=anneal_steps,
            )

            print(f"    -> side ≈ {score:.4f}")

            if score < best_score:
                best_score = score
                best_layout = packed[:]

        # Final cleanup, just in case
        best_layout = collision_cleanup(best_layout, iterations=80, push_amount=0.01)
        print(f"[KEY {k}] best side ≈ {best_score:.4f}")
        key_layouts[k] = best_layout

    return key_layouts


# =========================
#  DERIVED N FROM KEYS
# =========================

def make_layout_for_n(n, key_layouts):
    if n in KEY_NS:
        return key_layouts[n]

    k = nearest_key(n)
    base = key_layouts[k][:n]

    # Slight expansion to reduce inherited tightness
    base = [(x * 1.015, y * 1.015, deg) for (x, y, deg) in base]

    # Shorter pack cycle
    rounds = 12 if n <= 80 else 16
    steps = 1600 if n <= 80 else 2000

    packed, _ = shrink_and_pack(
        base,
        rounds=rounds,
        shrink_factor=0.988,
        anneal_steps=steps,
    )

    packed = collision_cleanup(packed, iterations=60, push_amount=0.01)
    return packed


# =========================
#  SUBMISSION GENERATION
# =========================

def write_submission(filename="submission.csv"):
    print("Building key layouts (this is the heavy part)...")
    key_layouts = build_key_layouts()
    print("Key layouts built.\nGenerating all n=1..200...")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])

        for n in range(1, 201):
            print(f"[n={n}] making layout...")
            layout = make_layout_for_n(n, key_layouts)

            for idx, (x, y, deg) in enumerate(layout):
                writer.writerow([
                    f"{n:03d}_{idx}",
                    f"s{x}",
                    f"s{y}",
                    f"s{deg}"
                ])

    print(f"Done. Wrote {filename}")


if __name__ == "__main__":
    random.seed(42)
    write_submission()
