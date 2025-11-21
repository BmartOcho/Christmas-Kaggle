import math
import random
import csv

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#  TREE SHAPE (exact match)
# =========================

TREE_POLYGON = Polygon([
    (0.0, 0.8),            # tip
    (0.25/2, 0.5),         # right upper branch
    (0.25/4, 0.5),
    (0.4/2, 0.25),         # right mid
    (0.4/4, 0.25),
    (0.7/2, 0.0),          # right bottom
    (0.15/2, 0.0),         # trunk top right
    (0.15/2, -0.2),        # trunk bottom right
    (-0.15/2, -0.2),       # trunk bottom left
    (-0.15/2, 0.0),        # trunk top left
    (-0.7/2, 0.0),         # left bottom
    (-0.4/4, 0.25),        # left mid
    (-0.4/2, 0.25),
    (-0.25/4, 0.5),        # left upper branch
    (-0.25/2, 0.5)
])


def place_tree(x: float, y: float, deg: float):
    """Return Shapely polygon for a tree at (x, y) rotated by deg degrees."""
    p = rotate(TREE_POLYGON, deg, origin=(0, 0))
    p = translate(p, xoff=x, yoff=y)
    return p


# =========================
#  OPTIONAL: VISUALIZATION
# =========================

def plot_layout(layout, title="layout"):
    """Quick debug plot. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    for (x, y, deg) in layout:
        poly = place_tree(x, y, deg)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.6)

    ax.set_aspect("equal", "box")
    plt.title(title)
    plt.show()


# =========================
#  BASE LAYOUT (NESTED ROW)
# =========================

def nested_row_layout(n: int, spacing: float = 0.8):
    """
    Initial heuristic layout:
    - Rough grid
    - Checkerboard 0° / 180° rotation
    - Every other row horizontally offset to encourage "branch pocket" nesting
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    layout = []

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                break

            # Checkerboard flipping
            deg = 0.0 if (r + c) % 2 == 0 else 180.0

            x = c * spacing
            y = r * spacing

            # Horizontal stagger for odd rows
            if r % 2 == 1:
                x += spacing * 0.4

            layout.append((x, y, deg))
            i += 1

    # Center the whole thing around (0, 0)
    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0

    layout = [(x - cx, y - cy, deg) for (x, y, deg) in layout]
    return layout


# =========================
#  SCORING / OVERLAP
# =========================

def bounding_side(polys):
    """Return side length of the minimal bounding square for polygons."""
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)


def has_overlap(poly, polys, idx_to_skip):
    """Check if `poly` overlaps any other in `polys` (touching is allowed)."""
    for j, other in enumerate(polys):
        if j == idx_to_skip:
            continue
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False


# =========================
#  SIMULATED ANNEALING
# =========================

def anneal_layout(initial_layout, steps=8000,
                  initial_step=0.05, final_step=0.005,
                  initial_rot=10.0, final_rot=2.0,
                  initial_T=0.05, final_T=0.001):
    """
    Annealing-style local search:
    - Randomly perturb one tree at a time
    - Reject moves that create overlaps
    - Use temperature schedule to sometimes accept worse moves
    - Track best layout by bounding square side length
    """
    layout = initial_layout[:]  # copy
    polys = [place_tree(x, y, deg) for (x, y, deg) in layout]

    curr_score = bounding_side(polys)
    best_score = curr_score
    best_layout = layout[:]

    for step in range(steps):
        t = step / steps

        # Linearly decay step sizes & temperature
        step_size = initial_step * (1 - t) + final_step * t
        rot_size = initial_rot * (1 - t) + final_rot * t
        T = initial_T * (1 - t) + final_T * t

        i = random.randint(0, len(layout) - 1)
        x, y, deg = layout[i]

        # Propose move
        nx = x + random.uniform(-step_size, step_size)
        ny = y + random.uniform(-step_size, step_size)
        ndeg = (deg + random.uniform(-rot_size, rot_size)) % 360.0

        new_poly = place_tree(nx, ny, ndeg)

        # Reject if overlaps any other tree
        if has_overlap(new_poly, polys, i):
            continue

        # Compute new score
        old_bounds = polys[i].bounds
        polys[i] = new_poly
        new_score = bounding_side(polys)

        delta = new_score - curr_score

        # Accept if better or with annealing probability
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            # accept
            curr_score = new_score
            layout[i] = (nx, ny, ndeg)
            if new_score < best_score:
                best_score = new_score
                best_layout = layout[:]
        else:
            # revert polygon if rejected
            polys[i] = place_tree(x, y, deg)

    return best_layout, best_score


# =========================
#  KEY N STRATEGY
# =========================

KEY_NS = [10, 20, 40, 80, 120, 160, 200]

def nearest_key_n(n: int):
    for k in KEY_NS:
        if n <= k:
            return k
    return KEY_NS[-1]


def build_key_layouts():
    """
    Compute deep layouts for a few key sizes only.
    This is where we spend most of the optimization effort.
    """
    key_layouts = {}
    for k in KEY_NS:
        print(f"[KEY] Building deep layout for n = {k}")
        base = nested_row_layout(k, spacing=0.8)

        # More steps for larger N
        if k <= 20:
            steps = 12000
        elif k <= 80:
            steps = 16000
        else:
            steps = 20000

        layout, score = anneal_layout(base, steps=steps)
        print(f"[KEY] n={k}, best_side≈{score:.4f}")
        key_layouts[k] = layout

    return key_layouts


def make_layout_for_n(n: int, key_layouts):
    """
    For a given n:
    - Find nearest larger key size
    - Take first n trees from that key layout
    - Run a short local anneal to adapt
    """
    if n in KEY_NS:
        # Use key layout directly
        return key_layouts[n]

    k = nearest_key_n(n)
    base_layout = key_layouts[k][:n]

    # Short refinement anneal
    steps = 2000 if n <= 80 else 4000
    layout, _ = anneal_layout(
        base_layout,
        steps=steps,
        initial_step=0.03,
        final_step=0.005,
        initial_rot=6.0,
        final_rot=2.0,
        initial_T=0.03,
        final_T=0.001,
    )
    return layout


# =========================
#  SUBMISSION GENERATION
# =========================

def write_submission(filename="submission.csv"):
    """
    Generate Kaggle submission for n = 1..200
    using key-sized deep layouts + derived refinements.
    """
    print("Building key layouts...")
    key_layouts = build_key_layouts()
    print("Key layouts ready.\n")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])

        for n in range(1, 201):
            print(f"[n={n}] Generating layout...")
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
    # If you want to inspect:
    # layout_40 = make_layout_for_n(40, build_key_layouts())
    # plot_layout(layout_40, title="n=40")
