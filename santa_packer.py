import math
import random
import csv

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#  TREE SHAPE (exact match)
# =========================

# Based on the Kaggle metric (without the 1e18 scale factor)
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

def nested_row_layout(n: int, spacing: float = 1.0):
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

    # Center the whole thing around (0, 0) to keep bounding square smaller
    xs = [x for x, _, _ in layout]
    ys = [y for _, y, _ in layout]
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0

    layout = [(x - cx, y - cy, deg) for (x, y, deg) in layout]
    return layout


# =========================
#  SCORING (LOCAL ONLY)
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

def anneal_layout(initial_layout, steps=6000, step_size=0.03, rot_size=4.0):
    """
    Very simple local search:
    - randomly perturb one tree at a time
    - reject move if overlap appears
    - keep track of best layout by bounding square side
    No temperature schedule yet, just greedy improvement.
    """
    layout = initial_layout[:]  # shallow copy
    polys = [place_tree(x, y, deg) for (x, y, deg) in layout]

    best_layout = layout[:]
    best_polys = polys[:]
    best_score = bounding_side(polys)

    for step in range(steps):
        i = random.randint(0, len(layout) - 1)
        x, y, deg = layout[i]

        # propose move
        nx = x + random.uniform(-step_size, step_size)
        ny = y + random.uniform(-step_size, step_size)
        ndeg = (deg + random.uniform(-rot_size, rot_size)) % 360.0

        new_poly = place_tree(nx, ny, ndeg)

        # overlap check
        if has_overlap(new_poly, polys, i):
            continue  # reject move

        # accept move
        layout[i] = (nx, ny, ndeg)
        polys[i] = new_poly

        # evaluate
        new_score = bounding_side(polys)
        if new_score < best_score:
            best_score = new_score
            best_layout = layout[:]
            best_polys = polys[:]

    return best_layout, best_score


def make_layout_for_n(n: int):
    """
    Convenience function:
    - build heuristic layout
    - run annealing with more steps for bigger n
    """
    base_layout = nested_row_layout(n)

    # crude schedule: more trees → more steps
    if n <= 20:
        steps = 3000
    elif n <= 80:
        steps = 6000
    else:
        steps = 9000

    improved_layout, _ = anneal_layout(base_layout, steps=steps)
    return improved_layout


# =========================
#  SUBMISSION GENERATION
# =========================

def write_submission(filename="submission.csv"):
    """
    Generate Kaggle submission for n = 1..200.
    Uses our layout generator per n.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])

        for n in range(1, 201):
            print(f"Generating layout for n = {n}...")
            layout = make_layout_for_n(n)

            for idx, (x, y, deg) in enumerate(layout):
                # Kaggle expects the 's' prefix as string
                writer.writerow([
                    f"{n:03d}_{idx}",
                    f"s{x}",
                    f"s{y}",
                    f"s{deg}"
                ])

    print(f"Done. Wrote {filename}")


if __name__ == "__main__":
    random.seed(42)  # reproducible-ish
    write_submission()
    # If you want to visually inspect some small n:
    # layout_10 = make_layout_for_n(10)
    # plot_layout(layout_10, title="n=10")
