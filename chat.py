import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


# ============================================================
# TREE GEOMETRY (MATCHES COMPETITION DEFINITION)
# ============================================================

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
    (-0.25/2, 0.5),
])


def place_tree(x, y, deg):
    return translate(rotate(TREE_POLYGON, deg, origin=(0,0)), xoff=x, yoff=y)


def bounding_side(polys):
    xs = []; ys = []
    for p in polys:
        for (x,y) in p.exterior.coords:
            xs.append(x); ys.append(y)
    return max(max(xs)-min(xs), max(ys)-min(ys))


def overlap(p1, p2):
    return p1.intersects(p2) and not p1.touches(p2)


# ============================================================
# SEARCH FOR BEST SPACING FOR N2 PAIR
# ============================================================

def search_n2():
    A = (0.0, 0.0, 0.0)
    polyA = place_tree(*A)

    best = None
    best_side = 999

    base_deg = 180.0

    slide_x = [i * 0.01 for i in range(0, 101)]
    slide_y = [i * 0.01 for i in range(0, 151)]

    for dx in slide_x:
        for dy in slide_y:
            Bx = dx
            By = dy
            Bd = base_deg

            polyB = place_tree(Bx, By, Bd)
            if overlap(polyA, polyB):
                continue

            side = bounding_side([polyA, polyB])
            if side < best_side:
                best_side = side
                best = (Bx, By, Bd)

    return A, best, best_side


# ============================================================
# ALIGNMENT FUNCTIONS
# ============================================================

def move_B_to_touch_A0(Bx, By, Bd, A0=(0.0, 0.8), B5=(0.05, 0.68)):
    Ax0, Ay0 = A0
    Bx5, By5 = B5
    dx = Ax0 - Bx5
    dy = Ay0 - By5
    return (Bx + dx, By + dy, Bd)


def move_pair_to_touch_A10(A, B, A10=(0.35, 0.00), A5=(0.35, 0.00)):
    Ax, Ay, Ad = A
    Bx, By, Bd = B
    dx = A5[0] - A10[0]
    dy = A5[1] - A10[1]
    return (Ax + dx, Ay + dy, Ad), (Bx + dx, By + dy, Bd)


# ============================================================
# GRID DUPLICATION: 3 COLUMNS × 2 ROWS = 6 PAIRS (12 TREES)
# ============================================================

def duplicate_pairs_grid(A_final, B_final, spacing_x=1.0, spacing_y=1.0):
    """
    Creates:
       Row 1: 3 copies of A_final + B_final, spaced horizontally
       Row 2: Same thing shifted up by spacing_y
    """
    pairs = []

    Ax, Ay, Ad = A_final
    Bx, By, Bd = B_final

    # Column offsets
    cols = [0, spacing_x, spacing_x*2]

    # Row offsets (0 and 1x vertical jump)
    rows = [0, spacing_y]

    for ry in rows:
        for cx in cols:
            A_shift = (Ax + cx, Ay + ry, Ad)
            B_shift = (Bx + cx, By + ry, Bd)
            pairs.append((A_shift, B_shift))

    return pairs


# ============================================================
# VISUALIZER FOR ENTIRE GRID
# ============================================================

def visualize_grid(pairs, filename="n2_grid.png"):
    fig, ax = plt.subplots(figsize=(12, 12))

    for idx, (A, B) in enumerate(pairs):
        Ax, Ay, Ad = A
        Bx, By, Bd = B
        polyA = place_tree(Ax, Ay, Ad)
        polyB = place_tree(Bx, By, Bd)

        # Tree A
        xs, ys = polyA.exterior.xy
        ax.fill(xs, ys, alpha=0.35, color="green")
        ax.text(Ax, Ay+0.9, f"A pair {idx}", fontsize=8)

        # Tree B
        xs2, ys2 = polyB.exterior.xy
        ax.fill(xs2, ys2, alpha=0.35, color="orange")

    ax.set_aspect("equal", "box")
    plt.title("6 N2 Pairs (3 Across, Then 3 Across Again Above)")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Running N2 locking search...")
    A, B_best, side = search_n2()
    print("Original best B:", B_best)
    print("Bounding box side:", side)

    print("\nMoving B5 to A0...")
    B_adj = move_B_to_touch_A0(*B_best)

    print("\nAligning A10 to A5/B0...")
    A_final, B_final = move_pair_to_touch_A10(A, B_adj)

    print("\nGenerating 6-pair grid with closer horizontal spacing...")
    pairs = duplicate_pairs_grid(A_final, B_final, spacing_x=.75, spacing_y=1.0)
    
    print(f"Total tree pairs: {len(pairs)}")
    visualize_grid(pairs)


if __name__ == "__main__":
    main()
