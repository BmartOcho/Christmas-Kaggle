import csv
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
    """Place a tree polygon at (x, y) with rotation deg."""
    return translate(rotate(TREE_POLYGON, deg, origin=(0, 0)), xoff=x, yoff=y)


def overlap(p1, p2):
    """True if polygons have area overlap (not just touching)."""
    return p1.intersects(p2) and not p1.touches(p2)


def single_tree_width():
    xs = [p[0] for p in TREE_POLYGON.exterior.coords]
    return max(xs) - min(xs)


def single_tree_height():
    ys = [p[1] for p in TREE_POLYGON.exterior.coords]
    return max(ys) - min(ys)


# ============================================================
# N2 PAIR SEARCH + ALIGNMENT (SAME LOGIC YOU USED BEFORE)
# ============================================================

def search_n2():
    """
    Find best position for B relative to A (upright at origin) to minimize
    bounding box side, with B rotated 180°.
    """
    A = (0.0, 0.0, 0.0)
    polyA = place_tree(*A)

    best = None
    best_side = 999.0
    base_deg = 180.0

    slide_x = [i * 0.01 for i in range(0, 101)]
    slide_y = [i * 0.01 for i in range(0, 151)]

    for dx in slide_x:
        for dy in slide_y:
            Bx, By, Bd = dx, dy, base_deg
            polyB = place_tree(Bx, By, Bd)

            if overlap(polyA, polyB):
                continue

            # Combined bounding box
            min_x = min(polyA.bounds[0], polyB.bounds[0])
            max_x = max(polyA.bounds[2], polyB.bounds[2])
            min_y = min(polyA.bounds[1], polyB.bounds[1])
            max_y = max(polyA.bounds[3], polyB.bounds[3])

            side = max(max_x - min_x, max_y - min_y)

            if side < best_side:
                best_side = side
                best = (Bx, By, Bd)

    return A, best


def move_B_to_touch_A0(Bx, By, Bd, A0=(0.0, 0.8), B5=(0.05, 0.68)):
    """
    Shift B so that its vertex B5 lands on A0.
    These coordinates are from your Illustrator / N2-adjusted inspection.
    """
    dx = A0[0] - B5[0]
    dy = A0[1] - B5[1]
    return (Bx + dx, By + dy, Bd)


def move_pair_to_touch_A10(A, B, A10=(0.35, 0.0), A5=(0.35, 0.0)):
    """
    Final tiny alignment so A10 and A5/B0 coincide.
    In practice this doesn't move much, but we keep it for consistency.
    """
    Ax, Ay, Ad = A
    Bx, By, Bd = B

    dx = A5[0] - A10[0]
    dy = A5[1] - A10[1]

    return (Ax + dx, Ay + dy, Ad), (Bx + dx, By + dy, Bd)


# ============================================================
# BUILD LAYOUT FOR A SINGLE N USING DUOS
# ============================================================

def build_layout_for_n(n, A_final, B_final,
                       pair_spacing_x, row_spacing_y,
                       pairs_per_row=20):
    """
    Build layout for given n (1..200) using:
    - as many tight duos (A,B) as possible
    - if n is odd, add one extra A at end of the last row
    All duos share same orientation.
    """
    placements = []

    Ax0, Ay0, Ad = A_final
    Bx0, By0, Bd = B_final

    if n == 1:
        # Single upright tree (A) at its base position
        placements.append((Ax0, Ay0, Ad))
        return placements

    pair_count = n // 2
    extra_single = n % 2

    # Place full duos
    for p in range(pair_count):
        row = p // pairs_per_row
        col = p % pairs_per_row

        dx = col * pair_spacing_x
        dy = row * row_spacing_y

        placements.append((Ax0 + dx, Ay0 + dy, Ad))  # A of the pair
        placements.append((Bx0 + dx, By0 + dy, Bd))  # B of the pair

    # If odd, add one more A to the right of the last full pair in the last row
    if extra_single == 1:
        if pair_count > 0:
            last_pair_index = pair_count - 1
            row = last_pair_index // pairs_per_row
            col = last_pair_index % pairs_per_row
            base_dx = col * pair_spacing_x
            base_dy = row * row_spacing_y
        else:
            # If somehow n == 1, we already returned above, so this is safe
            base_dx = 0.0
            base_dy = 0.0

        extra_x = Ax0 + base_dx + pair_spacing_x
        extra_y = Ay0 + base_dy
        placements.append((extra_x, extra_y, Ad))

    return placements


# ============================================================
# CSV EXPORT FOR ALL N = 1..200
# ============================================================

def write_full_submission_csv(filename, A_final, B_final):
    pair_spacing_x = single_tree_width()
    row_spacing_y = single_tree_height()  # effectively 1.0 with this polygon

    rows = []

    for n in range(1, 201):
        layout = build_layout_for_n(
            n,
            A_final,
            B_final,
            pair_spacing_x=pair_spacing_x,
            row_spacing_y=row_spacing_y,
            pairs_per_row=20  # adjust if you want wider rows
        )

        for (x, y, d) in layout:
            rows.append((n, x, y, d))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "x", "y", "deg"])
        for n, x, y, d in rows:
            writer.writerow([n, f"{x:.6f}", f"{y:.6f}", f"{d:.6f}"])

    print(f"Wrote {len(rows)} trees total across n=1..200 to {filename}.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Searching tight N2 pair...")
    A, B_best = search_n2()

    print("Aligning B5 to A0...")
    B_adj = move_B_to_touch_A0(*B_best)

    print("Aligning A10 to A5/B0...")
    A_final, B_final = move_pair_to_touch_A10(A, B_adj)

    print("Generating full 1..200 submission CSV...")
    write_full_submission_csv("submission.csv", A_final, B_final)


if __name__ == "__main__":
    main()
