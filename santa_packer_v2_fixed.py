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
    p = rotate(TREE_POLYGON, deg, origin=(0, 0))
    p = translate(p, xoff=x, yoff=y)
    return p


# =========================
#  HELPER: BOUND & OVERLAP
# =========================

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

    xs = [x for x,_,_ in layout]
    ys = [y for _,y,_ in layout]
    cx = (max(xs)+min(xs))/2
    cy = (max(ys)+min(ys))/2

    return [(x-cx, y-cy, deg) for (x,y,deg) in layout]


# =========================
#  CLEANUP PASS (guaranteed)
# =========================

def collision_cleanup(layout, iterations=60):
    layout = layout[:]  # copy
    polys = [place_tree(x,y,deg) for (x,y,deg) in layout]

    for _ in range(iterations):
        moved = False
        for i in range(len(layout)):
            for j in range(i+1, len(layout)):
                p1 = polys[i]
                p2 = polys[j]
                if p1.intersects(p2) and not p1.touches(p2):
                    x1,y1,d1 = layout[i]
                    x2,y2,d2 = layout[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = math.hypot(dx, dy) or 1.0
                    push = 0.02 / dist
                    x1 += dx * push
                    y1 += dy * push
                    x2 -= dx * push
                    y2 -= dy * push
                    layout[i] = (x1,y1,d1)
                    layout[j] = (x2,y2,d2)
                    polys[i] = place_tree(x1,y1,d1)
                    polys[j] = place_tree(x2,y2,d2)
                    moved = True
        if not moved:
            break

    return layout


# =========================
#  SAFE REFINEMENT ANNEAL
# =========================

def anneal_layout(initial_layout, steps=4000,
                  initial_step=0.015, final_step=0.002,
                  initial_rot=3.0, final_rot=1.0,
                  initial_T=0.01, final_T=0.0005):

    layout = initial_layout[:]
    polys = [place_tree(x,y,deg) for (x,y,deg) in layout]

    curr_score = bounding_side(polys)
    best_score = curr_score
    best_layout = layout[:]

    for step in range(steps):
        t = step/steps
        step_size = initial_step*(1-t) + final_step*t
        rot_size  = initial_rot*(1-t) + final_rot*t
        T         = initial_T*(1-t) + final_T*t

        i = random.randint(0, len(layout)-1)
        x,y,deg = layout[i]

        nx = x + random.uniform(-step_size, step_size)
        ny = y + random.uniform(-step_size, step_size)
        ndeg = (deg + random.uniform(-rot_size, rot_size)) % 360

        new_poly = place_tree(nx,ny,ndeg)

        if has_overlap(new_poly, polys, i):
            continue

        old_poly = polys[i]
        polys[i] = new_poly

        new_score = bounding_side(polys)
        delta = new_score - curr_score

        if delta <= 0 or random.random() < math.exp(-delta/max(T,1e-9)):
            layout[i] = (nx,ny,ndeg)
            curr_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_layout = layout[:]
        else:
            polys[i] = old_poly

    return best_layout, best_score


# =========================
#  KEY SIZE STRATEGY
# =========================

KEY_NS = [10, 20, 40, 80, 120, 160, 200]

def nearest_key(n):
    for k in KEY_NS:
        if n <= k:
            return k
    return KEY_NS[-1]


def build_key_layouts():
    key_layouts = {}
    for k in KEY_NS:
        print(f"[KEY] Begin n={k}")
        base = nested_row_layout(k, spacing=0.8)

        if k <= 20:
            steps = 15000
        elif k <= 80:
            steps = 20000
        else:
            steps = 26000

        deep, sc = anneal_layout(base, steps=steps)
        deep = collision_cleanup(deep)

        print(f"[KEY] Done n={k}, best side ≈ {sc:.4f}")
        key_layouts[k] = deep
    return key_layouts


def make_layout_for_n(n, key_layouts):
    if n in KEY_NS:
        return key_layouts[n]

    k = nearest_key(n)
    base = key_layouts[k][:n]

    # slight expansion before refinement
    base = [(x*1.02, y*1.02, deg) for (x,y,deg) in base]

    # short refinement anneal
    steps = 2000 if n <= 80 else 4000
    refined, sc = anneal_layout(base, steps=steps)

    # final defensive cleanup
    refined = collision_cleanup(refined)
    return refined


# =========================
#  SUBMISSION OUTPUT
# =========================

def write_submission(filename="submission.csv"):
    print("Building key layouts...")
    key_layouts = build_key_layouts()
    print("Key layouts ready.")

    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])

        for n in range(1,201):
            print(f"[n={n}] generating...")
            layout = make_layout_for_n(n, key_layouts)

            for idx, (x,y,deg) in enumerate(layout):
                w.writerow([
                    f"{n:03d}_{idx}",
                    f"s{x}",
                    f"s{y}",
                    f"s{deg}"
                ])

    print("Done. Wrote submission.csv")


if __name__ == "__main__":
    random.seed(42)
    write_submission()
