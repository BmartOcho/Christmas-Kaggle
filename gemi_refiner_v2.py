import math
import csv
import random
import pandas as pd
from copy import deepcopy
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#   CONFIG
# =========================
MAX_N = 200
GLOBAL_SEED = 999 # Different seed
PHYSICS_ITERS = 300
FAST_CHECK_DIST_SQ = 2.1 ** 2

# =========================
#   GEOMETRY & PHYSICS
# =========================
TREE_POLYGON = Polygon([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5)
])

def place_tree(x, y, deg):
    return translate(rotate(TREE_POLYGON, deg, origin=(0, 0)), xoff=x, yoff=y)

def layout_to_polys(layout):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]

def get_bounds(polys):
    if not polys: return 0,0,0,0
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return minx, miny, maxx, maxy

def get_score(layout, n) -> float:
    polys = layout_to_polys(layout)
    minx, miny, maxx, maxy = get_bounds(polys)
    side = max(maxx - minx, maxy - miny)
    return (side * side) / n

def solve_overlaps(layout, polys):
    n = len(layout)
    moves = 0
    for i in range(n):
        x1, y1, _ = layout[i]
        for j in range(i + 1, n):
            x2, y2, _ = layout[j]
            if (x1-x2)**2 + (y1-y2)**2 > FAST_CHECK_DIST_SQ: continue
            p1 = polys[i]
            p2 = polys[j]
            if p1.intersects(p2):
                moves += 1
                dx = x1 - x2
                dy = y1 - y2
                dist = math.hypot(dx, dy) or 0.01
                force = 0.02
                nx, ny = (dx/dist)*force, (dy/dist)*force
                layout[i] = (layout[i][0] + nx, layout[i][1] + ny, layout[i][2])
                layout[j] = (layout[j][0] - nx, layout[j][1] - ny, layout[j][2])
                polys[i] = place_tree(*layout[i])
                polys[j] = place_tree(*layout[j])
                x1, y1, _ = layout[i]
    return layout, polys, moves

def optimize_layout(layout):
    polys = layout_to_polys(layout)
    squeeze = 0.995
    for step in range(PHYSICS_ITERS):
        if step % 2 == 0:
            layout = [(x*squeeze, y*squeeze, d) for x,y,d in layout]
            polys = layout_to_polys(layout)
        layout, polys, moves = solve_overlaps(layout, polys)
        if moves == 0 and step > 50: break
    return layout

# =========================
#   REFINEMENT LOGIC
# =========================

def parse_group(df, n):
    rows = df[df["id"].str.startswith(f"{n:03d}_")]
    layout = []
    for _, row in rows.iterrows():
        x = float(str(row["x"]).replace('s',''))
        y = float(str(row["y"]).replace('s',''))
        deg = float(str(row["deg"]).replace('s',''))
        layout.append((x, y, deg))
    return layout

def tile_layout(base_layout, rows, cols):
    polys = layout_to_polys(base_layout)
    minx, miny, maxx, maxy = get_bounds(polys)
    block_w = maxx - minx
    block_h = maxy - miny
    
    spacing_x = block_w * 1.0
    spacing_y = block_h * 1.0
    
    new_layout = []
    start_x = -(spacing_x * cols) / 2 + spacing_x / 2
    start_y = -(spacing_y * rows) / 2 + spacing_y / 2

    for r in range(rows):
        for c in range(cols):
            dx = start_x + c * spacing_x
            dy = start_y + r * spacing_y
            for (bx, by, bdeg) in base_layout:
                # Add slight jitter to help physics settle
                jx = random.uniform(-0.05, 0.05)
                jy = random.uniform(-0.05, 0.05)
                new_layout.append((bx + dx + jx, by + dy + jy, bdeg))
    return new_layout

def main():
    print("Reading submission_gemmi.csv...")
    df = pd.read_csv("submission_gemmi.csv")
    
    layouts = {}
    scores = {}
    
    for n in range(1, MAX_N + 1):
        l = parse_group(df, n)
        layouts[n] = l
        scores[n] = get_score(l, n)
        
    print("Starting Refinement Loop...")
    improved = 0
    
    # Iterate over potential composite numbers
    for n in range(4, MAX_N + 1):
        best_score_n = scores[n]
        best_layout_n = layouts[n]
        
        # Find factors
        factors = []
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        
        for (r, c) in factors:
            # Try tiling (r x c) blocks of size (n // (r*c)) -> Wait, standard tiling is block size * copies
            # We want to construct N using blocks of size B. N = B * (rows * cols)
            
            # Strategy: Try to construct N using blocks of size 'base_n'
            # We iterate through possible base block sizes
            divs = [d for d in range(1, n) if n % d == 0]
            
            for base_n in divs:
                tiles_needed = n // base_n
                # Factorize tiles_needed into rows * cols
                grid_opts = []
                for i in range(1, int(math.sqrt(tiles_needed)) + 1):
                    if tiles_needed % i == 0:
                        grid_opts.append((i, tiles_needed // i))
                        
                base_layout = layouts[base_n]
                
                for (rows, cols) in grid_opts:
                    # Try this tiling
                    cand = tile_layout(base_layout, rows, cols)
                    cand = optimize_layout(cand) # SQUEEZE IT
                    s = get_score(cand, n)
                    
                    if s < best_score_n - 0.0001:
                        print(f"Refined N={n}: {best_score_n:.4f} -> {s:.4f} (using {base_n} tiled {rows}x{cols})")
                        best_score_n = s
                        best_layout_n = cand
                        improved += 1

        layouts[n] = best_layout_n
        scores[n] = best_score_n

    print(f"Refinement Complete. Improved {improved} groups.")
    
    # Output
    rows_out = []
    for n in range(1, MAX_N + 1):
        layout = layouts[n]
        for i, (x, y, deg) in enumerate(layout):
            rows_out.append({
                "id": f"{n:03d}_{i}", 
                "x": f"s{x}", "y": f"s{y}", "deg": f"s{deg}"
            })
            
    pd.DataFrame(rows_out).to_csv("submission_gemmi_go.csv", index=False)
    print("submission_gemmi_go.csv written.")

if __name__ == "__main__":
    main()