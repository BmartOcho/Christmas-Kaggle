import math
import random
import csv
import time
from copy import deepcopy
from typing import List, Tuple

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#   CONFIG
# =========================

MAX_N = 200
GLOBAL_SEED = 42

# Physics Settings
PHYSICS_ITERS = 600    # Increased for better squeezing
PANIC_THRESHOLD = 50   

# Safety for overlap checks
FAST_CHECK_DIST_SQ = 2.1 ** 2 

# =========================
#   EXACT TREE SHAPE
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

# =========================
#   PHYSICS ENGINE
# =========================

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

def teleport_stuck_tree(layout, polys):
    n = len(layout)
    collisions = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]):
                collisions[i] += 1; collisions[j] += 1
    
    worst_idx = -1
    max_col = 0
    for i in range(n):
        if collisions[i] > max_col:
            max_col = collisions[i]; worst_idx = i
            
    if worst_idx != -1:
        xs = [x for x,_,_ in layout]
        ys = [y for _,y,_ in layout]
        cx, cy = sum(xs)/n, sum(ys)/n
        angle = random.uniform(0, 6.28)
        radius = math.sqrt(n) * 0.7 # Teleport closer
        nx = cx + math.cos(angle) * radius
        ny = cy + math.sin(angle) * radius
        layout[worst_idx] = (nx, ny, random.uniform(0, 360))
        polys[worst_idx] = place_tree(*layout[worst_idx])
    return layout, polys

def optimize_layout_physics(layout, iterations=200):
    polys = layout_to_polys(layout)
    n = len(layout)
    squeeze = 0.992 if n < 50 else 0.996
    stuck_counter = 0
    best_layout = deepcopy(layout)
    best_score = float('inf')

    for step in range(iterations):
        if step % 2 == 0:
            new_layout = [(x*squeeze, y*squeeze, d) for x,y,d in layout]
            layout = new_layout
            polys = layout_to_polys(layout)

        layout, polys, moves = solve_overlaps(layout, polys)
        
        if moves > 0: stuck_counter += 1
        else: stuck_counter = 0
            
        if stuck_counter > PANIC_THRESHOLD:
            layout, polys = teleport_stuck_tree(layout, polys)
            stuck_counter = 0
            
        if moves == 0:
            polys = layout_to_polys(layout) # Refresh exact
            s = get_score(layout, n)
            if s < best_score:
                best_score = s
                best_layout = deepcopy(layout)
    return best_layout

# =========================
#   FACTORIZATION ENGINE
# =========================

def get_factors(n):
    """Returns list of factor pairs (r, c) such that r*c = n."""
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return factors

def generate_tiled_layout(base_layout, rows, cols):
    """
    Tiles 'base_layout' into a 'rows x cols' grid.
    """
    polys = layout_to_polys(base_layout)
    minx, miny, maxx, maxy = get_bounds(polys)
    
    # Dimensions of the block
    block_w = maxx - minx
    block_h = maxy - miny
    
    # Tighter spacing
    spacing_x = block_w * 1.0
    spacing_y = block_h * 1.0
    
    new_layout = []
    
    # Center offset
    total_w = spacing_x * cols
    total_h = spacing_y * rows
    start_x = -total_w / 2 + spacing_x / 2
    start_y = -total_h / 2 + spacing_y / 2

    for r in range(rows):
        for c in range(cols):
            dx = start_x + c * spacing_x
            dy = start_y + r * spacing_y
            
            for (bx, by, bdeg) in base_layout:
                new_layout.append((bx + dx, by + dy, bdeg))
                
    return new_layout

# =========================
#   MAIN SOLVER
# =========================

def main():
    random.seed(GLOBAL_SEED)
    best_layouts = {} 
    total_score = 0.0
    
    print(f"Starting GENERATOR (v5) optimization...")
    start_time = time.time()

    for n in range(1, MAX_N + 1):
        
        candidates = []
        
        # --- 1. PHYSICS CANDIDATE ---
        # Run physics for all small numbers and primes to avoid "long skinny" tiling issues
        if n <= 65 or n in [67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]: 
            # Initial spiral
            layout = []
            for k in range(n):
                r = 0.6 * math.sqrt(k)
                theta = 2.4 * k 
                layout.append((r*math.cos(theta), r*math.sin(theta), random.uniform(0,360)))
            
            layout = optimize_layout_physics(layout, iterations=PHYSICS_ITERS)
            score = get_score(layout, n)
            candidates.append((score, layout, "Physics"))

        # --- 2. TILING CANDIDATES ---
        divs = [i for i in range(1, n) if n % i == 0]
        
        for d in divs:
            num_blocks = n // d
            grid_options = get_factors(num_blocks)
            
            if d in best_layouts:
                base_layout = best_layouts[d]
                
                for (rows, cols) in grid_options:
                    tiled = generate_tiled_layout(base_layout, rows, cols)
                    s = get_score(tiled, n)
                    candidates.append((s, tiled, f"Tile({d}x{rows}x{cols})"))
                    
                    if rows != cols:
                        tiled = generate_tiled_layout(base_layout, cols, rows)
                        s = get_score(tiled, n)
                        candidates.append((s, tiled, f"Tile({d}x{cols}x{rows})"))

        # --- PICK WINNER ---
        if not candidates:
            # INTELLIGENT FALLBACK (Fixes the 2523 score bug)
            prev_layout = best_layouts[n-1]
            xs = [x for x,_,_ in prev_layout]
            ys = [y for _,y,_ in prev_layout]
            cx, cy = sum(xs)/(n-1), sum(ys)/(n-1)
            
            angle = random.uniform(0, 6.28)
            dist = math.sqrt(n) * 0.6
            nx = cx + math.cos(angle) * dist
            ny = cy + math.sin(angle) * dist
            
            layout = prev_layout + [(nx, ny, random.uniform(0, 360))]
            layout = optimize_layout_physics(layout, iterations=400)
            candidates.append((get_score(layout, n), layout, "Fallback"))

        candidates.sort(key=lambda x: x[0]) 
        best_score, best_lay, source = candidates[0]
        
        # Center Result
        xs = [x for x,_,_ in best_lay]
        ys = [y for _,y,_ in best_lay]
        mx, my = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        best_lay = [(x-mx, y-my, d) for x,y,d in best_lay]
        
        best_layouts[n] = best_lay
        total_score += best_score
        
        if n % 10 == 0 or n < 10:
            print(f"N={n:03d} | Score={best_score:.4f} | Method={source}")

    print(f"\nGenerator Complete. Est Score: {total_score:.4f}")

    # Write CSV
    with open("submission_gemmi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, MAX_N + 1):
            layout = best_layouts[n]
            for i, (x, y, deg) in enumerate(layout):
                w.writerow([f"{n:03d}_{i}", f"s{x}", f"s{y}", f"s{deg}"])
    
    print("submission_gemmi.csv written.")

if __name__ == "__main__":
    main()