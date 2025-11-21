import math
import random
import csv
import time
from copy import deepcopy
from typing import List, Tuple

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

# =========================
#   CONFIG & TUNING
# =========================

MAX_N = 200
GLOBAL_SEED = 42

# We run a "Championship" for low N (Try multiple layouts, pick best)
# High N just uses the previous result to save time.
CHAMPIONSHIP_CUTOFF = 45  # For N < 45, we try both Fresh vs Incremental

# Physics iterations
ITERATIONS = 250 
PANIC_THRESHOLD = 50 # If stuck for 50 passes, teleport the tree

# Safety Distance (squared)
FAST_CHECK_DIST_SQ = 2.2 ** 2  # 2.2 is very safe for diagonal trees

# =========================
#   GEOMETRY
# =========================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5)
]
TREE_POLYGON = Polygon(TREE_COORDS)

def place_tree(x, y, deg):
    return translate(rotate(TREE_POLYGON, deg, origin=(0, 0)), xoff=x, yoff=y)

def layout_to_polys(layout):
    return [place_tree(x, y, deg) for (x, y, deg) in layout]

def get_bounding_side(polys) -> float:
    if not polys: return 0.0
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    return max(maxx - minx, maxy - miny)

def get_score(layout, n) -> float:
    polys = layout_to_polys(layout)
    s = get_bounding_side(polys)
    return (s * s) / n

# =========================
#   PHYSICS ENGINE
# =========================

def solve_overlaps(layout, polys):
    """
    Standard repulsion loop. Returns (layout, polys, number_of_moves)
    """
    n = len(layout)
    moves = 0
    
    for i in range(n):
        x1, y1, _ = layout[i]
        for j in range(i + 1, n):
            x2, y2, _ = layout[j]
            
            # Fast Check
            if (x1-x2)**2 + (y1-y2)**2 > FAST_CHECK_DIST_SQ:
                continue

            # Strict Check
            p1 = polys[i]
            p2 = polys[j]
            
            if p1.intersects(p2):
                moves += 1
                dx = x1 - x2
                dy = y1 - y2
                dist = math.hypot(dx, dy) or 0.01
                
                # Repulsion vector
                force = 0.02
                nx = (dx/dist) * force
                ny = (dy/dist) * force
                
                # Apply to Layout
                layout[i] = (layout[i][0] + nx, layout[i][1] + ny, layout[i][2])
                layout[j] = (layout[j][0] - nx, layout[j][1] - ny, layout[j][2])
                
                # Update Polys locally for next check
                polys[i] = place_tree(*layout[i])
                polys[j] = place_tree(*layout[j])
                
                # Update local coord vars
                x1, y1, _ = layout[i]

    return layout, polys, moves

def teleport_stuck_tree(layout, polys):
    """
    The Scalpel: Finds the tree causing the most overlaps and 
    teleports it to the outer rim of the cluster.
    """
    n = len(layout)
    # Count collisions per tree
    collisions = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if polys[i].intersects(polys[j]):
                collisions[i] += 1
                collisions[j] += 1
    
    # Find worst offender
    worst_idx = -1
    max_col = 0
    for i in range(n):
        if collisions[i] > max_col:
            max_col = collisions[i]
            worst_idx = i
            
    if worst_idx != -1:
        # Teleport to random spot on perimeter
        # Calculate center of mass
        xs = [x for x,_,_ in layout]
        ys = [y for _,y,_ in layout]
        cx, cy = sum(xs)/n, sum(ys)/n
        
        angle = random.uniform(0, 6.28)
        radius = math.sqrt(n) * 0.8 # Guess radius
        nx = cx + math.cos(angle) * radius
        ny = cy + math.sin(angle) * radius
        
        layout[worst_idx] = (nx, ny, random.uniform(0, 360))
        polys[worst_idx] = place_tree(*layout[worst_idx])
        
    return layout, polys

def optimize_layout(layout, iterations=200):
    """
    Squeeze, Shake, and Teleport.
    """
    polys = layout_to_polys(layout)
    n = len(layout)
    
    # Squeeze factor
    squeeze = 0.992 if n < 50 else 0.996
    
    stuck_counter = 0
    
    best_layout = deepcopy(layout)
    best_score = float('inf')

    for step in range(iterations):
        
        # 1. Gravity (Squeeze towards center)
        if step % 2 == 0: # Don't squeeze every frame, let it settle
            new_layout = []
            for x, y, d in layout:
                new_layout.append((x * squeeze, y * squeeze, d))
            layout = new_layout
            polys = layout_to_polys(layout)

        # 2. Physics (Resolve Overlaps)
        layout, polys, moves = solve_overlaps(layout, polys)
        
        # 3. Anti-Jamming (Teleport)
        if moves > 0:
            stuck_counter += 1
        else:
            stuck_counter = 0
            
        if stuck_counter > PANIC_THRESHOLD:
            # Teleport ONE tree instead of expanding everyone
            layout, polys = teleport_stuck_tree(layout, polys)
            stuck_counter = 0 # Reset
            
        # 4. Record Best Valid State
        if moves == 0:
            # Double check strictly
            s = get_bounding_side(polys)
            if s < best_score:
                best_score = s
                best_layout = deepcopy(layout)
    
    return best_layout

def ensure_valid_teleport(layout):
    """
    Final safety pass. If overlaps exist, teleport untill fixed.
    No expansion allowed.
    """
    polys = layout_to_polys(layout)
    passes = 0
    while True:
        layout, polys, moves = solve_overlaps(layout, polys)
        if moves == 0:
            return layout
        
        passes += 1
        if passes % 20 == 0:
            # If simple physics failing, teleport
            layout, polys = teleport_stuck_tree(layout, polys)
        
        if passes > 500:
            # Emergency break (should rarely happen with teleport)
            print("Warning: Emergency expansion used.")
            layout = [(x*1.01, y*1.01, d) for x,y,d in layout]
            polys = layout_to_polys(layout)

# =========================
#   GENERATORS
# =========================

def get_spiral_layout(n):
    layout = []
    # Sunflower spiral
    phi = (1 + math.sqrt(5)) / 2
    for k in range(n):
        r = 0.6 * math.sqrt(k) # 0.6 spacing
        theta = 2 * math.pi * k / (phi**2)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        layout.append((x, y, random.uniform(0, 360)))
    return layout

def get_incremental_layout(prev_layout, n):
    layout = deepcopy(prev_layout)
    # Add one tree at the edge
    xs = [x for x,_,_ in layout]
    ys = [y for _,y,_ in layout]
    cx, cy = sum(xs)/(n-1), sum(ys)/(n-1)
    
    angle = random.uniform(0, 6.28)
    dist = math.sqrt(n) * 0.6
    nx = cx + math.cos(angle) * dist
    ny = cy + math.sin(angle) * dist
    layout.append((nx, ny, random.uniform(0, 360)))
    return layout

# =========================
#   MAIN
# =========================

def main():
    random.seed(GLOBAL_SEED)
    final_layouts = {}
    total_score = 0.0
    
    print(f"Starting CHAMPIONSHIP optimization for 1 to {MAX_N}...")
    start_time = time.time()
    
    # Previous layout holder
    prev_best = []

    for n in range(1, MAX_N + 1):
        
        candidates = []
        
        # --- STRATEGY 1: INCREMENTAL (The Incumbent) ---
        if n > 1:
            cand_inc = get_incremental_layout(prev_best, n)
            cand_inc = optimize_layout(cand_inc, iterations=ITERATIONS)
            cand_inc = ensure_valid_teleport(cand_inc)
            score_inc = get_score(cand_inc, n)
            candidates.append((score_inc, cand_inc, "Inc"))
        else:
            # N=1
            cand_inc = [(0.0, 0.0, 0.0)]
            candidates.append((0.0, cand_inc, "Inc"))

        # --- STRATEGY 2: FRESH SPIRAL (The Challenger) ---
        # Only run this for lower N where it matters most, or if N is small enough to be fast
        if n <= CHAMPIONSHIP_CUTOFF:
            cand_spi = get_spiral_layout(n)
            cand_spi = optimize_layout(cand_spi, iterations=ITERATIONS + 100) # Give spiral more time to settle
            cand_spi = ensure_valid_teleport(cand_spi)
            score_spi = get_score(cand_spi, n)
            candidates.append((score_spi, cand_spi, "Spi"))

        # --- PICK WINNER ---
        candidates.sort(key=lambda x: x[0]) # Sort by score (lowest is best)
        best_score, best_lay, source = candidates[0]
        
        # Center the winner
        xs = [x for x,_,_ in best_lay]
        ys = [y for _,y,_ in best_lay]
        mx, my = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        best_lay = [(x-mx, y-my, d) for x,y,d in best_lay]
        
        final_layouts[n] = best_lay
        prev_best = best_lay # Set the new incumbent
        
        total_score += best_score
        
        # Logging
        if n % 5 == 0 or n == 1:
            elapsed = time.time() - start_time
            msg = f"N={n:03d} | Sc={best_score:.4f} | src={source}"
            if len(candidates) > 1:
                # Show if Spiral beat Incremental
                diff = candidates[1][0] - candidates[0][0]
                if source == "Spi":
                    msg += f" (Beat Inc by {diff:.4f})"
            print(msg)

    print(f"\nOptimization Complete.")
    print(f"Est Total Score: {total_score:.4f}")

    # Write CSV
    with open("submission_championship.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])
        for n in range(1, MAX_N + 1):
            layout = final_layouts[n]
            for i, (x, y, deg) in enumerate(layout):
                w.writerow([f"{n:03d}_{i}", f"s{x}", f"s{y}", f"s{deg}"])
    
    print("submission_championship.csv written.")

if __name__ == "__main__":
    main()