import matplotlib.pyplot as plt
import glob
import sys

def read_tile(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return [], []

    itr = iter(lines)
    try:
        num_points = int(next(itr))
        points = []
        for _ in range(num_points):
            parts = next(itr).split()
            points.append((float(parts[0]), float(parts[1])))
            
        num_tris = int(next(itr))
        tris = []
        for _ in range(num_tris):
            parts = next(itr).split()
            tris.append((int(parts[0]), int(parts[1]), int(parts[2])))
            
        return points, tris
    except StopIteration:
        return [], []

def main():
    files = glob.glob("tile_*.dat")
    if not files:
        print("No tile data files found. Run the C++ executable first.")
        return

    plt.figure(figsize=(10, 10))
    
    # Colors for different tiles to distinguish them
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple']
    
    print(f"Found {len(files)} tiles.")
    
    for i, fname in enumerate(sorted(files)):
        print(f"Plotting {fname}...")
        points, tris = read_tile(fname)
        
        if not points:
            print(f"  -> Empty point set in {fname}")
            continue
            
        c = colors[i % len(colors)]
        
        # Plot Points (Small dots)
        px = [p[0] for p in points]
        py = [p[1] for p in points]
        plt.scatter(px, py, s=1, c=c, alpha=0.5, label=fname)

        # Plot Triangles (Edges)
        for t in tris:
            if t[0] < len(points) and t[1] < len(points) and t[2] < len(points):
                pts = [points[t[0]], points[t[1]], points[t[2]], points[t[0]]]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                plt.plot(xs, ys, color=c, linewidth=0.5, alpha=0.4)
        
    plt.title("Unstructured Mesh Tiles (Overlaid)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    
    output_file = "mesh_visualization.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    main()