#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <string>
#include <mpi.h>
#include <petsc/private/dmpleximpl.h>

#if !defined(ANSI_DECLARATORS)
  #define ANSI_DECLARATORS
#endif
#include <triangle.h>

// =========================================================
// JITTER + LLOYD SMOOTH MESH GENERATOR FOR [0,1]^2 WITH FIXED BOUNDARIES
// =========================================================
// 
// Strategy:
// 1. Boundary Gen: Create points explicitly on [0,1]^2 edges.
// 2. Interior Gen: Create Hex Grid points, REJECTING those near edges.
// 3. Annealing: Jitter -> Smooth loop.
// 4. Constraint: Boundary nodes only move tangentially.
// =========================================================

const int TILDE_DIM = 3; 
const double DOMAIN_SIZE = 1.0;
const double TARGET_SPACING = 0.01; 

const double EPSILON = 1e-13;     
const double START_JITTER = 0.30;
double MIN_ANGLE_THRESHOLD = 30.0;

const int ANNEAL_CYCLES = 3; 
const int FINAL_SMOOTH_ITERS = 2;

struct Point {
    double x, y;
    uint64_t id = 0; // Added for deterministic jitter
};

// Taken from PETSc in src/dm/impls/plex/generators/triangle/trigenerate.c
static void InitInput_Triangle(struct triangulateio *inputCtx)
{
  inputCtx->numberofpoints             = 0;
  inputCtx->numberofpointattributes    = 0;
  inputCtx->pointlist                  = NULL;
  inputCtx->pointattributelist         = NULL;
  inputCtx->pointmarkerlist            = NULL;
  inputCtx->numberofsegments           = 0;
  inputCtx->segmentlist                = NULL;
  inputCtx->segmentmarkerlist          = NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->trianglelist               = NULL;
  inputCtx->numberofholes              = 0;
  inputCtx->holelist                   = NULL;
  inputCtx->numberofregions            = 0;
  inputCtx->regionlist                 = NULL;
}

static void InitOutput_Triangle(struct triangulateio *outputCtx)
{
  outputCtx->numberofpoints        = 0;
  outputCtx->pointlist             = NULL;
  outputCtx->pointattributelist    = NULL;
  outputCtx->pointmarkerlist       = NULL;
  outputCtx->numberoftriangles     = 0;
  outputCtx->trianglelist          = NULL;
  outputCtx->triangleattributelist = NULL;
  outputCtx->neighborlist          = NULL;
  outputCtx->segmentlist           = NULL;
  outputCtx->segmentmarkerlist     = NULL;
  outputCtx->numberofedges         = 0;
  outputCtx->edgelist              = NULL;
  outputCtx->edgemarkerlist        = NULL;
}

static void FiniOutput_Triangle(struct triangulateio *outputCtx)
{
  free(outputCtx->pointlist);
  free(outputCtx->pointmarkerlist);
  free(outputCtx->segmentlist);
  free(outputCtx->segmentmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
}

// --- Boundary Logic ---

// Returns true if p is on a boundary.
// Modifies dx, dy to ensure movement is only tangential (sliding).
bool apply_boundary_constraint(Point& p, double& dx, double& dy) {
    bool on_boundary = false;

    // Left (x=0)
    if (std::abs(p.x) < EPSILON) {
        p.x = 0.0; dx = 0.0; 
        on_boundary = true;
    }
    // Right (x=1)
    else if (std::abs(p.x - DOMAIN_SIZE) < EPSILON) {
        p.x = DOMAIN_SIZE; dx = 0.0;
        on_boundary = true;
    }

    // Bottom (y=0)
    if (std::abs(p.y) < EPSILON) {
        p.y = 0.0; dy = 0.0; 
        on_boundary = true;
    }
    // Top (y=1)
    else if (std::abs(p.y - DOMAIN_SIZE) < EPSILON) {
        p.y = DOMAIN_SIZE; dy = 0.0;
        on_boundary = true;
    }

    return on_boundary;
}

struct Triangle {
    int v0, v1, v2;
    bool inCircumcircle(const Point& p, const std::vector<Point>& pts) const {
        const Point& a = pts[v0]; const Point& b = pts[v1]; const Point& c = pts[v2];
        double ax = a.x-p.x; double ay = a.y-p.y;
        double bx = b.x-p.x; double by = b.y-p.y;
        double cx = c.x-p.x; double cy = c.y-p.y;
        return ((ax*ax + ay*ay)*(bx*cy - cx*by) - (bx*bx + by*by)*(ax*cy - cx*ay) + (cx*cx + cy*cy)*(ax*by - bx*ay)) > 1e-10;
    }
};

struct Edge {
    int v0, v1;
    bool operator==(const Edge& o) const { return (v0 == o.v0 && v1 == o.v1) || (v0 == o.v1 && v1 == o.v0); }
};

// --- Stateless Random ---
struct RngState { uint64_t s; };
uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}
double next_double(RngState& state) {
    return (splitmix64(state.s) >> 11) * (1.0 / 9007199254740992.0);
}

// Helper for hashing
void hash_combine(uint64_t& h, double v) {
    union { double d; uint64_t u; } c;
    c.d = v;
    h ^= c.u + 0x9e3779b9 + (h << 6) + (h >> 2);
}

// --- Jitter ---
void apply_jitter(std::vector<Point>& points, double amount, int seed_offset) {
    for (size_t i = 0; i < points.size(); ++i) {
        // Hash based on STABLE ID + Cycle Offset
        // This ensures that even if the point moves, the jitter sequence is deterministic
        uint64_t h = points[i].id;
        
        // Mix in the cycle/seed offset to ensure different jitter each step
        h ^= seed_offset + 0x9e3779b9 + (h << 6) + (h >> 2);

        RngState rng = {h};

        double jx = (next_double(rng) - 0.5) * 2.0 * amount * TARGET_SPACING;
        double jy = (next_double(rng) - 0.5) * 2.0 * amount * TARGET_SPACING;

        // CRITICAL: Boundary nodes effectively ignore perpendicular jitter here
        apply_boundary_constraint(points[i], jx, jy);

        points[i].x += jx;
        points[i].y += jy;
    }
}

// Wrapper for Triangle library
std::vector<Triangle> triangulation(const std::vector<Point>& points) {
    struct triangulateio in;
    struct triangulateio out;    
    
    // Initialize structures
    InitInput_Triangle(&in);
    InitOutput_Triangle(&out);

    in.numberofpoints = points.size();
    in.pointlist = new double[in.numberofpoints * 2];

    for (size_t i = 0; i < points.size(); ++i) {
        in.pointlist[i * 2] = points[i].x;
        in.pointlist[i * 2 + 1] = points[i].y;
    }

    char args[32];
    (void*)PetscStrncpy(args, "ezQ", sizeof(args));    

    triangulate(args, &in, &out, NULL);

    const PetscInt numCells = out.numberoftriangles;
    std::vector<Triangle> mesh;
    mesh.reserve(numCells);

    for (int i = 0; i < numCells; ++i) {
        Triangle t;
        t.v0 = (int)out.trianglelist[i * 3 + 0];
        t.v1 = (int)out.trianglelist[i * 3 + 1];
        t.v2 = (int)out.trianglelist[i * 3 + 2];
        mesh.push_back(t);
    }

    delete[] in.pointlist;
    FiniOutput_Triangle(&out);

    return mesh;
}

// --- Relaxation ---

// Helper: Calculate minimum angle (degrees) of a triangle
double clamp_val(double v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); }

double get_min_angle_deg(const Point& a, const Point& b, const Point& c) {
    double ab2 = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
    double bc2 = (b.x-c.x)*(b.x-c.x) + (b.y-c.y)*(b.y-c.y);
    double ca2 = (c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y);
    
    double ab = std::sqrt(ab2);
    double bc = std::sqrt(bc2);
    double ca = std::sqrt(ca2);

    if (ab < 1e-9 || bc < 1e-9 || ca < 1e-9) return 0.0;

    // Law of Cosines
    double ang_a = std::acos(clamp_val((ab2 + ca2 - bc2) / (2.0 * ab * ca)));
    double ang_b = std::acos(clamp_val((ab2 + bc2 - ca2) / (2.0 * ab * bc)));
    double ang_c = std::acos(clamp_val((ca2 + bc2 - ab2) / (2.0 * ca * bc)));

    return std::min({ang_a, ang_b, ang_c}) * 180.0 / 3.14159265358979323846;
}

// Helper for relax_points to calculate local min angle
double calculate_local_min_angle(int node_idx, const Point& node_pos, 
                                 const std::vector<Point>& points, 
                                 const std::vector<Triangle>& mesh, 
                                 const std::vector<int>& connected_tris) {
    double min_a = 180.0;
    for (int t_idx : connected_tris) {
        const auto& tri = mesh[t_idx];
        Point p1, p2;
        // Find the other two vertices (neighbors)
        if (tri.v0 == node_idx) { p1 = points[tri.v1]; p2 = points[tri.v2]; }
        else if (tri.v1 == node_idx) { p1 = points[tri.v0]; p2 = points[tri.v2]; }
        else { p1 = points[tri.v0]; p2 = points[tri.v1]; }

        double ang = get_min_angle_deg(node_pos, p1, p2);
        if (ang < min_a) min_a = ang;
    }
    return min_a;
}

void relax_points(std::vector<Point>& points, const std::vector<Triangle>& mesh, double min_safe_x, double min_safe_y, double max_safe_x, double max_safe_y) {
    int n = points.size();
    
    // Accumulators for Area-Weighted Centroids
    std::vector<double> wx(n, 0.0), wy(n, 0.0);
    std::vector<double> w_sum(n, 0.0);
    
    // Build adjacency map for conditional check
    std::vector<std::vector<int>> point_to_tris(n);

    for (size_t k = 0; k < mesh.size(); ++k) {
        const auto& t = mesh[k];
        const Point& p0 = points[t.v0];
        const Point& p1 = points[t.v1];
        const Point& p2 = points[t.v2];

        // 1. Calculate Triangle Centroid
        double cx = (p0.x + p1.x + p2.x) / 3.0;
        double cy = (p0.y + p1.y + p2.y) / 3.0;

        // 2. Calculate Triangle Area
        // Area = 0.5 * |(x1-x0)(y2-y0) - (y1-y0)(x2-x0)|
        double area = 0.5 * std::abs((p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x));

        // 3. Accumulate weighted centroid for all vertices of this triangle
        wx[t.v0] += area * cx; wy[t.v0] += area * cy; w_sum[t.v0] += area;
        wx[t.v1] += area * cx; wy[t.v1] += area * cy; w_sum[t.v1] += area;
        wx[t.v2] += area * cx; wy[t.v2] += area * cy; w_sum[t.v2] += area;
        
        point_to_tris[t.v0].push_back(k);
        point_to_tris[t.v1].push_back(k);
        point_to_tris[t.v2].push_back(k);
    }

    // Define candidate relaxation factors to test per point
    std::vector<double> candidate_factors = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};

    for (int i=0; i<n; ++i) {
        if (w_sum[i] < 1e-12) continue;
        
        // Update if in valid safe zone (ignoring deep ghost layers)
        if (points[i].x > min_safe_x && points[i].x < max_safe_x && points[i].y > min_safe_y && points[i].y < max_safe_y) {
            
            // Target is the area-weighted average of surrounding triangle centroids
            double tx = wx[i] / w_sum[i]; 
            double ty = wy[i] / w_sum[i];
            
            // Calculate the full vector to the centroid
            double full_dx = tx - points[i].x;
            double full_dy = ty - points[i].y;

            double current_min_angle = calculate_local_min_angle(i, points[i], points, mesh, point_to_tris[i]);
            
            Point best_candidate = points[i];
            double best_angle = current_min_angle;

            // Test each relaxation factor
            for (double factor : candidate_factors) {
                double dx = full_dx * factor;
                double dy = full_dy * factor;

                // TEST: Boundary Projection
                apply_boundary_constraint(points[i], dx, dy); // Note: points[i] is passed by ref but not modified here effectively for logic check, wait... 
                // apply_boundary_constraint modifies the point passed to it if it snaps. 
                // We need to be careful not to modify points[i] permanently yet.
                // Let's create a temp point for the constraint check.
                Point temp_p = points[i];
                apply_boundary_constraint(temp_p, dx, dy); 
                
                Point candidate = points[i];
                // Apply the (potentially constrained) movement
                // Note: apply_boundary_constraint modifies 'p' to snap it, and 'dx/dy' to zero out perpendicular movement.
                // So we should use the snapped position + the constrained delta.
                candidate.x = temp_p.x + dx;
                candidate.y = temp_p.y + dy;

                double candidate_angle = calculate_local_min_angle(i, candidate, points, mesh, point_to_tris[i]);

                // Optimization Logic:
                // We prefer the move if it improves the angle.
                // If multiple factors improve it, we pick the one with the best angle.
                if (candidate_angle > best_angle) {
                    best_angle = candidate_angle;
                    best_candidate = candidate;
                  //   std::cout << "Point " << i << " improved from " << current_min_angle << " to " << candidate_angle 
                  //             << " using factor " << factor << "\n";
                }
            }

            // Apply the best move found (if any improved the state, or if we want to enforce a move)
            // Here we only move if we found a better state or if the current state is valid and the new state is also valid (smoothing).
            // But strictly following "best angle" is safest.
            
            // However, if ALL moves make it worse, we should probably stay put (which is covered since best_candidate starts as points[i]).
            // BUT, if the current angle is very good, and a move makes it slightly worse but still good, standard Lloyd would move.
            // This "Hill Climbing" approach might get stuck. 
            // Let's stick to: "Pick the factor that results in the MAX local min angle".
            
            points[i] = best_candidate;
        }
    }
}

// Helper for boundary check
bool is_on_boundary(const Point& p) {
    return p.x < EPSILON || p.x > 1.0-EPSILON || p.y < EPSILON || p.y > 1.0-EPSILON;
}

// Helper to create a point with a stable ID based on initial position
Point create_stable_point(double x, double y) {
    uint64_t h = 0;
    hash_combine(h, x);
    hash_combine(h, y);
    return {x, y, h};
}

// Helper to remove duplicates from cloud
void remove_duplicates(std::vector<Point>& points) {
    if (points.empty()) return;

    // Sort to bring duplicates together
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        if (std::abs(a.x - b.x) > 1e-9) return a.x < b.x;
        return a.y < b.y;
    });

    std::vector<Point> unique_points;
    unique_points.push_back(points[0]);

    for (size_t i = 1; i < points.size(); ++i) {
        const Point& prev = unique_points.back();
        const Point& curr = points[i];
        double dist_sq = (prev.x - curr.x)*(prev.x - curr.x) + (prev.y - curr.y)*(prev.y - curr.y);
        
        if (dist_sq > 1e-12) { // Keep if distance > 1e-6
            unique_points.push_back(curr);
        }
    }
    points = unique_points;
}

// --- Main Processing ---
void process_tile(int tile_x, int tile_y) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   

    double tile_s = DOMAIN_SIZE / TILDE_DIM;
    double t_min_x = tile_x * tile_s; double t_max_x = (tile_x + 1) * tile_s;
    double t_min_y = tile_y * tile_s; double t_max_y = (tile_y + 1) * tile_s;
    
    // UNIFIED PADDING LOGIC
    // We generate a halo large enough to absorb the boundary effects of annealing.
    // The distortion from the boundary travels approx 1 edge per cycle.
    // We add +3 for safety 
    double pad = TARGET_SPACING * (ANNEAL_CYCLES + FINAL_SMOOTH_ITERS + 3);

    // Safety Check: Ensure the required halo doesn't exceed the tile size.
    // In a domain decomposition, needing a halo larger than the subdomain 
    // implies we need data from neighbors-of-neighbors, which is inefficient/complex.
    if (pad > tile_s/2.0) {
        std::cerr << "Error: Annealing cycles (" << ANNEAL_CYCLES << ") require a halo of " 
                  << pad << ", which exceeds the tile size (" << tile_s << ").\n"
                  << "Reduce ANNEAL_CYCLES or increase tile size (by having fewer tiles or more points per tile).\n";
        std::exit(EXIT_FAILURE);
    }    

    double search_min_x = t_min_x - pad; double search_max_x = t_max_x + pad;
    double search_min_y = t_min_y - pad; double search_max_y = t_max_y + pad;

    std::vector<Point> cloud;

    // 1. EXPLICIT CORNERS
    // Add corners if they are in the search box.
    // Use type 5 for corners to distinguish IDs.
    
    // (0,0)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON && 
        search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        cloud.push_back(create_stable_point(0.0, 0.0));
    }
    // (1,0)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON && 
        search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        cloud.push_back(create_stable_point(DOMAIN_SIZE, 0.0));
    }
    // (0,1)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON && 
        search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        cloud.push_back(create_stable_point(0.0, DOMAIN_SIZE));
    }
    // (1,1)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON && 
        search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        cloud.push_back(create_stable_point(DOMAIN_SIZE, DOMAIN_SIZE));
    }

    // 2. EXPLICIT BOUNDARY GENERATION (Edges only)
    // Iterate 4 boundaries. Add points if they fall within search box.
    // FIX: Exclude corners (EPSILON checks) to avoid duplication with explicit corners above.
    
    // Left (x=0)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON) {
        int min_i = floor(search_min_y / TARGET_SPACING);
        int max_i = ceil(search_max_y / TARGET_SPACING);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_SPACING;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(0.0, y));
        }
    }
    // Right (x=1)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_y / TARGET_SPACING);
        int max_i = ceil(search_max_y / TARGET_SPACING);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_SPACING;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(DOMAIN_SIZE, y));
        }
    }
    // Bottom (y=0)
    if (search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        int min_i = floor(search_min_x / TARGET_SPACING);
        int max_i = ceil(search_max_x / TARGET_SPACING);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_SPACING;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(x, 0.0));
        }
    }
    // Top (y=1)
    if (search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_x / TARGET_SPACING);
        int max_i = ceil(search_max_x / TARGET_SPACING);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_SPACING;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(x, DOMAIN_SIZE));
        }
    }

    // 3. INTERIOR GENERATION (with Exclusion)
    int min_ix = floor(search_min_x / TARGET_SPACING);
    int max_ix = ceil(search_max_x / TARGET_SPACING);
    double hex_y_spacing = TARGET_SPACING * 0.8660254;
    int min_iy = floor(search_min_y / hex_y_spacing);
    int max_iy = ceil(search_max_y / hex_y_spacing);
    
    // Distance from wall to reject interior points
    // FIX: Must be larger than max jitter to prevent collision with boundary
    // Max jitter is START_JITTER * TARGET_SPACING. We add a safety margin.
    double exclusion = TARGET_SPACING * (START_JITTER + 0.1); 

    for (int iy = min_iy; iy <= max_iy; ++iy) {
        double cy = iy * hex_y_spacing;
        double row_offset = (iy % 2 != 0) ? (TARGET_SPACING * 0.5) : 0.0;
        for (int ix = min_ix; ix <= max_ix; ++ix) {
            double cx = ix * TARGET_SPACING + row_offset;
            
            // Rejection Sampling for Exclusion Zone
            if (cx < exclusion) continue;
            if (cx > DOMAIN_SIZE - exclusion) continue;
            if (cy < exclusion) continue;
            if (cy > DOMAIN_SIZE - exclusion) continue;

            // Only add point if strictly inside bounding box (handled by loop indices mostly)
            cloud.push_back(create_stable_point(cx, cy));
        }
    }

    // FIX: Remove any accidental duplicates (e.g. from corner/edge overlaps or precision issues)
    // This prevents Bowyer-Watson from exploding.
    remove_duplicates(cloud);

    // 4. ANNEALING
    // We relax all points that are strictly inside the generated cloud.
    // The outer hull acts as a fixed boundary condition.
    // FIX: We must freeze the outer rim of the generated cloud. 
    // If we relax the very edge, it collapses inward due to lack of outer neighbors.
    // This collapse propagates errors inward.
    // We freeze a strip of width ~ 1.5 * spacing.
    double frozen_margin = TARGET_SPACING * 1.5;

    double s_min_x = search_min_x + frozen_margin; 
    double s_max_x = search_max_x - frozen_margin;
    double s_min_y = search_min_y + frozen_margin; 
    double s_max_y = search_max_y - frozen_margin;

    std::vector<Triangle> mesh;
    double current_jitter = START_JITTER;

    for (int cycle = 0; cycle < ANNEAL_CYCLES; ++cycle) {
        apply_jitter(cloud, current_jitter, cycle);
        mesh = triangulation(cloud);
        relax_points(cloud, mesh, s_min_x, s_min_y, s_max_x, s_max_y);
    }
    // Final Polish
    for(int k=0; k<FINAL_SMOOTH_ITERS; ++k) {
        mesh = triangulation(cloud);
        relax_points(cloud, mesh, s_min_x, s_min_y, s_max_x, s_max_y);
    }
    
    // Final Emit
    mesh = triangulation(cloud);
    
    // NEW: Write full triangulation (including ghosts) to disk for visualization
    {
        std::string filename = "tile_" + std::to_string(tile_x) + "_" + std::to_string(tile_y) + ".dat";
        std::ofstream outfile(filename);
        if (outfile.is_open()) {
            // Write Points
            outfile << cloud.size() << "\n";
            for (const auto& p : cloud) {
                outfile << p.x << " " << p.y << "\n";
            }
            // Write Triangles
            outfile << mesh.size() << "\n";
            for (const auto& t : mesh) {
                outfile << t.v0 << " " << t.v1 << " " << t.v2 << "\n";
            }
            outfile.close();
        }
    }

    int count = 0;
    int boundary_nodes = 0;
    
    for (const auto& tri : mesh) {
        const Point& p0 = cloud[tri.v0];
        const Point& p1 = cloud[tri.v1];
        const Point& p2 = cloud[tri.v2];
        double cx = (p0.x+p1.x+p2.x)/3.0; 
        double cy = (p0.y+p1.y+p2.y)/3.0;

        if (cx >= t_min_x && cx < t_max_x && cy >= t_min_y && cy < t_max_y) {
            count++;
            if (is_on_boundary(p0)) boundary_nodes++;
            if (is_on_boundary(p1)) boundary_nodes++;
            if (is_on_boundary(p2)) boundary_nodes++;
        }
    }
     
    std::cout << "Rank " << rank << " Tile [" << tile_x << "," << tile_y << "] : " << count << " elements. "
              << "(Touched Boundary Nodes: " << boundary_nodes << ")\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Generating Annealed Mesh with Explicit Boundaries...\n";
        std::cout << "Running on " << size << " MPI ranks for " << TILDE_DIM << "x" << TILDE_DIM << " tiles.\n";
    }

    // Distribute tiles cyclically among ranks
    for (int y = 0; y < TILDE_DIM; ++y) {
        for (int x = 0; x < TILDE_DIM; ++x) {
            int global_id = y * TILDE_DIM + x;
            
            if (global_id % size == rank) {
                process_tile(x, y);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
