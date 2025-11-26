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
#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <map>

#if !defined(ANSI_DECLARATORS)
  #define ANSI_DECLARATORS
#endif
#include <triangle.h>

// =========================================================
// JITTER + LLOYD SMOOTH MESH GENERATOR FOR [0,DOMAIN_SIZE]^2
// =========================================================
// 
// Strategy:
// 1. Boundary Gen: Create points explicitly on [0,DOMAIN_SIZE]^2 edges.
// 2. Interior Gen: Create Hex Grid points, REJECTING those near edges.
// 3. Annealing: Jitter -> Smooth loop.
// 4. Constraint: Boundary nodes only move tangentially.
// =========================================================

const int TILE_DIM = 2; 
const double DOMAIN_SIZE = 1.0;
const double TARGET_EDGE_LENGTH = 0.0075; 

// We need these to be relative to edge length to support very fine meshes (e.g. 10^-6 spacing)
const double TOL_LEN = TARGET_EDGE_LENGTH * 1e-4;
const double TOL_LEN_SQ = TOL_LEN * TOL_LEN;
const double TOL_AREA = TOL_LEN_SQ * 1e-2;

const double EPSILON = 1e-13;     
const double START_JITTER = 0.30;
double MIN_ANGLE_THRESHOLD = 30.0;

const int ANNEAL_ITERS = 3; 
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
        // Hash based on STABLE ID + iters Offset
        // This ensures that even if the point moves, the jitter sequence is deterministic
        uint64_t h = points[i].id;

        // Mix in the iters/seed offset to ensure different jitter each step
        h ^= seed_offset + 0x9e3779b9 + (h << 6) + (h >> 2);

        RngState rng = {h};

        double jx = (next_double(rng) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;
        double jy = (next_double(rng) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;

        // CRITICAL: Boundary nodes effectively ignore perpendicular jitter here
        apply_boundary_constraint(points[i], jx, jy);

        points[i].x += jx;
        points[i].y += jy;
    }
}

// Wrapper for Triangle library - https://www.cs.cmu.edu/~quake/triangle.html
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

    if (ab < TOL_LEN || bc < TOL_LEN || ca < TOL_LEN) return 0.0;

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
        if (w_sum[i] < TOL_AREA) continue;
        
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

    // FIX: Use strict lexicographical sort. 
    // Using tolerance in sort comparator violates Strict Weak Ordering and causes non-deterministic behavior.
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.id < b.id; // Tie-breaker for absolute stability
    });

    std::vector<Point> unique_points;
    unique_points.reserve(points.size());
    unique_points.push_back(points[0]);

    for (size_t i = 1; i < points.size(); ++i) {
        const Point& prev = unique_points.back();
        const Point& curr = points[i];
        
        // Check distance with tolerance
        double dist_sq = (prev.x - curr.x)*(prev.x - curr.x) + (prev.y - curr.y)*(prev.y - curr.y);
        
        // Only keep if distance is physically significant relative to mesh size
        if (dist_sq > TOL_LEN_SQ) {
            unique_points.push_back(curr);
        }
    }
    points = unique_points;
}

// Helper to determine which rank owns a point based on spatial location
int get_owner_rank(double x, double y, int size) {
    double tile_s = DOMAIN_SIZE / TILE_DIM;
    int tx = std::floor(x / tile_s);
    int ty = std::floor(y / tile_s);
    
    // Clamp to handle numerical noise at upper boundaries
    if (tx < 0) tx = 0; 
    if (tx >= TILE_DIM) tx = TILE_DIM - 1;
    if (ty < 0) ty = 0; 
    if (ty >= TILE_DIM) ty = TILE_DIM - 1;

    int global_tile_id = ty * TILE_DIM + tx;
    return global_tile_id % size;
}

// --- Main Processing ---
void process_tile(int tile_x, int tile_y, 
                  std::vector<Point>& acc_cloud, 
                  std::vector<Triangle>& acc_mesh, 
                  std::map<uint64_t, int>& id_to_idx) {

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);                     

    double tile_s = DOMAIN_SIZE / TILE_DIM;
    double t_min_x = tile_x * tile_s; double t_max_x = (tile_x + 1) * tile_s;
    double t_min_y = tile_y * tile_s; double t_max_y = (tile_y + 1) * tile_s;
    
    // UNIFIED PADDING LOGIC
    // We generate a halo large enough to absorb the boundary effects of annealing.
    // The distortion from the boundary travels approx 1 edge per iter.
    // We add +3 for safety 
    double pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + FINAL_SMOOTH_ITERS + 3);

    // Safety Check: Ensure the required halo doesn't exceed the tile size.
    // In a domain decomposition, needing a halo larger than the subdomain 
    // implies we need data from neighbors-of-neighbors, which is inefficient/complex.
    if (pad > tile_s/2.0) {
        std::cerr << "Error: Annealing iters (" << ANNEAL_ITERS << ") require a halo of " 
                  << pad << ", which exceeds the tile size (" << tile_s << ").\n"
                  << "Reduce ANNEAL_ITERS or increase tile size (by having fewer tiles or more points per tile).\n";
        std::exit(EXIT_FAILURE);
    }    

    double search_min_x = t_min_x - pad; double search_max_x = t_max_x + pad;
    double search_min_y = t_min_y - pad; double search_max_y = t_max_y + pad;

    std::vector<Point> cloud;

    // 1. EXPLICIT CORNERS
    // Add corners if they are in the search box.
    
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
        int min_i = floor(search_min_y / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_y / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_EDGE_LENGTH;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(0.0, y));
        }
    }
    // Right (x=1)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_y / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_y / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_EDGE_LENGTH;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(DOMAIN_SIZE, y));
        }
    }
    // Bottom (y=0)
    if (search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        int min_i = floor(search_min_x / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_x / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_EDGE_LENGTH;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(x, 0.0));
        }
    }
    // Top (y=1)
    if (search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_x / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_x / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_EDGE_LENGTH;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) cloud.push_back(create_stable_point(x, DOMAIN_SIZE));
        }
    }

    // 3. INTERIOR GENERATION (with Exclusion)
    int min_ix = floor(search_min_x / TARGET_EDGE_LENGTH);
    int max_ix = ceil(search_max_x / TARGET_EDGE_LENGTH);
    double hex_y_spacing = TARGET_EDGE_LENGTH * 0.8660254;
    int min_iy = floor(search_min_y / hex_y_spacing);
    int max_iy = ceil(search_max_y / hex_y_spacing);
    
    // Distance from wall to reject interior points
    // FIX: Must be larger than max jitter to prevent collision with boundary
    // Max jitter is START_JITTER * TARGET_EDGE_LENGTH. 
    // We need a larger safety margin to prevent slivers.
    // 0.6 ensures that even with 0.3 jitter, the closest approach is ~0.3L, 
    // which results in an aspect ratio of ~3:1 (acceptable).
    double exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.35); 

    for (int iy = min_iy; iy <= max_iy; ++iy) {
        double cy = iy * hex_y_spacing;
        double row_offset = (iy % 2 != 0) ? (TARGET_EDGE_LENGTH * 0.5) : 0.0;
        for (int ix = min_ix; ix <= max_ix; ++ix) {
            double cx = ix * TARGET_EDGE_LENGTH + row_offset;
            
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
    double frozen_margin = TARGET_EDGE_LENGTH * 1.5;

    double s_min_x = search_min_x + frozen_margin; 
    double s_max_x = search_max_x - frozen_margin;
    double s_min_y = search_min_y + frozen_margin; 
    double s_max_y = search_max_y - frozen_margin;

    std::vector<Triangle> mesh;
    double current_jitter = START_JITTER;

    for (int iter = 0; iter < ANNEAL_ITERS; ++iter) {
        apply_jitter(cloud, current_jitter, iter);
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
    
    // MERGE INTO ACCUMULATOR
    // We only keep triangles that are geometrically "owned" by this tile to avoid duplication
    // when a rank owns adjacent tiles.
    for (const auto& tri : mesh) {
        const Point& p0 = cloud[tri.v0];
        const Point& p1 = cloud[tri.v1];
        const Point& p2 = cloud[tri.v2];

        // Robust Ownership Rule:
        // A triangle is owned by the rank that owns the triangle's "lowest" vertex.
        // We define "lowest" using the stable ID to ensure all ranks agree.
        // If multiple vertices are on the same rank, that rank definitely owns it.
        // If vertices are on different ranks, the one with the smallest ID decides.
        
        uint64_t min_id = std::min({p0.id, p1.id, p2.id});
        
        // Find which point has this ID
        const Point* min_p = &p0;
        if (p1.id == min_id) min_p = &p1;
        if (p2.id == min_id) min_p = &p2;

        // Check if WE own this determining vertex
        // Note: We use the global 'get_owner_rank' function which is purely spatial and deterministic.
        int owner = get_owner_rank(min_p->x, min_p->y, comm_size);

        if (owner == comm_rank) {
            Triangle new_t;
            Point* pts[3] = { (Point*)&p0, (Point*)&p1, (Point*)&p2 };
            int*   v_idx[3] = { &new_t.v0, &new_t.v1, &new_t.v2 };

            // This builds a mapping between point id and its index in the accumulated cloud
            for(int k=0; k<3; ++k) {
                uint64_t pid = pts[k]->id;
                if (id_to_idx.find(pid) == id_to_idx.end()) {
                    int new_idx = acc_cloud.size();
                    acc_cloud.push_back(*pts[k]);
                    id_to_idx[pid] = new_idx;
                }
                *v_idx[k] = id_to_idx[pid];
            }
            acc_mesh.push_back(new_t);
        }
    }
}

DM CreateDistributedDM(const std::vector<Point>& cloud, const std::vector<Triangle>& mesh) {
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    PetscInt num_local_pts = cloud.size();
    std::vector<PetscInt> global_ids(num_local_pts, -1);
    PetscInt num_owned = 0;

    // 1. Identify Owned Vertices
    for (int i = 0; i < num_local_pts; ++i) {
        if (get_owner_rank(cloud[i].x, cloud[i].y, comm_size) == comm_rank) {
            num_owned++;
        }
    }

    // 2. Calculate Global Offsets - petscint to ensure large counts work
    PetscInt start_id = 0;
    MPI_Exscan(&num_owned, &start_id, 1, MPIU_INT, MPI_SUM, MPI_COMM_WORLD);

    // 3. Assign Global IDs to Owned Vertices
    PetscInt current_id = start_id;
    for (int i = 0; i < num_local_pts; ++i) {
        if (get_owner_rank(cloud[i].x, cloud[i].y, comm_size) == comm_rank) {
            global_ids[i] = current_id++;
        }
    }

    // 4. Resolve Ghost IDs
    // We need to ask the owners for the Global IDs of our ghost points.
    std::vector<std::vector<uint64_t>> send_ids(comm_size);
    std::vector<std::vector<int>>      send_req_indices(comm_size); // Map back to local index

    for (int i = 0; i < num_local_pts; ++i) {
        if (global_ids[i] == -1) {
            int owner = get_owner_rank(cloud[i].x, cloud[i].y, comm_size);
            send_ids[owner].push_back(cloud[i].id);
            send_req_indices[owner].push_back(i);
        }
    }

    // Exchange counts
    std::vector<int> send_counts(comm_size), recv_counts(comm_size);
    for(int r=0; r<comm_size; ++r) send_counts[r] = send_ids[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Prepare Receive Buffers (Requests from others)
    std::vector<std::vector<uint64_t>> recv_ids(comm_size);
    std::vector<std::vector<PetscInt>> send_answers(comm_size);
    
    // Post Receives
    for(int r=0; r<comm_size; ++r) {
        if(recv_counts[r] > 0) {
            recv_ids[r].resize(recv_counts[r]);
        }
    }

    // Simple Point-to-Point Exchange (Blocking for simplicity, use Irecv/Isend in prod)
    for(int r=0; r<comm_size; ++r) {
        if (r == comm_rank) continue;
        if (send_counts[r] > 0) {
            MPI_Send(send_ids[r].data(), send_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, MPI_COMM_WORLD);
        }
    }
    
    // Process Incoming Requests
    // We need a map for fast lookup of OUR owned points
    std::map<uint64_t, PetscInt> my_owned_map;
    for(int i=0; i<num_local_pts; ++i) {
        if (get_owner_rank(cloud[i].x, cloud[i].y, comm_size) == comm_rank) {
            my_owned_map[cloud[i].id] = global_ids[i];
        }
    }

    for(int r=0; r<comm_size; ++r) {
        if (r == comm_rank) continue;
        if (recv_counts[r] > 0) {
            MPI_Status status;
            MPI_Recv(recv_ids[r].data(), recv_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, MPI_COMM_WORLD, &status);
            
            send_answers[r].resize(recv_counts[r]);
            for(int k=0; k<recv_counts[r]; ++k) {
                send_answers[r][k] = my_owned_map[recv_ids[r][k]];
            }
            // Send answers back
            MPI_Send(send_answers[r].data(), recv_counts[r], MPIU_INT, r, 101, MPI_COMM_WORLD);
        }
    }

    // Receive Answers
    for(int r=0; r<comm_size; ++r) {
        if (r == comm_rank) continue;
        if (send_counts[r] > 0) {
            std::vector<PetscInt> answers(send_counts[r]);
            MPI_Status status;
            MPI_Recv(answers.data(), send_counts[r], MPIU_INT, r, 101, MPI_COMM_WORLD, &status);
            
            for(int k=0; k<send_counts[r]; ++k) {
                int local_idx = send_req_indices[r][k];
                global_ids[local_idx] = answers[k];
            }
        }
    }

    // 5. Build DMPlex
    PetscInt num_cells = mesh.size();
    std::vector<PetscInt> cells(num_cells * 3);
    for(int i=0; i<num_cells; ++i) {
        cells[i*3 + 0] = global_ids[mesh[i].v0];
        cells[i*3 + 1] = global_ids[mesh[i].v1];
        cells[i*3 + 2] = global_ids[mesh[i].v2];
    }

    // Prepare coordinates for DMPlexCreateFromCellListParallelPetsc
    // FIX: Only pass OWNED points to PETSc. 
    // acc_cloud contains ghosts (vertices of owned triangles that are owned by neighbors).
    // PETSc expects 'numPoints' to be the count of locally owned vertices, 
    // and 'coords' to be the coordinates of those specific vertices in Global ID order.
    std::vector<PetscReal> owned_coords;
    owned_coords.reserve(num_owned * 2);
    
    for(int i=0; i<num_local_pts; ++i) {
        if (get_owner_rank(cloud[i].x, cloud[i].y, comm_size) == comm_rank) {
            owned_coords.push_back(cloud[i].x);
            owned_coords.push_back(cloud[i].y);
        }
    }

    DM dm;
    // Build the DM
    PetscInt two = 2;
    PetscInt three = 3;
    // Pass num_owned instead of num_local_pts, and owned_coords instead of coords
    (void*)DMPlexCreateFromCellListParallelPetsc(MPI_COMM_WORLD, two, num_cells, num_owned, PETSC_DECIDE, \
         three, PETSC_TRUE, cells.data(), two, owned_coords.data(), NULL, NULL, &dm);
    return dm;
}

int main(int argc, char** argv) {

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_rank == 0) {
        std::cout << "Generating Unstructured Mesh of 2D box...\n";
        std::cout << "Running on " << comm_size << " MPI ranks for " << TILE_DIM << "x" << TILE_DIM << " tiles.\n";
    }

    std::vector<Point> rank_cloud;
    std::vector<Triangle> rank_mesh;
    std::map<uint64_t, int> id_map;

    // Distribute tiles cyclically among ranks
    for (int y = 0; y < TILE_DIM; ++y) {
        for (int x = 0; x < TILE_DIM; ++x) {
            int global_id = y * TILE_DIM + x;
            
            if (global_id % comm_size == comm_rank) {
                process_tile(x, y, rank_cloud, rank_mesh, id_map);
            }
        }
    }

    std::cout << "Rank " << comm_rank << " generated " << rank_mesh.size() << " triangles.\n";

    DM dm = CreateDistributedDM(rank_cloud, rank_mesh);
    PetscCall(PetscObjectSetName((PetscObject)dm, "Mesh"));
    PetscViewer viewer;
    // Can view this in paraview with:
    // /home/sdargavi/projects/dependencies/petsc_main/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5
    // then using the XDMF reader
    // paraview box_mesh.xmf
    PetscCall(PetscViewerHDF5Open(MPI_COMM_WORLD, "box_mesh.h5", FILE_MODE_WRITE, &viewer));
    PetscCall(DMView(dm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    rank_cloud.clear();
    rank_mesh.clear();
    id_map.clear();

    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}
