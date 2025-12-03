#include "box_2D_gen_unstruc_mesh.h"
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
#include <set>

#if !defined(ANSI_DECLARATORS)
  #define ANSI_DECLARATORS
#endif
#include <triangle.h>

// =========================================================
// Unstructured mesh generator for 2D box
// Builds a PETSc DMPlex
// =========================================================
// 
// Strategy:
// 1. Boundary Gen: Create points explicitly on [0,DOMAIN_SIZE]^2 boundaries.
// 2. Interior Gen: Create random points inside small squares, REJECTING those near boundaries.
// 3. Iterate: jitter -> triangulate -> smooth loop.
// 4. Constraint: Boundary nodes only move tangentially.
// =========================================================

static int TILE_DIM_X = -1;
static int TILE_DIM_Y = -1;
const double DOMAIN_SIZE = 1.0;
static double TARGET_EDGE_LENGTH = 0.0025; 

static double TOL_LEN;
static double TOL_LEN_SQ;
static double TOL_VOLUME;

const double EPSILON = 1e-13;     
const double START_JITTER = 0.30;
// This is the goal, it isn't necessarily achieved
double MIN_ANGLE_THRESHOLD = 30.0;

// Jitter + smooth iterations first
const int ANNEAL_ITERS = 3; 
// Then just smooth iterations
const int FINAL_SMOOTH_ITERS = 2;

// ~~~~~~~~~~~~~~~~~

struct Point {
    double x, y; // coordinates
    uint64_t unique_hash_id = 0; // unique id based on hashing coordinates
    int valence = 0; // <--- Store connectivity here
    int fixed_owner = -1; // <--- NEW: Explicit ownership override
};
struct Triangle {
    int v0, v1, v2;
};
struct Edge {
    int v0, v1;
    bool operator==(const Edge& o) const { return (v0 == o.v0 && v1 == o.v1) || (v0 == o.v1 && v1 == o.v0); }
};
// Stateless random
struct RngState { uint64_t s; };

// ~~~~~~~~~~~~~~~~~

// Taken from PETSc in src/dm/impls/plex/generators/triangle/trigenerate.c
// to interface with Triangle library
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

// ~~~~~~~~~~~~~~~~~

// Helper to determine which rank owns a point based on spatial location
static int get_owner_rank(const Point& p, int size) {

   // Use fixed owner if it is available
   if (p.fixed_owner != -1) return p.fixed_owner;

    double tile_s_x = DOMAIN_SIZE / TILE_DIM_X;
    double tile_s_y = DOMAIN_SIZE / TILE_DIM_Y;
    int tx = std::floor(p.x / tile_s_x);
    int ty = std::floor(p.y / tile_s_y);
    
    // Clamp to handle numerical noise at upper boundaries
    if (tx < 0) tx = 0; 
    if (tx >= TILE_DIM_X) tx = TILE_DIM_X - 1;
    if (ty < 0) ty = 0; 
    if (ty >= TILE_DIM_Y) ty = TILE_DIM_Y - 1;

    int global_tile_id = ty * TILE_DIM_X + tx;
    // With 1 tile per rank, the tile ID is the rank
    return global_tile_id;
}

// NEW: Deterministically resolve ownership of boundary nodes
static void ResolveBoundaryOwnership(MPI_Comm comm, std::vector<Point>& points) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    struct Claim {
        uint64_t id;
        int geo_rank;
        double x, y; // Include coordinates for synchronization
    };

    // 1. Identify points near boundary to exchange
    // We use a safe margin to catch any point that might be ambiguous
    double margin = TARGET_EDGE_LENGTH * 2.0; 
    
    std::vector<std::vector<Claim>> send_buffers(size);
    std::set<uint64_t> involved_ids; // Only resolve points near boundaries
    
    // Simple neighbor discovery: check all ranks (for small scale) or use grid logic
    // Here we use the grid logic to only send to actual neighbors
    int tx = rank % TILE_DIM_X;
    int ty = rank / TILE_DIM_X;
    
    for (auto& p : points) {
        // Check if point is near a tile boundary
        double t_min_x = tx * (DOMAIN_SIZE / TILE_DIM_X);
        double t_max_x = (tx + 1) * (DOMAIN_SIZE / TILE_DIM_X);
        double t_min_y = ty * (DOMAIN_SIZE / TILE_DIM_Y);
        double t_max_y = (ty + 1) * (DOMAIN_SIZE / TILE_DIM_Y);

        // Strict bounding box check:
        // A rank should only claim a point if it is within its tile (plus margin).
        if (p.x < t_min_x - margin || p.x > t_max_x + margin ||
            p.y < t_min_y - margin || p.y > t_max_y + margin) {
            continue;
        }

        bool near_x = (std::abs(p.x - t_min_x) < margin) || (std::abs(p.x - t_max_x) < margin);
        bool near_y = (std::abs(p.y - t_min_y) < margin) || (std::abs(p.y - t_max_y) < margin);

        if (near_x || near_y) {
            involved_ids.insert(p.unique_hash_id);
            // Let's reset the fixed owner in case we have moved the point again
            p.fixed_owner = -1;
            int my_geo_rank = get_owner_rank(p, size); // fixed_owner is -1 here

            // It's a boundary candidate. Send to neighbors.
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = tx + dx;
                    int ny = ty + dy;
                    if (nx >= 0 && nx < TILE_DIM_X && ny >= 0 && ny < TILE_DIM_Y) {
                        int n_rank = ny * TILE_DIM_X + nx;
                        send_buffers[n_rank].push_back({p.unique_hash_id, my_geo_rank, p.x, p.y});
                    }
                }
            }
        }
    }

    // 2. Exchange Candidates
    std::vector<int> send_counts(size), recv_counts(size);
    for(int r=0; r<size; ++r) send_counts[r] = send_buffers[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<std::vector<Claim>> recv_buffers(size);
    std::vector<MPI_Request> requests;

    for(int r=0; r<size; ++r) {
        if (recv_counts[r] > 0) {
            recv_buffers[r].resize(recv_counts[r]);
            MPI_Request req;
            MPI_Irecv(recv_buffers[r].data(), recv_counts[r] * sizeof(Claim), MPI_BYTE, r, 999, comm, &req);
            requests.push_back(req);
        }
    }
    for(int r=0; r<size; ++r) {
        if (send_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_buffers[r].data(), send_counts[r] * sizeof(Claim), MPI_BYTE, r, 999, comm, &req);
            requests.push_back(req);
        }
    }
    if (!requests.empty()) MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // 3. Resolve Ownership
    // We track:
    //  - The set of geometric ranks calculated by all claimers (to detect disagreement)
    //  - The lowest rank that claimed the point (to pick a winner)
    //  - The coordinates associated with that lowest rank (to sync geometry)
    struct ResolutionData {
        std::set<int> observed_geo_ranks;
        int best_rank = 999999;
        double best_x = 0, best_y = 0;
    };
    std::map<uint64_t, ResolutionData> resolution_map;
    
    // Initialize with self ONLY for involved points
    for(const auto& p : points) {
        if (involved_ids.count(p.unique_hash_id)) {
            int my_geo_rank = get_owner_rank(p, size);
            ResolutionData& data = resolution_map[p.unique_hash_id];
            data.observed_geo_ranks.insert(my_geo_rank);
            data.best_rank = rank;
            data.best_x = p.x;
            data.best_y = p.y;
        }
    }

    // Update with neighbors
    for(int r=0; r<size; ++r) {
        for(const auto& claim : recv_buffers[r]) {
            involved_ids.insert(claim.id); // Mark as involved if a neighbor claims it
            ResolutionData& data = resolution_map[claim.id];
            
            data.observed_geo_ranks.insert(claim.geo_rank);
            
            // Update best rank (lowest wins) and its coordinates
            if (r < data.best_rank) {
                data.best_rank = r;
                data.best_x = claim.x;
                data.best_y = claim.y;
            }
        }
    }

    // 4. Apply to points
    for(auto& p : points) {
        // Only process points involved in the boundary resolution
        if (involved_ids.count(p.unique_hash_id)) {
            const ResolutionData& data = resolution_map[p.unique_hash_id];

            // double old_x = p.x;
            // double old_y = p.y;
            
            // 4a. Synchronize Coordinates
            // Everyone adopts the coordinates of the 'best_rank' to ensure geometric consistency
            p.x = data.best_x;
            p.y = data.best_y;

            // 4b. Resolve Ownership Conflict
            // Check if there is disagreement on the geometric rank
            if (data.observed_geo_ranks.size() > 1) {
                
                // Disagreement exists! We must enforce a fixed owner.
                // Strategy: Lowest rank ID among those who claimed it wins.
                const std::set<int>& claims = data.observed_geo_ranks;
                if (!claims.empty()) {
                    int resolved_rank = *claims.begin(); // Lowest rank wins
                    
                    p.fixed_owner = resolved_rank;

                  //   // print out all of the ranks which disagree on ownership
                  //   std::cout << "[Rank " << rank << "] Point (" << std::setprecision(6) << old_x << ", " << old_y << " - new (" << p.x << ", " << p.y
                  //             << ") CONFLICT. Geo Ranks: ";
                  //   for(int g : data.observed_geo_ranks) std::cout << g << " ";
                  //   std::cout << ". Assigned to Rank " << resolved_rank << "\n";
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~

// Returns true if point is on a boundary.
// Modifies dx, dy to ensure movement is only tangential (sliding).
static bool apply_boundary_constraint(Point& p, double& dx, double& dy) {
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

// ~~~~~~~~~~~~~~~~~

static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static double next_double(RngState& state) {
    return (splitmix64(state.s) >> 11) * (1.0 / 9007199254740992.0);
}

// Helper for hashing
static void hash_combine(uint64_t& h, double v) {
    union { double d; uint64_t u; } c;
    c.d = v;
    h ^= c.u + 0x9e3779b9 + (h << 6) + (h >> 2);
}

// Helper to create a point with a unique hash ID based on initial position
static Point create_point_with_unique_hash_id(double x, double y) {
    uint64_t h = 0;
    hash_combine(h, x);
    hash_combine(h, y);
    return {x, y, h};
}

// ~~~~~~~~~~~~~~~~~

// Applies deterministic jitter
static void apply_jitter(std::vector<Point>& points, double amount, int seed_offset) {
    for (size_t i = 0; i < points.size(); ++i) {
        // Hash based on unique hash ID + iters Offset
        // This ensures that even if the point moves, the jitter sequence is deterministic
        uint64_t h = points[i].unique_hash_id;

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

// ~~~~~~~~~~~~~~~~~

// Wrapper for Triangle library - https://www.cs.cmu.edu/~quake/triangle.html
static std::vector<Triangle> triangulation(const std::vector<Point>& points) {
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
    std::vector<Triangle> triangles;
    triangles.reserve(numCells);

    for (int i = 0; i < numCells; ++i) {
        Triangle t;
        t.v0 = (int)out.trianglelist[i * 3 + 0];
        t.v1 = (int)out.trianglelist[i * 3 + 1];
        t.v2 = (int)out.trianglelist[i * 3 + 2];
        triangles.push_back(t);
    }

    delete[] in.pointlist;
    FiniOutput_Triangle(&out);

    return triangles;
}

// ~~~~~~~~~~~~~~~~~

// Helper: Calculate minimum angle (degrees) of a triangle
static double clamp_val(double v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); }

// Gets the smallest angle in a triangle in degrees
static double get_min_angle_deg(const Point& a, const Point& b, const Point& c) {
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

// ~~~~~~~~~~~~~~~~~

// Helper for relax_points to calculate local min angle
static double calculate_local_min_angle(int node_idx, const Point& node_pos, 
                                 const std::vector<Point>& points, 
                                 const std::vector<Triangle>& triangles, 
                                 const std::vector<int>& connected_tris) {
    double min_a = 180.0;
    for (int t_idx : connected_tris) {
        const auto& tri = triangles[t_idx];
        Point p1, p2;
        // Find the other two points (neighbors)
        if (tri.v0 == node_idx) { p1 = points[tri.v1]; p2 = points[tri.v2]; }
        else if (tri.v1 == node_idx) { p1 = points[tri.v0]; p2 = points[tri.v2]; }
        else { p1 = points[tri.v0]; p2 = points[tri.v1]; }

        double ang = get_min_angle_deg(node_pos, p1, p2);
        if (ang < min_a) min_a = ang;
    }
    return min_a;
}

// ~~~~~~~~~~~~~~~~~

// Lloyd-smoothing - only smooths points within a certain box
// that way we can lock the outermost points in the halo, so the halo 
// doesn't shrink inwards
static void relax_points(std::vector<Point>& points, const std::vector<Triangle>& triangles, double min_safe_x, double min_safe_y, double max_safe_x, double max_safe_y) {
    int n = points.size();
    
    // Accumulators for volume-Weighted Centroids
    std::vector<double> wx(n, 0.0), wy(n, 0.0);
    std::vector<double> w_sum(n, 0.0);
    
    // Build adjacency map for conditional check
    std::vector<std::vector<int>> point_to_tris(n);

    for (size_t k = 0; k < triangles.size(); ++k) {
        const auto& t = triangles[k];
        const Point& p0 = points[t.v0];
        const Point& p1 = points[t.v1];
        const Point& p2 = points[t.v2];

        // 1. Calculate Triangle Centroid
        double cx = (p0.x + p1.x + p2.x) / 3.0;
        double cy = (p0.y + p1.y + p2.y) / 3.0;

        // 2. Calculate Triangle volume
        // volume = 0.5 * |(x1-x0)(y2-y0) - (y1-y0)(x2-x0)|
        double volume = 0.5 * std::abs((p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x));

        // 3. Accumulate weighted centroid for all points of this triangle
        wx[t.v0] += volume * cx; wy[t.v0] += volume * cy; w_sum[t.v0] += volume;
        wx[t.v1] += volume * cx; wy[t.v1] += volume * cy; w_sum[t.v1] += volume;
        wx[t.v2] += volume * cx; wy[t.v2] += volume * cy; w_sum[t.v2] += volume;

        point_to_tris[t.v0].push_back(k);
        point_to_tris[t.v1].push_back(k);
        point_to_tris[t.v2].push_back(k);
    }

    // Define candidate relaxation factors to test per point
    std::vector<double> candidate_factors = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};

    for (int i=0; i<n; ++i) {
        if (w_sum[i] < TOL_VOLUME) continue;
        
        // Update if in valid safe zone (ignoring deep ghost layers)
        if (points[i].x > min_safe_x && points[i].x < max_safe_x && points[i].y > min_safe_y && points[i].y < max_safe_y) {
            
            // Target is the volume-weighted average of surrounding triangle centroids
            double tx = wx[i] / w_sum[i]; 
            double ty = wy[i] / w_sum[i];
            
            // Calculate the full vector to the centroid
            double full_dx = tx - points[i].x;
            double full_dy = ty - points[i].y;

            double current_min_angle = calculate_local_min_angle(i, points[i], points, triangles, point_to_tris[i]);
            
            Point best_candidate = points[i];
            double best_angle = current_min_angle;

            // Test each relaxation factor
            for (double factor : candidate_factors) {
                double dx = full_dx * factor;
                double dy = full_dy * factor;

                // TEST: Boundary Projection
                apply_boundary_constraint(points[i], dx, dy);
                // We need to be careful not to modify points[i] permanently yet.
                // Let's create a temp point for the constraint check.
                Point temp_p = points[i];
                apply_boundary_constraint(temp_p, dx, dy); 
                
                Point candidate = points[i];
                candidate.x = temp_p.x + dx;
                candidate.y = temp_p.y + dy;

                double candidate_angle = calculate_local_min_angle(i, candidate, points, triangles, point_to_tris[i]);

                // Optimization Logic:
                // We prefer the move if it improves the angle.
                // If multiple factors improve it, we pick the one with the best angle.
                if (candidate_angle > best_angle) {
                    best_angle = candidate_angle;
                    best_candidate = candidate;
                }
            }

            // Apply the best move found (if any improved the state, or if we want to enforce a move)
            // Here we only move if we found a better state or if the current state is valid and the new state is also valid (smoothing).
            // But strictly following "best angle" is safest.
            // However, if ALL moves make it worse, we should probably stay put (which is covered since best_candidate starts as points[i]).
            points[i] = best_candidate;
        }
    }
}

// ~~~~~~~~~~~~~~~~~

// Helper to remove duplicates from cloud
static void remove_duplicates(std::vector<Point>& points) {
    if (points.empty()) return;

    // Use strict lexicographical sort. 
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.unique_hash_id < b.unique_hash_id; // Tie-breaker for absolute stability
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

// ~~~~~~~~~~~~~~~~~

// Builds a tile, creates points and triangulates
static void process_tile(MPI_Comm comm, int tile_x, int tile_y, 
                  std::vector<Point>& points_on_owned_triangles_and_orphans, 
                  std::vector<Triangle>& triangles_owned) {

    // Local map to track points added to the accumulator for this tile
    std::map<uint64_t, int> id_to_idx;

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);      
    
    double tile_s_x = DOMAIN_SIZE / TILE_DIM_X;
    double tile_s_y = DOMAIN_SIZE / TILE_DIM_Y;
    double t_min_x = tile_x * tile_s_x; double t_max_x = (tile_x + 1) * tile_s_x;
    double t_min_y = tile_y * tile_s_y; double t_max_y = (tile_y + 1) * tile_s_y;
    
    // UNIFIED PADDING LOGIC
    // We generate a halo large enough to absorb the boundary effects of annealing.
    // The distortion from the boundary travels approx 1 edge per iter.
    // We add +8 for safety 
    double pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + FINAL_SMOOTH_ITERS + 8);

    // Safety Check: Ensure the required halo doesn't exceed the tile size.
    // In a domain decomposition, needing a halo larger than the subdomain 
    // implies we need data from neighbors-of-neighbors, which is inefficient/complex.
    double min_dim = std::min(tile_s_x, tile_s_y);
    if (pad > min_dim/2.0) {
        std::cerr << "Error: Annealing iters (" << ANNEAL_ITERS << ") require a halo of " 
                  << pad << ", which exceeds the tile size (" << min_dim << ").\n"
                  << "Reduce ANNEAL_ITERS or increase tile size (by having fewer tiles or more points per tile).\n";
        std::exit(EXIT_FAILURE);
    }    

    double search_min_x = t_min_x - pad; double search_max_x = t_max_x + pad;
    double search_min_y = t_min_y - pad; double search_max_y = t_max_y + pad;

    std::vector<Point> points_with_halos;

    // 1. EXPLICIT CORNERS
    // Add corners if they are in the search box.
    
    // (0,0)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON && 
        search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        points_with_halos.push_back(create_point_with_unique_hash_id(0.0, 0.0));
    }
    // (1,0)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON && 
        search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        points_with_halos.push_back(create_point_with_unique_hash_id(DOMAIN_SIZE, 0.0));
    }
    // (0,1)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON && 
        search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        points_with_halos.push_back(create_point_with_unique_hash_id(0.0, DOMAIN_SIZE));
    }
    // (1,1)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON && 
        search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        points_with_halos.push_back(create_point_with_unique_hash_id(DOMAIN_SIZE, DOMAIN_SIZE));
    }

    // 2. EXPLICIT BOUNDARY GENERATION (Edges only)
    // Iterate 4 boundaries. Add points if they fall within search box.
    // Exclude corners (EPSILON checks) to avoid duplication with explicit corners above.
    
    // Left (x=0)
    if (search_min_x <= EPSILON && search_max_x >= -EPSILON) {
        int min_i = floor(search_min_y / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_y / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_EDGE_LENGTH;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) points_with_halos.push_back(create_point_with_unique_hash_id(0.0, y));
        }
    }
    // Right (x=1)
    if (search_min_x <= DOMAIN_SIZE + EPSILON && search_max_x >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_y / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_y / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double y = i * TARGET_EDGE_LENGTH;
            if (y > EPSILON && y < DOMAIN_SIZE - EPSILON) points_with_halos.push_back(create_point_with_unique_hash_id(DOMAIN_SIZE, y));
        }
    }
    // Bottom (y=0)
    if (search_min_y <= EPSILON && search_max_y >= -EPSILON) {
        int min_i = floor(search_min_x / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_x / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_EDGE_LENGTH;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) points_with_halos.push_back(create_point_with_unique_hash_id(x, 0.0));
        }
    }
    // Top (y=1)
    if (search_min_y <= DOMAIN_SIZE + EPSILON && search_max_y >= DOMAIN_SIZE - EPSILON) {
        int min_i = floor(search_min_x / TARGET_EDGE_LENGTH);
        int max_i = ceil(search_max_x / TARGET_EDGE_LENGTH);
        for(int i=min_i; i<=max_i; ++i) {
            double x = i * TARGET_EDGE_LENGTH;
            if (x > EPSILON && x < DOMAIN_SIZE - EPSILON) points_with_halos.push_back(create_point_with_unique_hash_id(x, DOMAIN_SIZE));
        }
    }

    // 3. INTERIOR GENERATION (Stratified Sampling)
    // To match the density of a hex grid (optimal for triangles_owned), we scale the spacing.
    // Hex Density: 1 pt / (sqrt(3)/2 * L^2). Square Density: 1 pt / S^2.
    // S = L * sqrt(sqrt(3)/2) ~ 0.9306 * L
    double strat_spacing = TARGET_EDGE_LENGTH * 0.9306;

    int min_ix = floor(search_min_x / strat_spacing);
    int max_ix = ceil(search_max_x / strat_spacing);
    int min_iy = floor(search_min_y / strat_spacing);
    int max_iy = ceil(search_max_y / strat_spacing);
    
    // Distance from wall to reject interior points
    // Must be larger than max jitter to prevent collision with boundary
    // Max jitter is START_JITTER * TARGET_EDGE_LENGTH. 
    // We need a larger safety margin to prevent slivers.
    // 0.6 ensures that even with 0.3 jitter, the closest approach is ~0.3L, 
    // which results in an aspect ratio of ~3:1 (acceptable).
    double exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.35); 

    for (int iy = min_iy; iy <= max_iy; ++iy) {
        for (int ix = min_ix; ix <= max_ix; ++ix) {
            // Deterministic RNG based on global grid index
            uint64_t h = 0;
            hash_combine(h, (double)ix);
            hash_combine(h, (double)iy);
            RngState rng = {h};

            // Random position within the cell [0, 1)
            double r1 = next_double(rng);
            double r2 = next_double(rng);

            double cx = (ix + r1) * strat_spacing;
            double cy = (iy + r2) * strat_spacing;
            
            // Rejection Sampling for Exclusion Zone
            if (cx < exclusion) continue;
            if (cx > DOMAIN_SIZE - exclusion) continue;
            if (cy < exclusion) continue;
            if (cy > DOMAIN_SIZE - exclusion) continue;

            // Only add point if strictly away from boundaries
            points_with_halos.push_back(create_point_with_unique_hash_id(cx, cy));
        }
    }

    // Remove any accidental duplicates (e.g. from corner/edge overlaps or precision issues)
    remove_duplicates(points_with_halos);

    // 4. ITERATIONS
    // We relax all points that are strictly inside the generated cloud.
    // The outer hull acts as a fixed boundary condition.
    // We must freeze the outer rim of the generated cloud. 
    // If we relax the very edge, it collapses inward due to lack of outer neighbors.
    // This collapse propagates errors inward.
    // We freeze a strip of width ~ 1.5 * spacing.
    double frozen_margin = TARGET_EDGE_LENGTH * 1.5;

    double s_min_x = search_min_x + frozen_margin; 
    double s_max_x = search_max_x - frozen_margin;
    double s_min_y = search_min_y + frozen_margin; 
    double s_max_y = search_max_y - frozen_margin;

    std::vector<Triangle> triangles_with_halos;
    double current_jitter = START_JITTER;

    // Jitter + triangulate + smooth loop
    for (int iter = 0; iter < ANNEAL_ITERS; ++iter) {
        apply_jitter(points_with_halos, current_jitter, iter);
        triangles_with_halos = triangulation(points_with_halos);
        relax_points(points_with_halos, triangles_with_halos, s_min_x, s_min_y, s_max_x, s_max_y);

        // NEW: Sync boundary points immediately to prevent divergence
        ResolveBoundaryOwnership(comm, points_with_halos);        
    }
    // Final smooth iterations without jitter
    for(int k=0; k<FINAL_SMOOTH_ITERS; ++k) {
        triangles_with_halos = triangulation(points_with_halos);
        relax_points(points_with_halos, triangles_with_halos, s_min_x, s_min_y, s_max_x, s_max_y);

        // NEW: Sync boundary points immediately to prevent divergence
        ResolveBoundaryOwnership(comm, points_with_halos);        
    }
    
    // Final mesh
    triangles_with_halos = triangulation(points_with_halos);
    
    // NEW: Resolve ownership before filtering
    ResolveBoundaryOwnership(comm, points_with_halos);

    // CALCULATE VALENCE (Connectivity)
    // We do this here because we have the full halo information.
    // For owned points (which are internal to this haloed mesh), this gives the correct global valence.
    // We use a set to count unique neighbors (edges), which handles both internal and boundary nodes correctly.
    std::vector<std::set<uint64_t>> adj(points_with_halos.size());
    for(const auto& t : triangles_with_halos) {
        adj[t.v0].insert(points_with_halos[t.v1].unique_hash_id);
        adj[t.v0].insert(points_with_halos[t.v2].unique_hash_id);
        
        adj[t.v1].insert(points_with_halos[t.v0].unique_hash_id);
        adj[t.v1].insert(points_with_halos[t.v2].unique_hash_id);
        
        adj[t.v2].insert(points_with_halos[t.v0].unique_hash_id);
        adj[t.v2].insert(points_with_halos[t.v1].unique_hash_id);
    }
    
    for(size_t i=0; i<points_with_halos.size(); ++i) {
        points_with_halos[i].valence = adj[i].size();
    }

    // We only keep triangles that are geometrically "owned" by this tile
    for (const auto& tri : triangles_with_halos) {
        const Point& p0 = points_with_halos[tri.v0];
        const Point& p1 = points_with_halos[tri.v1];
        const Point& p2 = points_with_halos[tri.v2];

        // Robust Ownership Rule:
        // A triangle is owned by the rank that owns the triangle's "lowest" point.
        // We define "lowest" using the unique hash ID to ensure all ranks agree.
        // If multiple points are on the same rank, that rank definitely owns it.
        // If points are on different ranks, the one with the smallest ID decides.
        uint64_t min_hash_id = std::min({p0.unique_hash_id, p1.unique_hash_id, p2.unique_hash_id});
        
        // Find which point has this ID
        const Point* min_p = &p0;
        if (p1.unique_hash_id == min_hash_id) min_p = &p1;
        if (p2.unique_hash_id == min_hash_id) min_p = &p2;

        // Check if WE own this determining point (using fixed_owner via overload)
        int owner = get_owner_rank(*min_p, comm_size);

        if (owner == comm_rank) {
            Triangle new_t;
            Point* pts[3] = { (Point*)&p0, (Point*)&p1, (Point*)&p2 };
            int*   v_idx[3] = { &new_t.v0, &new_t.v1, &new_t.v2 };

            // This builds a mapping between the unique hash id and its index in the output array
            for(int k=0; k<3; ++k) {
                uint64_t pid = pts[k]->unique_hash_id;
                if (id_to_idx.find(pid) == id_to_idx.end()) {
                    int new_idx = points_on_owned_triangles_and_orphans.size();
                    points_on_owned_triangles_and_orphans.push_back(*pts[k]);
                    id_to_idx[pid] = new_idx;
                }
                *v_idx[k] = id_to_idx[pid];
            }
            triangles_owned.push_back(new_t);
        }
    }

    // Explicitly add "orphaned" points that we own spatially.
    // It is possible to own a point spatially (it's in our tile) but NOT own any of the
    // triangles connected to it (due to the min_id rule giving them to neighbors).
    // We must still track this point so we can assign it a Global ID and answer requests.
    for (const auto& p : points_with_halos) {
        if (get_owner_rank(p, comm_size) == comm_rank) {
             uint64_t pid = p.unique_hash_id;
             if (id_to_idx.find(pid) == id_to_idx.end()) {
                 int new_idx = points_on_owned_triangles_and_orphans.size();
                 points_on_owned_triangles_and_orphans.push_back(p);
                 id_to_idx[pid] = new_idx;
             }
        }
    }
    id_to_idx.clear();
}

// ~~~~~~~~~~~~~~~~~

// Return a PETSc DM for the points and triangles passed in
static DM CreateDM(MPI_Comm comm, const std::vector<Point>& points_on_owned_triangles_and_orphans, const std::vector<Triangle>& triangles_owned) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    PetscInt num_points_on_owned_triangles_and_orphans = points_on_owned_triangles_and_orphans.size();
    std::vector<PetscInt> global_ids(num_points_on_owned_triangles_and_orphans, -1);
    PetscInt num_points_owned = 0;

    // 1. Identify Owned points
    for (int i = 0; i < num_points_on_owned_triangles_and_orphans; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            num_points_owned++;
        }
    }

    // 2. Calculate Global Offsets - petscint to ensure large counts work
    PetscInt start_id = 0;
    MPI_Exscan(&num_points_owned, &start_id, 1, MPIU_INT, MPI_SUM, comm);

    // 3. Assign Global IDs to Owned points
    PetscInt current_id = start_id;
    for (int i = 0; i < num_points_on_owned_triangles_and_orphans; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            global_ids[i] = current_id++;
        }
    }

    // 4. Resolve Ghost IDs
    // We need to ask the owners for the Global IDs of our ghost points.
    std::vector<std::vector<uint64_t>> send_ids(comm_size);
    std::vector<std::vector<int>>      send_req_indices(comm_size);

    for (int i = 0; i < num_points_on_owned_triangles_and_orphans; ++i) {
        // If we don't own it we need to find out who does and ask them for the global id
        if (global_ids[i] == -1) {
            int owner = get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size);
            // We send the unique hash id to identify the point
            send_ids[owner].push_back(points_on_owned_triangles_and_orphans[i].unique_hash_id);
            send_req_indices[owner].push_back(i);
        }
    }

    // Exchange counts
    std::vector<int> send_counts(comm_size), recv_counts(comm_size);
    for(int r=0; r<comm_size; ++r) send_counts[r] = send_ids[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    // ---------------------------------------------------------
    // PHASE 1: Exchange Hash IDs (Requests)
    // ---------------------------------------------------------
    std::vector<std::vector<uint64_t>> recv_ids(comm_size);
    std::vector<MPI_Request> requests;
    requests.reserve(comm_size * 2);

    // 1. Post Receives for incoming requests
    for(int r=0; r<comm_size; ++r) {
        if(recv_counts[r] > 0) {
            recv_ids[r].resize(recv_counts[r]);
            MPI_Request req;
            MPI_Irecv(recv_ids[r].data(), recv_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, comm, &req);
            requests.push_back(req);
        }
    }

    // 2. Post Sends for our requests
    for(int r=0; r<comm_size; ++r) {
        if (r == comm_rank) continue;
        if (send_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_ids[r].data(), send_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, comm, &req);
            requests.push_back(req);
        }
    }

    // 3. Wait for Phase 1 to complete
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    requests.clear();

    // ---------------------------------------------------------
    // PROCESSING: Lookup Global IDs
    // ---------------------------------------------------------
    
    // We need a map for fast lookup between the unique hash id and the global id
    std::map<uint64_t, PetscInt> points_owned_l2g_map;
    for(int i=0; i<num_points_on_owned_triangles_and_orphans; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            points_owned_l2g_map[points_on_owned_triangles_and_orphans[i].unique_hash_id] = global_ids[i];
        }
    }

    std::vector<std::vector<PetscInt>> send_answers(comm_size);
    for(int r=0; r<comm_size; ++r) {
        if (recv_counts[r] > 0) {
            send_answers[r].resize(recv_counts[r]);
            for(int k=0; k<recv_counts[r]; ++k) {
                send_answers[r][k] = points_owned_l2g_map[recv_ids[r][k]];
            }
        }
    }

    // ---------------------------------------------------------
    // PHASE 2: Exchange Global IDs (Answers)
    // ---------------------------------------------------------
    std::vector<std::vector<PetscInt>> recv_answers(comm_size);

    // 1. Post Receives for answers to our requests
    // We expect 'send_counts[r]' answers from rank r
    for(int r=0; r<comm_size; ++r) {
        if (r == comm_rank) continue;
        if (send_counts[r] > 0) {
            recv_answers[r].resize(send_counts[r]);
            MPI_Request req;
            MPI_Irecv(recv_answers[r].data(), send_counts[r] * sizeof(PetscInt), MPI_BYTE, r, 101, comm, &req);
            requests.push_back(req);
        }
    }

    // 2. Post Sends for answers we generated
    for(int r=0; r<comm_size; ++r) {
        if (recv_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_answers[r].data(), recv_counts[r] * sizeof(PetscInt), MPI_BYTE, r, 101, comm, &req);
            requests.push_back(req);
        }
    }

    // 3. Wait for Phase 2 to complete
    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    // ---------------------------------------------------------
    // FINALIZE: Update the global ids
    // ---------------------------------------------------------
    for(int r=0; r<comm_size; ++r) {
        if (send_counts[r] > 0) {
            for(int k=0; k<send_counts[r]; ++k) {
                int local_idx = send_req_indices[r][k];
                global_ids[local_idx] = recv_answers[r][k];
            }
        }
    }

    // 5. Build DMPlex
    PetscInt num_tris_owned = triangles_owned.size();
    std::vector<PetscInt> cells(num_tris_owned * 3);
    // Global ids for all points on owned triangles
    for(int i=0; i<num_tris_owned; ++i) {
        cells[i*3 + 0] = global_ids[triangles_owned[i].v0];
        cells[i*3 + 1] = global_ids[triangles_owned[i].v1];
        cells[i*3 + 2] = global_ids[triangles_owned[i].v2];
    }

    // Prepare coordinates for DMPlexCreateFromCellListParallelPetsc
    // points_on_owned_triangles_and_orphans contains ghosts (points of owned triangles that are owned by neighbors).
    // PETSc expects 'numPoints' to be the count of locally owned points, 
    // and 'coords' to be the coordinates of those specific points
    std::vector<PetscReal> coords_points_owned;
    coords_points_owned.reserve(num_points_owned * 2);
    
    for(int i=0; i<num_points_on_owned_triangles_and_orphans; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            coords_points_owned.push_back(points_on_owned_triangles_and_orphans[i].x);
            coords_points_owned.push_back(points_on_owned_triangles_and_orphans[i].y);
        }
    }

    DM dm;
    // Build the DM
    PetscInt two = 2;
    PetscInt three = 3;
    (void*)DMPlexCreateFromCellListParallelPetsc(comm, two, num_tris_owned, num_points_owned, PETSC_DECIDE, \
         three, PETSC_TRUE, cells.data(), two, coords_points_owned.data(), NULL, NULL, &dm);
    return dm;
}

// ~~~~~~~~~~~~~~~~~

// Label boundary faces and vertices based on geometric location
static void LabelBoundaries(DM dm) {

    // Create a label named "Face Sets" (standard name for boundary markers)
    // Values: 1=Bottom, 2=Right, 3=Top, 4=Left
    PetscCallVoid(DMCreateLabel(dm, "Face Sets"));
    DMLabel label;
    PetscCallVoid(DMGetLabel(dm, "Face Sets", &label));
    
    // Get coordinates
    Vec coordsVec;
    PetscCallVoid(DMGetCoordinatesLocal(dm, &coordsVec));
    PetscSection coordSection;
    PetscCallVoid(DMGetCoordinateSection(dm, &coordSection));

    const PetscScalar *coords;
    PetscCallVoid(VecGetArrayRead(coordsVec, &coords));
    
    // 1. Label Vertices (Depth 0)
    PetscInt vStart, vEnd;
    PetscCallVoid(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    
    for (PetscInt v = vStart; v < vEnd; ++v) {
        PetscInt dof;
        PetscCallVoid(PetscSectionGetDof(coordSection, v, &dof));
        if (dof > 0) {
            PetscInt off;
            PetscCallVoid(PetscSectionGetOffset(coordSection, v, &off));
            double x = coords[off];
            double y = coords[off+1];
            
            PetscInt val = 0;
            // Priority for corners: Bottom > Right > Top > Left
            if (std::abs(y) < EPSILON) val = 1;              // Bottom
            else if (std::abs(x - DOMAIN_SIZE) < EPSILON) val = 2; // Right
            else if (std::abs(y - DOMAIN_SIZE) < EPSILON) val = 3; // Top
            else if (std::abs(x) < EPSILON) val = 4;         // Left

            if (val != 0) PetscCallVoid(DMLabelSetValue(label, v, val));
        }
    }
    
    PetscCallVoid(DMCreateLabel(dm, "markers"));
    PetscCallVoid(DMGetLabel(dm, "markers", &label));

    // 2. Label Edges (Depth 1 in 2D)
    PetscInt eStart, eEnd;
    PetscCallVoid(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
    
    for (PetscInt e = eStart; e < eEnd; ++e) {
        PetscInt num_points;
        const PetscInt *points;
        PetscCallVoid(DMPlexGetConeSize(dm, e, &num_points));
        PetscCallVoid(DMPlexGetCone(dm, e, &points));
        
        // Compute centroid of edge to determine boundary
        double cx = 0, cy = 0;
        int count = 0;
        
        for(int i=0; i<num_points; ++i) {
            PetscInt v = points[i];
            PetscInt off, dof;
            PetscCallVoid(PetscSectionGetDof(coordSection, v, &dof));
            if (dof > 0) {
                PetscCallVoid(PetscSectionGetOffset(coordSection, v, &off));
                cx += coords[off];
                cy += coords[off+1];
                count++;
            }
        }
        
        if (count > 0) {
            cx /= count;
            cy /= count;
            
            PetscInt val = 0;
            if (std::abs(cy) < EPSILON) val = 1;              // Bottom
            else if (std::abs(cx - DOMAIN_SIZE) < EPSILON) val = 2; // Right
            else if (std::abs(cy - DOMAIN_SIZE) < EPSILON) val = 3; // Top
            else if (std::abs(cx) < EPSILON) val = 4;         // Left

            if (val != 0) PetscCallVoid(DMLabelSetValue(label, e, val));
        }
    }

    PetscCallVoid(VecRestoreArrayRead(coordsVec, &coords));
}

// ~~~~~~~~~~~~~~~~~

// Perform rigorous checks on mesh topology and geometry
static bool CheckMeshIntegrity(MPI_Comm comm, const std::vector<Point>& points_on_owned_triangles_and_orphans, const std::vector<Triangle>& triangles_owned) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Count Owned Points (needed for Euler)
    long num_points_owned = 0;
    for (const auto& p : points_on_owned_triangles_and_orphans) {
        if (get_owner_rank(p, size) == rank) {
            num_points_owned++;
        }
    }

    // 2. Accumulate Local Stats
    double local_total_area = 0.0;
    double local_boundary_len = 0.0;
    long local_boundary_edge_count = 0;
    long local_bad_edge_count = 0;
    double local_max_edge_len = 0.0;
    
    const double MAX_EDGE_RATIO = 3.0; 
    const double THRESHOLD_LEN = TARGET_EDGE_LENGTH * MAX_EDGE_RATIO;

    for (const auto& t : triangles_owned) {
        const Point& p0 = points_on_owned_triangles_and_orphans[t.v0];
        const Point& p1 = points_on_owned_triangles_and_orphans[t.v1];
        const Point& p2 = points_on_owned_triangles_and_orphans[t.v2];

        // Edge Lengths
        double d01_sq = std::pow(p1.x-p0.x, 2) + std::pow(p1.y-p0.y, 2);
        double d12_sq = std::pow(p2.x-p1.x, 2) + std::pow(p2.y-p1.y, 2);
        double d20_sq = std::pow(p0.x-p2.x, 2) + std::pow(p0.y-p2.y, 2);
        
        double d01 = std::sqrt(d01_sq);
        double d12 = std::sqrt(d12_sq);
        double d20 = std::sqrt(d20_sq);

        local_max_edge_len = std::max({local_max_edge_len, d01, d12, d20});

        if (d01 > THRESHOLD_LEN || d12 > THRESHOLD_LEN || d20 > THRESHOLD_LEN) {
            local_bad_edge_count++;
        }

        // Area
        double volume = 0.5 * std::abs((p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x));
        local_total_area += volume;

        // Boundary Consistency
        Point pts[3] = {p0, p1, p2};
        for (int i = 0; i < 3; i++) {
            Point& ep1 = pts[i];
            Point& ep2 = pts[(i + 1) % 3];
            
            bool is_bdy = false;
            if (std::abs(ep1.x) < EPSILON && std::abs(ep2.x) < EPSILON) is_bdy = true; 
            else if (std::abs(ep1.x - DOMAIN_SIZE) < EPSILON && std::abs(ep2.x - DOMAIN_SIZE) < EPSILON) is_bdy = true; 
            else if (std::abs(ep1.y) < EPSILON && std::abs(ep2.y) < EPSILON) is_bdy = true; 
            else if (std::abs(ep1.y - DOMAIN_SIZE) < EPSILON && std::abs(ep2.y - DOMAIN_SIZE) < EPSILON) is_bdy = true; 
            
            if (is_bdy) {
                double len = std::sqrt(std::pow(ep1.x-ep2.x, 2) + std::pow(ep1.y-ep2.y, 2));
                local_boundary_len += len;
                local_boundary_edge_count++;
            }
        }
    }

    // 3. Global Reductions
    long num_tris_owned = triangles_owned.size();
    long num_tris_owned_global, num_points_owned_global;
    double global_total_area, global_boundary_len;
    long global_boundary_edge_count, global_bad_edge_count;
    double global_max_edge_len;

    MPI_Reduce(&num_tris_owned, &num_tris_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&num_points_owned, &num_points_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_total_area, &global_total_area, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_boundary_len, &global_boundary_len, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_boundary_edge_count, &global_boundary_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_bad_edge_count, &global_bad_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max_edge_len, &global_max_edge_len, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int success = 1;
    if (rank == 0) {
        long long total_degrees = (long long)num_tris_owned_global * 3;
        long long double_edges = total_degrees + global_boundary_edge_count;
        long long E_total = double_edges / 2;
        long long V_total = num_points_owned_global;
        long long F_total = num_tris_owned_global;
        long long euler = V_total - E_total + F_total;

        bool area_pass = std::abs(global_total_area - 1.0) < 1e-6;
        bool perim_pass = std::abs(global_boundary_len - 4.0) < 1e-4;
        bool euler_pass = (euler == 1);
        bool edge_pass = (global_bad_edge_count == 0);

        if (!area_pass || !perim_pass || !euler_pass || !edge_pass) {
            success = 0;
            std::cout << "\n!!! MESH INTEGRITY CHECK FAILED !!!\n";
            if (!area_pass) std::cout << "  [FAIL] Total Area: " << std::fixed << std::setprecision(6) << global_total_area << " (Expected 1.0)\n";
            if (!perim_pass) std::cout << "  [FAIL] Boundary Perimeter: " << global_boundary_len << " (Expected 4.0)\n";
            if (!euler_pass) std::cout << "  [FAIL] Euler Characteristic: " << euler << " (Expected 1)\n";
            if (!edge_pass) {
                std::cout << "  [FAIL] Bad Edges: " << global_bad_edge_count << " edges > " << MAX_EDGE_RATIO << "x target.\n";
                std::cout << "         Max Edge: " << global_max_edge_len << "\n";
            }
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        }
    }
    
    MPI_Bcast(&success, 1, MPI_INT, 0, comm);
    return (success == 1);
}

// Print mesh statistics on rank 0
static void ComputeAndPrintStats(MPI_Comm comm, const std::vector<Point>& points_on_owned_triangles_and_orphans, const std::vector<Triangle>& triangles_owned) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Compute load imbalance & Connectivity
    long points_owned = 0;
    const int MAX_CONN = 30;
    long local_conn_bins[MAX_CONN] = {0};
    
    // Bin the valences
    for (const auto& p : points_on_owned_triangles_and_orphans) {
        if (get_owner_rank(p, size) == rank) {
            points_owned++;
            
            // Bin Connectivity using pre-calculated valence
            int degree = p.valence;
            if (degree >= MAX_CONN) degree = MAX_CONN - 1;
            local_conn_bins[degree]++;
        }
    }

    long min_points_owned_global, max_points_owned_global, num_points_owned_global;
    MPI_Reduce(&points_owned, &min_points_owned_global, 1, MPI_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&points_owned, &max_points_owned_global, 1, MPI_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&points_owned, &num_points_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);

    long global_conn_bins[MAX_CONN] = {0};
    MPI_Reduce(local_conn_bins, global_conn_bins, MAX_CONN, MPI_LONG, MPI_SUM, 0, comm);

    // 2. Triangle Statistics (Volume & Angles)
    long num_tris_owned = triangles_owned.size();
    long num_tris_owned_global;
    MPI_Reduce(&num_tris_owned, &num_tris_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);

    double local_min_volume = 1e30, local_max_volume = -1.0;
    double local_min_angle = 360.0, local_max_angle = -1.0;

    // 3. Edge Orientation Statistics
    std::vector<long> local_bins(18, 0);
    std::set<std::pair<int, int>> processed_edges;

    for (const auto& t : triangles_owned) {
        const Point& p0 = points_on_owned_triangles_and_orphans[t.v0];
        const Point& p1 = points_on_owned_triangles_and_orphans[t.v1];
        const Point& p2 = points_on_owned_triangles_and_orphans[t.v2];

        double d01_sq = std::pow(p1.x-p0.x, 2) + std::pow(p1.y-p0.y, 2);
        double d12_sq = std::pow(p2.x-p1.x, 2) + std::pow(p2.y-p1.y, 2);
        double d20_sq = std::pow(p0.x-p2.x, 2) + std::pow(p0.y-p2.y, 2);
        
        double d01 = std::sqrt(d01_sq);
        double d12 = std::sqrt(d12_sq);
        double d20 = std::sqrt(d20_sq);

        // volume
        double volume = 0.5 * std::abs((p1.x - p0.x)*(p2.y - p0.y) - (p1.y - p0.y)*(p2.x - p0.x));
        if (volume < local_min_volume) local_min_volume = volume;
        if (volume > local_max_volume) local_max_volume = volume;

        // Angles
        if (d01 > 1e-14 && d12 > 1e-14 && d20 > 1e-14) {
            double a0 = std::acos(clamp_val((d01_sq + d20_sq - d12_sq) / (2.0*d01*d20))) * 180.0 / 3.14159265358979323846;
            double a1 = std::acos(clamp_val((d01_sq + d12_sq - d20_sq) / (2.0*d01*d12))) * 180.0 / 3.14159265358979323846;
            double a2 = std::acos(clamp_val((d12_sq + d20_sq - d01_sq) / (2.0*d12*d20))) * 180.0 / 3.14159265358979323846;

            local_min_angle = std::min({local_min_angle, a0, a1, a2});
            local_max_angle = std::max({local_max_angle, a0, a1, a2});
        }

        // Edges
        int v[3] = {t.v0, t.v1, t.v2};
        for (int i = 0; i < 3; ++i) {
            int idx1 = v[i];
            int idx2 = v[(i + 1) % 3];
            
            // Sort indices for uniqueness check
            if (idx1 > idx2) std::swap(idx1, idx2);
            
            if (processed_edges.find({idx1, idx2}) == processed_edges.end()) {
                processed_edges.insert({idx1, idx2});
                
                const Point& ep1 = points_on_owned_triangles_and_orphans[idx1];
                const Point& ep2 = points_on_owned_triangles_and_orphans[idx2];
                
                // Check ownership of edge midpoint
                double mx = (ep1.x + ep2.x) * 0.5;
                double my = (ep1.y + ep2.y) * 0.5;
                
                if (get_owner_rank(Point{mx, my}, size) == rank) {
                    double dx = ep2.x - ep1.x;
                    double dy = ep2.y - ep1.y;
                    // Angle in [0, 180)
                    double angle = std::atan2(dy, dx) * 180.0 / 3.14159265358979323846;
                    if (angle < 0) angle += 180.0;
                    if (angle >= 180.0) angle -= 180.0;
                    
                    int bin = static_cast<int>(angle / 10.0);
                    if (bin < 0) bin = 0;
                    if (bin >= 18) bin = 17;
                    
                    local_bins[bin]++;
                }
            }
        }
    }

    if (triangles_owned.empty()) {
        local_min_volume = 1e30; local_max_volume = -1.0;
        local_min_angle = 360.0; local_max_angle = -1.0;
    }

    double global_min_volume, global_max_volume;
    double global_min_angle, global_max_angle;

    MPI_Reduce(&local_min_volume, &global_min_volume, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_max_volume, &global_max_volume, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_min_angle, &global_min_angle, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_max_angle, &global_max_angle, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    std::vector<long> global_bins(18);
    MPI_Reduce(local_bins.data(), global_bins.data(), 18, MPI_LONG, MPI_SUM, 0, comm);

    // Output stats to rank 0
    if (rank == 0) {
        std::cout << "\n=== Mesh Statistics ===\n";
        std::cout << "Points:\n";
        std::cout << "  Total: " << num_points_owned_global << "\n";
        std::cout << "  Min per Rank: " << min_points_owned_global << "\n";
        std::cout << "  Max per Rank: " << max_points_owned_global << "\n";
        std::cout << "  Imbalance Ratio (Max/Avg): " << (double)max_points_owned_global / ((double)num_points_owned_global / size) << "\n";
        
        std::cout << "Connectivity (Valence):\n";
        for(int i=0; i<MAX_CONN; ++i) {
            if (global_conn_bins[i] > 0) {
                double pct = 100.0 * global_conn_bins[i] / num_points_owned_global;
                std::cout << "  Degree " << std::setw(2) << i << ": " 
                          << std::setw(8) << global_conn_bins[i] 
                          << " (" << std::fixed << std::setprecision(2) << pct << "%)\n";
            }
        }

        std::cout << "Triangles:\n";
        std::cout << "  Total: " << num_tris_owned_global << "\n";
        std::cout << "  Volume Min: " << std::scientific << global_min_volume << "\n";
        std::cout << "  Volume Max: " << std::scientific << global_max_volume << "\n";
        
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "  Volume Ratio: " << (global_min_volume > 0 ? global_max_volume / global_min_volume : -1.0) << "\n";
        std::cout << "  Angle Min: " << global_min_angle << " deg\n";
        std::cout << "  Angle Max: " << global_max_angle << " deg\n";

        std::cout << "Edge Orientations (10 deg bins):\n";
        long total_edges = 0;
        for(long c : global_bins) total_edges += c;
        
        if (total_edges > 0) {
            for (int i = 0; i < 18; ++i) {
                double pct = 100.0 * global_bins[i] / total_edges;
                std::cout << "  " << std::setw(3) << (i*10) << "-" << std::setw(3) << ((i+1)*10) 
                          << " deg: " << std::fixed << std::setprecision(2) << pct << "%\n";
            }
        }
        std::cout << "=======================\n";
    }
}

// ~~~~~~~~~~~~~~~~~

DM GenerateBoxMeshDM(MPI_Comm comm, double target_edge_length, PetscBool print_stats) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    // 1. Setup Globals
    TARGET_EDGE_LENGTH = target_edge_length;
    TOL_LEN = TARGET_EDGE_LENGTH * 1e-4;
    TOL_LEN_SQ = TOL_LEN * TOL_LEN;
    TOL_VOLUME = TOL_LEN_SQ * 1e-2;

    // Factorize comm_size into M x N such that M*N = comm_size and M, N are as close as possible
    // This minimizes the perimeter (halo) for a given area.
    int m = std::round(std::sqrt(comm_size));
    while (comm_size % m != 0) {
        m--;
    }
    int n = comm_size / m;
    
    TILE_DIM_X = m;
    TILE_DIM_Y = n;

    if (comm_rank == 0 && print_stats) {
        std::cout << "Generating Unstructured Mesh of 2D box...\n";
        std::cout << "Target Edge Length: " << TARGET_EDGE_LENGTH << "\n";
        std::cout << "Running on " << comm_size << " MPI ranks with decomposition " << TILE_DIM_X << "x" << TILE_DIM_Y << ".\n";
    }

    std::vector<Point> points_on_owned_triangles_and_orphans;
    std::vector<Triangle> triangles_owned;

    // 2. Distribute tiles (1 per rank)
    for (int y = 0; y < TILE_DIM_Y; ++y) {
        for (int x = 0; x < TILE_DIM_X; ++x) {
            int global_id = y * TILE_DIM_X + x;
            
            // Since we have exactly comm_size tiles, global_id maps directly to rank
            if (global_id == comm_rank) {
                process_tile(comm, x, y, points_on_owned_triangles_and_orphans, triangles_owned);
            }
        }
    }

    if (print_stats) std::cout << "Rank " << comm_rank << " generated " << triangles_owned.size() << " triangles_owned.\n";

    // 3. Check Integrity
    if (!CheckMeshIntegrity(comm, points_on_owned_triangles_and_orphans, triangles_owned)) {
        return NULL;
    }

    // 4. Print stats
    if (print_stats) ComputeAndPrintStats(comm, points_on_owned_triangles_and_orphans, triangles_owned);

    // 5. Create the DM
    if (comm_rank == 0) std::cout << "Creating DM...\n";
    DM dm = CreateDM(comm, points_on_owned_triangles_and_orphans, triangles_owned);
    (void*)PetscObjectSetName((PetscObject)dm, "Mesh");
    
    // 6. Label boundaries
    LabelBoundaries(dm);

    return dm;
}

// =========================================================
// Main Driver
// =========================================================

int main(int argc, char** argv) {

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);    

    // Parse command line options
    double target_len = 0.0025;
    PetscBool set;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-target_edge_length", &target_len, &set));

    PetscBool write_mesh = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_mesh", &write_mesh, NULL));

    PetscBool print_stats = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_stats", &print_stats, NULL));

    // Generate the DMPlex for this mesh
    DM dm = GenerateBoxMeshDM(MPI_COMM_WORLD, target_len, print_stats); 

    // Check a valid mesh has been generated
    if (dm) {

        // Write output if requested
        // Can view this in paraview with:
        // /home/sdargavi/projects/dependencies/petsc_main/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5
        // then using the XDMF reader in:
        // paraview box_mesh.xmf
        if (write_mesh) {
           PetscViewer viewer;
           if (comm_rank == 0) {
                 std::cout << "Writing out mesh...\n";        
           }
           PetscCall(PetscViewerHDF5Open(MPI_COMM_WORLD, "box_mesh.h5", FILE_MODE_WRITE, &viewer));
           PetscCall(DMView(dm, viewer));
           PetscCall(PetscViewerDestroy(&viewer));
        }         

        PetscCall(DMDestroy(&dm));
    } else {
        PetscCall(PetscFinalize());
        return EXIT_FAILURE;
    }

    PetscCall(PetscFinalize());
    return 0;
}
