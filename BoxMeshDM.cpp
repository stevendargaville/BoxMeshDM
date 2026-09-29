#include "BoxMeshDM.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <mpi.h>
#include <petsc/private/dmpleximpl.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include "petscconf.h"
#include <unordered_map>
#include <set>
#include <unordered_set>

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
// 1. Boundary Gen: Create points explicitly on [0,DOMAIN_SIZE[0]] x [0,DOMAIN_SIZE[1]] boundaries.
// 2. Interior Gen: Create random points inside small squares, REJECTING those near boundaries.
// 3. Iterate: jitter -> triangulate -> smooth loop.
// 4. Constraint: Boundary nodes only move tangentially.
//
// Structure: the pipeline is written once as templates on the dimension DIM. Everything that
// depends on the dimension (the Delaunay backend, the hash id layout, the simplex kernels, the
// boundary feature and wall tables, the rank grid factorisation, the integrity/stats formulas)
// is an explicit specialisation, never a branch on DIM inside a template body. Only DIM=2 is
// reachable; the DIM=3 specialisations are stubs that abort, so the generic code is checked to
// compile for 3D. For DIM=2 every floating-point expression is the same as the original 2D
// code, token for token and in the same order, so the generated mesh is bit-identical.
// =========================================================

// Axis-indexed globals, [0]=x, [1]=y, [2]=z. Only the first DIM entries are used.
static int TILE_DIM[3] = {-1, -1, -1};

// Agglomeration: the fine tile grid is forced to be a refinement of a coarse
// grid of COARSE_DIM[0] x COARSE_DIM[1] tiles, each coarse tile being split into
// SUB_DIM[0] x SUB_DIM[1] fine tiles. AGG_FACTOR = SUB_DIM[0] * SUB_DIM[1] fine
// ranks make up one coarse group, and they are numbered consecutively.
static int AGG_FACTOR = 1;                  // k: fine ranks per coarse group
static int COARSE_DIM[3] = {-1, -1, -1};    // Mc, Nc
static int SUB_DIM[3] = {1, 1, 1};          // a, b: sub-block tiles per coarse tile along each axis

static double DOMAIN_SIZE[3] = {1.0, 1.0, 1.0};
static double TARGET_EDGE_LENGTH = 0.0025;

static double TOL_LEN;
static double TOL_LEN_SQ;
static double TOL_VOLUME;

const double EPSILON = 1e-13;
const double START_JITTER = 0.30;

// Jitter + smooth iterations first
const int ANNEAL_ITERS = 3;

// Grid index used for the right/top walls in the unique hash id. Every other grid
// index must stay below this (and so within the 31 bits packed into the hash id)
const int MAX_GRID_IDX = 2000000000;

// Number of 10 degree bins in the edge orientation histogram of the stats
const int NUM_ORIENTATION_BINS = 18;

// ~~~~~~~~~~~~~~~~~

template <int DIM>
struct Point {
    double c[DIM]; // coordinates
    uint64_t unique_hash_id = 0; // unique id based on hashing coordinates

    // Constructors for C++11 compatibility
    Point() : unique_hash_id(0) { for (int d = 0; d < DIM; ++d) c[d] = 0; }
    Point(const double* c_, uint64_t id) : unique_hash_id(id) { for (int d = 0; d < DIM; ++d) c[d] = c_[d]; }
};
// A triangle (DIM=2) or tetrahedron (DIM=3), as indices into a point list
template <int DIM>
struct Simplex {
    int v[DIM + 1];
};
struct Edge {
    int v0, v1;
    bool operator==(const Edge& o) const { return (v0 == o.v0 && v1 == o.v1) || (v0 == o.v1 && v1 == o.v0); }
};
// Stateless random
struct RngState { uint64_t s; };

// ~~~~~~~~~~~~~~~~~

// Stops the run in a DIM=3 specialisation that is not written yet. None of these are
// reachable: GenerateBoxMeshDMImpl<3> is only instantiated, never called
static void AbortNotImplemented3D(const char *what) {
    std::cerr << "ERROR: " << what << " is not implemented in 3D yet.\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

// Steps idx through the box [lo, hi] (inclusive) with axis 0 fastest. Returns false once
// every index has been visited, so "do { ... } while (next_index<DIM>(idx, lo, hi))" visits
// the box in the order of nested loops with the last axis outermost (a continue in the body
// moves on to the next index)
template <int DIM>
static inline bool next_index(int *idx, const int *lo, const int *hi) {
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] < hi[d]) { ++idx[d]; return true; }
        idx[d] = lo[d];
    }
    return false;
}

// Compile-time unrolled loop: StaticFor<N>::run(f) calls f(0), f(1), ..., f(N-1) as straight-line
// code. Used for per-axis/per-vertex loops whose floating-point code has to compile exactly as the
// original hand-written x/y code did: at -O2 a plain "for (d < DIM)" loop is often left rolled,
// and a rolled loop can change which multiply-adds the compiler contracts into FMAs (e.g. by
// hoisting a loop-invariant product out of the loop), which changes the mesh in FMA builds
template <int N>
struct StaticFor {
    template <typename F>
    static inline void run(F& f) { StaticFor<N - 1>::run(f); f(N - 1); }
};
template <>
struct StaticFor<0> {
    template <typename F>
    static inline void run(F&) {}
};

// Print coordinates as "x, y" (", z")
template <int DIM>
static void print_coords(std::ostream& os, const double *c) {
    for (int d = 0; d < DIM; ++d) {
        if (d > 0) os << ", ";
        os << c[d];
    }
}

// Print a grid shape as "AxB" ("xC")
template <int DIM>
static void print_dims(std::ostream& os, const int *dims) {
    for (int d = 0; d < DIM; ++d) {
        if (d > 0) os << "x";
        os << dims[d];
    }
}

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

// Hierarchical tile <-> rank mapping. The AGG_FACTOR fine tiles that make up
// coarse group cg are exactly ranks [cg*AGG_FACTOR, (cg+1)*AGG_FACTOR), so a
// downstream consumer merging every AGG_FACTOR consecutive ranks recovers the
// coarse decomposition. With AGG_FACTOR == 1 these reduce to plain row-major.
// Both the coarse group and the position within it are numbered with axis 0 fastest.
template <int DIM>
static inline int tile_to_rank(const int *t) {
    int cg = 0, sub = 0;
    for (int d = DIM - 1; d >= 0; --d) {
        cg = cg * COARSE_DIM[d] + t[d] / SUB_DIM[d];
        sub = sub * SUB_DIM[d] + t[d] % SUB_DIM[d];
    }
    return cg * AGG_FACTOR + sub;
}

template <int DIM>
static inline void rank_to_tile(int rank, int *t) {
    int cg = rank / AGG_FACTOR, sub = rank % AGG_FACTOR;
    for (int d = 0; d < DIM; ++d) {
        t[d] = (cg % COARSE_DIM[d]) * SUB_DIM[d] + (sub % SUB_DIM[d]);
        cg /= COARSE_DIM[d];
        sub /= SUB_DIM[d];
    }
}

// Helper to determine which rank owns a point based on spatial location
template <int DIM>
static int get_owner_rank(const Point<DIM>& p, int size) {

    int t[DIM];
    for (int d = 0; d < DIM; ++d) {
        double tile_s = DOMAIN_SIZE[d] / TILE_DIM[d];
        t[d] = std::floor(p.c[d] / tile_s);
    }

    // Clamp to handle numerical noise at upper boundaries
    for (int d = 0; d < DIM; ++d) {
        if (t[d] < 0) t[d] = 0;
        if (t[d] >= TILE_DIM[d]) t[d] = TILE_DIM[d] - 1;
    }

    // With 1 tile per rank, the tile maps directly to a rank
    (void)size;
    return tile_to_rank<DIM>(t);
}

// Deterministically resolve ownership of boundary nodes
template <int DIM>
static void ResolveBoundaryOwnership(MPI_Comm comm, std::vector<Point<DIM> >& points, const double *box_min, const double *box_max,
                                    const double *interior_min, const double *interior_max, double pad) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    struct Claim {
        uint64_t id;
        int geo_rank;
        double c[DIM]; // Include coordinates for synchronization
    };

    // Create MPI Datatype for Claim
    // This maps the struct layout so MPI knows how to send it as a single unit
    MPI_Datatype MPI_CLAIM;
    {
        int blocklengths[3] = {1, 1, DIM}; // 1 uint64, 1 int, DIM doubles (x, y)
        MPI_Aint displacements[3];
        MPI_Datatype types[3] = {MPI_UINT64_T, MPI_INT, MPI_DOUBLE};

        Claim dummy;
        MPI_Aint base_addr;
        MPI_Get_address(&dummy, &base_addr);
        MPI_Get_address(&dummy.id, &displacements[0]);
        MPI_Get_address(&dummy.geo_rank, &displacements[1]);
        MPI_Get_address(&dummy.c[0], &displacements[2]);

        // Make displacements relative to the start of the struct
        for(int i=0; i<3; i++) displacements[i] -= base_addr;

        MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_CLAIM);
        MPI_Type_commit(&MPI_CLAIM);
    }

    std::vector<std::vector<Claim>> send_buffers(size);
    std::unordered_set<uint64_t> involved_ids; // Only resolve points near boundaries

    // Simple neighbor discovery: check all ranks (for small scale) or use grid logic
    // Here we use the grid logic to only send to actual neighbors
    int t[DIM];
    rank_to_tile<DIM>(rank, t);

    // Neighbour offsets in {-1,0,1}^DIM, visited with axis 0 fastest (dx inner, dy outer)
    int off_lo[DIM], off_hi[DIM];
    for (int d = 0; d < DIM; ++d) { off_lo[d] = -1; off_hi[d] = 1; }

    for (auto& p : points) {
        // Check if point is within the resolution box
        // We use strict inequality to match relax_points logic (points on the exact edge of the box are frozen)
        // but we also don't send purely interior points as they are not shared
        bool in_box = true, near_tile_edge = false;
        for (int d = 0; d < DIM; ++d) {
            if (!(p.c[d] > box_min[d] && p.c[d] < box_max[d])) in_box = false;
            if (p.c[d] <= interior_min[d] || p.c[d] >= interior_max[d]) near_tile_edge = true;
        }
        if (in_box && near_tile_edge) {

            involved_ids.insert(p.unique_hash_id);
            int my_geo_rank = get_owner_rank(p, size);

            // It's a boundary candidate. Send to neighbors.
            int off[DIM];
            for (int d = 0; d < DIM; ++d) off[d] = off_lo[d];
            do {
                bool is_self = true, in_grid = true;
                int n[DIM];
                for (int d = 0; d < DIM; ++d) {
                    if (off[d] != 0) is_self = false;
                    n[d] = t[d] + off[d];
                    if (!(n[d] >= 0 && n[d] < TILE_DIM[d])) in_grid = false;
                }
                if (is_self) continue;
                if (in_grid) {

                    // Only send if point could be in the neighbor halo
                    // Check if point is within (Neighbor_Box + Pad)
                    // For safety let's expand it
                    double safe_pad = pad * 1.2;

                    bool relevant = true;
                    auto check_axis = [&](int d) {
                        double tile_s = DOMAIN_SIZE[d] / TILE_DIM[d];
                        double n_min = n[d] * tile_s; double n_max = (n[d] + 1) * tile_s;
                        if (p.c[d] < n_min - safe_pad) relevant = false;
                        if (p.c[d] > n_max + safe_pad) relevant = false;
                    };
                    StaticFor<DIM>::run(check_axis);

                    if (relevant) {
                        int n_rank = tile_to_rank<DIM>(n);
                        Claim claim;
                        claim.id = p.unique_hash_id;
                        claim.geo_rank = my_geo_rank;
                        for (int d = 0; d < DIM; ++d) claim.c[d] = p.c[d];
                        send_buffers[n_rank].push_back(claim);
                    }
                }
            } while (next_index<DIM>(off, off_lo, off_hi));
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
            // Use MPI_CLAIM type, count is number of items (not bytes)
            MPI_Irecv(recv_buffers[r].data(), recv_counts[r], MPI_CLAIM, r, 999, comm, &req);
            requests.push_back(req);
        }
    }
    for(int r=0; r<size; ++r) {
        if (send_counts[r] > 0) {
            MPI_Request req;
            // Use MPI_CLAIM type, count is number of items (not bytes)
            MPI_Isend(send_buffers[r].data(), send_counts[r], MPI_CLAIM, r, 999, comm, &req);
            requests.push_back(req);
        }
    }
    if (!requests.empty()) MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // Clean up the custom type
    MPI_Type_free(&MPI_CLAIM);
    send_buffers.clear();
    send_counts.clear();
    recv_counts.clear();
    // Explicitly delete the memory
    std::vector<std::vector<Claim>>().swap(send_buffers);
    std::vector<int>().swap(send_counts);
    std::vector<int>().swap(recv_counts);

    // 3. Resolve Ownership
    // We track:
    // - The set of geometric ranks calculated by all claimers (to detect disagreement)
    // - The lowest rank that claimed the point (to pick a winner)
    // - The coordinates associated with that lowest rank (to sync geometry)
    struct ResolutionData {
        int best_rank;
        double best_c[DIM];
        ResolutionData() : best_rank(999999) { for (int d = 0; d < DIM; ++d) best_c[d] = 0; }
    };
    std::unordered_map<uint64_t, ResolutionData> resolution_map;

    // Initialize with self ONLY for involved points
    for(const auto& p : points) {
        if (involved_ids.count(p.unique_hash_id)) {
            ResolutionData& data = resolution_map[p.unique_hash_id];
            data.best_rank = rank;
            for (int d = 0; d < DIM; ++d) data.best_c[d] = p.c[d];
        }
    }

    // Update with neighbors
    for(int r=0; r<size; ++r) {
        for(const auto& claim : recv_buffers[r]) {
            involved_ids.insert(claim.id); // Mark as involved if a neighbor claims it
            ResolutionData& data = resolution_map[claim.id];

            // Update best rank (lowest wins) and its coordinates
            // CRITICAL: We ALWAYS take the coordinates from the lowest rank,
            // regardless of whether there is an ownership dispute.
            if (r < data.best_rank) {
                data.best_rank = r;
                for (int d = 0; d < DIM; ++d) data.best_c[d] = claim.c[d];
            }
        }
    }

    // 4. Apply to points
    for(auto& p : points) {
        // Only process points involved in the boundary resolution
        if (involved_ids.count(p.unique_hash_id)) {
            const ResolutionData& data = resolution_map[p.unique_hash_id];

            // 4a. Synchronize Coordinates
            // Everyone adopts the coordinates of the 'best_rank' to ensure geometric consistency
            // This happens for ALL shared points now.
            for (int d = 0; d < DIM; ++d) p.c[d] = data.best_c[d];
        }
    }
    involved_ids.clear();
    resolution_map.clear();
}

// ~~~~~~~~~~~~~~~~~

// Kept out of line: the callers then add the (possibly zeroed) move to the point as a
// separate multiply and add, as the original 2D code compiled. Inlined, the compiler can
// contract "candidate += factor * full_move" into an FMA, which changes the mesh in FMA builds
#if defined(__GNUC__)
#define BOXMESHDM_NOINLINE __attribute__((noinline))
#else
#define BOXMESHDM_NOINLINE
#endif

// Returns true if point is on a boundary.
// Modifies delta to ensure movement is only tangential (sliding): each axis on which the
// point is within EPSILON of a wall (x=0 / x=DOMAIN_SIZE[0], y=0 / y=DOMAIN_SIZE[1], ...)
// is snapped exactly onto that wall and its component of the move zeroed.
template <int DIM>
BOXMESHDM_NOINLINE static bool apply_boundary_constraint(Point<DIM>& p, double *delta) {
    bool on_boundary = false;

    auto constrain_axis = [&](int d) {
        // Low wall (e.g. Left x=0, Bottom y=0)
        if (std::abs(p.c[d]) < EPSILON) {
            p.c[d] = 0.0; delta[d] = 0.0;
            on_boundary = true;
        }
        // High wall (e.g. Right x=DOMAIN_SIZE[0], Top y=DOMAIN_SIZE[1])
        else if (std::abs(p.c[d] - DOMAIN_SIZE[d]) < EPSILON) {
            p.c[d] = DOMAIN_SIZE[d]; delta[d] = 0.0;
            on_boundary = true;
        }
    };
    StaticFor<DIM>::run(constrain_axis);

    return on_boundary;
}

// Check a tangential move of a wall point stays clear of the corners.
// wall_p must already be snapped by apply_boundary_constraint, so comparing
// against the walls exactly identifies which wall(s) it is on.
// Clamping the move instead would place the point exactly on the corner
// (or inside its EPSILON capture zone), creating a duplicate of the explicit
// corner point; Triangle then drops one of the two and the orphaned vertex
// breaks the Euler characteristic in CheckMeshIntegrity.
// Rejecting the move keeps the previous position, which is already distinct,
// so no coincident points can ever be created.
// General rule: if the point is on any wall, every free axis (one it is not on a wall of)
// must stay corner_margin clear of both ends. In 2D a wall point on the left/right wall slides
// in y and one on the bottom/top wall slides in x, exactly as before. A corner has no free axis
// so this returns true for it (the old code returned false), but a corner's move is zero on
// every axis and corners are exact, so the candidate is bit-equal to the corner and accepting
// or rejecting it gives the same result. That relies on corners being generated and snapped
// exactly onto the walls.
template <int DIM>
static bool boundary_move_valid(const Point<DIM>& wall_p, const Point<DIM>& candidate) {
    double corner_margin = 0.25 * TARGET_EDGE_LENGTH;
    bool on_wall[DIM];
    bool any_wall = false;
    auto find_walls = [&](int d) {
        on_wall[d] = (wall_p.c[d] == 0.0 || wall_p.c[d] == DOMAIN_SIZE[d]);
        if (on_wall[d]) any_wall = true;
    };
    StaticFor<DIM>::run(find_walls);
    if (!any_wall) return true;
    // The point slides along its free axes, keep it away from the corners
    bool valid = true;
    auto check_free_axis = [&](int e) {
        if (on_wall[e]) return;
        if (candidate.c[e] < corner_margin || candidate.c[e] > DOMAIN_SIZE[e] - corner_margin) valid = false;
    };
    StaticFor<DIM>::run(check_free_axis);
    return valid;
}

// Ensure interior points stay interior by reflecting them back if they cross the boundary
template <int DIM>
static void keep_interior_point_inside(Point<DIM>& p) {
    // Reflect each axis
    auto reflect_axis = [&](int d) {
        if (p.c[d] < 0.0) p.c[d] = -p.c[d];
        else if (p.c[d] > DOMAIN_SIZE[d]) p.c[d] = DOMAIN_SIZE[d] - (p.c[d] - DOMAIN_SIZE[d]);
    };
    StaticFor<DIM>::run(reflect_axis);

    // Enforce strict interiority (keep away from EPSILON capture zone)
    // This prevents an interior point from becoming a boundary point in the next iteration
    auto clear_axis = [&](int d) {
        if (p.c[d] <= EPSILON) p.c[d] = EPSILON * 2.0;
        if (p.c[d] >= DOMAIN_SIZE[d] - EPSILON) p.c[d] = DOMAIN_SIZE[d] - EPSILON * 2.0;
    };
    StaticFor<DIM>::run(clear_axis);
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

// Helper to create a point with a unique ID based on grid indices
// type: 0=Interior, 1=Boundary/Corner
// idx: Grid indices (can be up to 2 billion)
template <int DIM>
static Point<DIM> create_point_with_unique_hash_id(const double *x, const int *idx, int type);

template <>
Point<2> create_point_with_unique_hash_id<2>(const double *x, const int *idx, int type) {
    int ix = idx[0], iy = idx[1];
    // Pack: [Type: 2 bits] [ix: 31 bits] [iy: 31 bits]
    // This guarantees uniqueness as long as (ix, iy) are unique within a type.
    uint64_t id = ((uint64_t)(type & 0x3) << 62) |
                  ((uint64_t)(ix & 0x7FFFFFFF) << 31) |
                  ((uint64_t)(iy & 0x7FFFFFFF));
    return Point<2>(x, id);
}

// Grid index used for the far (right/top) walls in the unique hash id
template <int DIM>
static int far_wall_grid_index();

template <>
int far_wall_grid_index<2>() {
    return MAX_GRID_IDX;
}

// Seed for the placement of the interior point of grid cell idx
template <int DIM>
static uint64_t interior_seed(const int *idx);

template <>
uint64_t interior_seed<2>(const int *idx) {
    int ix = idx[0], iy = idx[1];
    // FIX: Use bit-packing to guarantee unique seed for every (ix, iy) pair.
    // Previous hash_combine method had collisions for certain integer pairs.
    uint64_t h = ((uint64_t)(uint32_t)ix << 32) | (uint32_t)iy;
    h = splitmix64(h);
    return h;
}

// ~~~~~~~~~~~~~~~~~

// Applies deterministic jitter
template <int DIM>
static void apply_jitter(std::vector<Point<DIM> >& points, double amount, int seed_offset) {
    for (size_t i = 0; i < points.size(); ++i) {
        // Hash based on unique hash ID + iters Offset
        // This ensures that even if the point moves, the jitter sequence is deterministic
        uint64_t h = points[i].unique_hash_id;

        // Mix in the iters/seed offset to ensure different jitter each step
        h ^= seed_offset + 0x9e3779b9 + (h << 6) + (h >> 2);

        RngState rng = {h};

        // One draw per axis, x first
        double jitter[DIM];
        for (int d = 0; d < DIM; ++d) {
            jitter[d] = (next_double(rng) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;
        }

        // CRITICAL: Boundary nodes effectively ignore perpendicular jitter here
        bool was_boundary = apply_boundary_constraint(points[i], jitter);

        Point<DIM> candidate = points[i];
        for (int d = 0; d < DIM; ++d) candidate.c[d] += jitter[d];

        if (!was_boundary) {
            keep_interior_point_inside(candidate);
            points[i] = candidate;
        } else if (boundary_move_valid(points[i], candidate)) {
            // Reject tangential jitter that would take a wall point into a corner
            points[i] = candidate;
        }
    }
}

// ~~~~~~~~~~~~~~~~~

// Delaunay triangulation of the points, returned in the order the backend produces them
template <int DIM>
static std::vector<Simplex<DIM> > triangulation(const std::vector<Point<DIM> >& points);

// Wrapper for Triangle library - https://www.cs.cmu.edu/~quake/triangle.html
template <>
std::vector<Simplex<2> > triangulation<2>(const std::vector<Point<2> >& points) {
    struct triangulateio in;
    struct triangulateio out;

    // Initialize structures
    InitInput_Triangle(&in);
    InitOutput_Triangle(&out);

    in.numberofpoints = points.size();
    in.pointlist = new double[in.numberofpoints * 2];

    for (size_t i = 0; i < points.size(); ++i) {
        in.pointlist[i * 2] = points[i].c[0];
        in.pointlist[i * 2 + 1] = points[i].c[1];
    }

    // z: zero-based indexing, Q: quiet. We only use the triangle list, so don't
    // ask for the edge list (e), which Triangle would build and we'd just free
    char args[32];
    (void)PetscStrncpy(args, "zQ", sizeof(args));

    triangulate(args, &in, &out, NULL);

    const PetscInt numCells = out.numberoftriangles;
    std::vector<Simplex<2> > triangles;
    triangles.reserve(numCells);

    for (int i = 0; i < numCells; ++i) {
        Simplex<2> t;
        t.v[0] = (int)out.trianglelist[i * 3 + 0];
        t.v[1] = (int)out.trianglelist[i * 3 + 1];
        t.v[2] = (int)out.trianglelist[i * 3 + 2];
        triangles.push_back(t);
    }

    delete[] in.pointlist;
    FiniOutput_Triangle(&out);

    return triangles;
}

// ~~~~~~~~~~~~~~~~~

// Helper: Calculate minimum angle (degrees) of a triangle
static double clamp_val(double v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); }

// Helper: Calculate max cosine of a triangle (proxy for min angle)
// Returns 1.0 for degenerate triangles (worst case, angle 0)
static double get_max_cosine_tri(const Point<2>& a, const Point<2>& b, const Point<2>& c) {
    double ab_sq = (a.c[0]-b.c[0])*(a.c[0]-b.c[0]) + (a.c[1]-b.c[1])*(a.c[1]-b.c[1]);
    double bc_sq = (b.c[0]-c.c[0])*(b.c[0]-c.c[0]) + (b.c[1]-c.c[1])*(b.c[1]-c.c[1]);
    double ca_sq = (c.c[0]-a.c[0])*(c.c[0]-a.c[0]) + (c.c[1]-a.c[1])*(c.c[1]-a.c[1]);

    double ab = std::sqrt(ab_sq);
    double bc = std::sqrt(bc_sq);
    double ca = std::sqrt(ca_sq);

    if (ab < TOL_LEN || bc < TOL_LEN || ca < TOL_LEN) return 1.0; // Degenerate

    // Law of Cosines: cos A = (b^2 + c^2 - a^2) / 2bc
    double cos_a = (ab_sq + ca_sq - bc_sq) / (2.0 * ab * ca);
    double cos_b = (ab_sq + bc_sq - ca_sq) / (2.0 * ab * bc);
    double cos_c = (ca_sq + bc_sq - ab_sq) / (2.0 * ca * bc);

    return std::max({cos_a, cos_b, cos_c});
}

// Simplex kernels. p holds pointers to the DIM+1 vertices in simplex order

// Unsigned volume (area in 2D)
template <int DIM>
static double simplex_volume(const Point<DIM> *const *p);

template <>
double simplex_volume<2>(const Point<2> *const *p) {
    const Point<2>& p0 = *p[0];
    const Point<2>& p1 = *p[1];
    const Point<2>& p2 = *p[2];
    // volume = 0.5 * |(x1-x0)(y2-y0) - (y1-y0)(x2-x0)|
    return 0.5 * std::abs((p1.c[0] - p0.c[0])*(p2.c[1] - p0.c[1]) - (p1.c[1] - p0.c[1])*(p2.c[0] - p0.c[0]));
}

// Centroid
template <int DIM>
static void simplex_centroid(const Point<DIM> *const *p, double *centroid);

template <>
void simplex_centroid<2>(const Point<2> *const *p, double *centroid) {
    const Point<2>& p0 = *p[0];
    const Point<2>& p1 = *p[1];
    const Point<2>& p2 = *p[2];
    centroid[0] = (p0.c[0] + p1.c[0] + p2.c[0]) / 3.0;
    centroid[1] = (p0.c[1] + p1.c[1] + p2.c[1]) / 3.0;
}

// Squared length of the vector d
template <int DIM>
static double dist_sq(const double *d);

template <>
double dist_sq<2>(const double *d) {
    double dx = d[0], dy = d[1];
    return dx*dx + dy*dy;
}

// Lloyd's quality measure of simplex tri of the star of point i, lower is better, with p
// standing in for point i (its current position or a candidate move). The other vertices are
// taken in simplex order. In 2D the max cosine of the triangle (minimising it is the same as
// maximising the minimum angle, without an acos)
template <int DIM>
static double star_simplex_badness(const Point<DIM>& p, int i, const Simplex<DIM>& tri, const std::vector<Point<DIM> >& points);

template <>
inline double star_simplex_badness<2>(const Point<2>& p, int i, const Simplex<2>& tri, const std::vector<Point<2> >& points) {
    Point<2> p1, p2;
    if (tri.v[0] == i) { p1 = points[tri.v[1]]; p2 = points[tri.v[2]]; }
    else if (tri.v[1] == i) { p1 = points[tri.v[0]]; p2 = points[tri.v[2]]; }
    else { p1 = points[tri.v[0]]; p2 = points[tri.v[1]]; }
    return get_max_cosine_tri(p, p1, p2);
}

// The edges of a simplex as pairs of local vertex numbers, in the order the spring
// relaxation visits them. Returns the number of edges
template <int DIM>
static int get_simplex_edges(const int (*&edges)[2]);

template <>
int get_simplex_edges<2>(const int (*&edges)[2]) {
    static const int table[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    edges = table;
    return 3;
}

// ~~~~~~~~~~~~~~~~~

// Lloyd-smoothing - only smooths points within a certain box
// that way we can lock the outermost points in the halo, so the halo
// doesn't shrink inwards
template <int DIM>
static void relax_points_lloyd(std::vector<Point<DIM> >& points, const std::vector<Simplex<DIM> >& triangles, const double *min_safe, const double *max_safe) {
    int n = points.size();

    // Accumulators for volume-Weighted Centroids, one array per axis
    std::vector<double> wc[DIM];
    for (int d = 0; d < DIM; ++d) wc[d].assign(n, 0.0);
    std::vector<double> w_sum(n, 0.0);

    // CSR format for point-to-triangle adjacency
    std::vector<int> tri_count(n, 0);
    for (const auto& t : triangles) {
        for (int a = 0; a <= DIM; ++a) tri_count[t.v[a]]++;
    }

    std::vector<int> tri_offset(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        tri_offset[i + 1] = tri_offset[i] + tri_count[i];
    }

    std::vector<int> tri_data(tri_offset[n]);
    std::fill(tri_count.begin(), tri_count.end(), 0);

    for (size_t k = 0; k < triangles.size(); ++k) {
        const auto& t = triangles[k];
        const Point<DIM> *p[DIM + 1];
        for (int a = 0; a <= DIM; ++a) p[a] = &points[t.v[a]];

        // 1. Calculate Triangle Centroid
        double centroid[DIM];
        simplex_centroid<DIM>(p, centroid);

        // 2. Calculate Triangle volume
        double volume = simplex_volume<DIM>(p);

        // 3. Accumulate weighted centroid for all points of this triangle
        auto accumulate_vertex = [&](int a) {
            int v = t.v[a];
            auto accumulate_axis = [&](int d) { wc[d][v] += volume * centroid[d]; };
            StaticFor<DIM>::run(accumulate_axis);
            w_sum[v] += volume;
        };
        StaticFor<DIM + 1>::run(accumulate_vertex);

        for (int a = 0; a <= DIM; ++a) tri_data[tri_offset[t.v[a]] + tri_count[t.v[a]]++] = k;
    }
    tri_count.clear();
    std::vector<int>().swap(tri_count);

    // Define candidate relaxation factors to test per point
    std::vector<double> candidate_factors = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    for (int i=0; i<n; ++i) {
        if (w_sum[i] < TOL_VOLUME) continue;

        // Update if in valid safe zone (ignoring deep ghost layers)
        bool in_safe_zone = true;
        for (int d = 0; d < DIM; ++d) {
            if (!(points[i].c[d] > min_safe[d] && points[i].c[d] < max_safe[d])) in_safe_zone = false;
        }
        if (in_safe_zone) {

            // Target is the volume-weighted average of surrounding triangle centroids
            // Calculate the full vector to the centroid
            double full_d[DIM];
            for (int d = 0; d < DIM; ++d) {
                double target = wc[d][i] / w_sum[i];
                full_d[d] = target - points[i].c[d];
            }

            // Use CSR data for triangle lookup
            int start = tri_offset[i];
            int end = tri_offset[i + 1];

            // Minimize the maximum cosine is the same as maximizing the minimum angle, but without
            // needing an acos which is expensive
            double current_max_cos = -2.0;
            for (int j = start; j < end; ++j) {
                int t_idx = tri_data[j];
                const auto& tri = triangles[t_idx];
                double mc = star_simplex_badness<DIM>(points[i], i, tri, points);
                if (mc > current_max_cos) current_max_cos = mc;
            }

            Point<DIM> best_candidate = points[i];
            double best_max_cos = current_max_cos;

            // Test each relaxation factor
            for (double factor : candidate_factors) {
                double delta[DIM];
                for (int d = 0; d < DIM; ++d) delta[d] = full_d[d] * factor;

                // Boundary Projection
                // We need to be careful not to modify points[i] permanently yet.
                // Let's create a temp point for the constraint check.
                Point<DIM> temp_p = points[i];
                bool was_boundary = apply_boundary_constraint(temp_p, delta);
                Point<DIM> candidate = temp_p;
                for (int d = 0; d < DIM; ++d) candidate.c[d] += delta[d];

                if (!was_boundary) {
                    keep_interior_point_inside(candidate);
                } else if (!boundary_move_valid(temp_p, candidate)) {
                    // Reject candidates that would take a wall point into a corner
                    continue;
                }

                // Inline max cosine calculation using CSR
                double candidate_max_cos = -2.0;
                for (int j = start; j < end; ++j) {
                    int t_idx = tri_data[j];
                    const auto& tri = triangles[t_idx];
                    double mc = star_simplex_badness<DIM>(candidate, i, tri, points);
                    if (mc > candidate_max_cos) candidate_max_cos = mc;
                }

                if (candidate_max_cos < best_max_cos) {
                    best_max_cos = candidate_max_cos;
                    best_candidate = candidate;
                }
            }
            points[i] = best_candidate;
        }
    }
    tri_data.clear();
    tri_offset.clear();
    for (int d = 0; d < DIM; ++d) wc[d].clear();
    w_sum.clear();
}

// Spring-Force Relaxation
// Moves points to minimize edge length deviation from TARGET_EDGE_LENGTH
template <int DIM>
static void relax_points_spring(std::vector<Point<DIM> >& points, const std::vector<Simplex<DIM> >& triangles, const double *min_safe, const double *max_safe) {
    int n = points.size();

    std::vector<double> force[DIM]; // One array per axis
    for (int d = 0; d < DIM; ++d) force[d].assign(n, 0.0);
    std::vector<int> valence(n, 0); // Count neighbors to scale force

    // Iterate over edges (implicitly via triangles)
    // We need to be careful not to double-count edges, but for a force sum it just changes the magnitude scale.
    // Iterating triangles is fine if we scale appropriately.
    const int (*edges)[2];
    const int num_edges = get_simplex_edges<DIM>(edges);

    for (const auto& t : triangles) {
        for (int i = 0; i < num_edges; ++i) {
            int idx1 = t.v[edges[i][0]];
            int idx2 = t.v[edges[i][1]];

            double d[DIM];
            for (int a = 0; a < DIM; ++a) d[a] = points[idx2].c[a] - points[idx1].c[a];
            double dist = std::sqrt(dist_sq<DIM>(d));

            if (dist < 1e-14) continue;

            // Spring Force: F = k * (current_len - target_len)
            // We want to PUSH if too close (dist < target), PULL if too far (dist > target).
            // Direction for idx1: towards idx2.
            // If dist < target (compressed), (dist - target) is negative. Force is away from idx2. Correct.

            double force_mag = (dist - TARGET_EDGE_LENGTH);

            // Normalize direction
            double f[DIM];
            for (int a = 0; a < DIM; ++a) {
                double na = d[a] / dist;
                f[a] = force_mag * na;
            }

            // Apply to idx1
            for (int a = 0; a < DIM; ++a) force[a][idx1] += f[a];
            valence[idx1]++;

            // Apply opposite to idx2
            for (int a = 0; a < DIM; ++a) force[a][idx2] -= f[a];
            valence[idx2]++;
        }
    }

    // Time step / Damping factor
    // 0.1 is a safe starting point for explicit integration
    double dt = 0.2;

    for (int i=0; i<n; ++i) {
        // Only move if we have neighbors
        if (valence[i] == 0) continue;

        // Update if in valid safe zone (ignoring deep ghost layers)
        bool in_safe_zone = true;
        for (int d = 0; d < DIM; ++d) {
            if (!(points[i].c[d] > min_safe[d] && points[i].c[d] < max_safe[d])) in_safe_zone = false;
        }
        if (in_safe_zone) {

            // Average the force by valence to keep scaling consistent
            // (This effectively makes it "force per neighbor")
            double delta[DIM];
            for (int d = 0; d < DIM; ++d) {
                double f = force[d][i] / valence[i];
                delta[d] = f * dt; // Move proportional to force
            }

            // Apply constraints and move
            Point<DIM> temp_p = points[i];
            bool was_boundary = apply_boundary_constraint(temp_p, delta);

            Point<DIM> candidate = temp_p;
            for (int d = 0; d < DIM; ++d) candidate.c[d] += delta[d];

            if (!was_boundary) {
                keep_interior_point_inside(candidate);
            } else if (!boundary_move_valid(temp_p, candidate)) {
                // Reject a tangential move that would take a wall point into a corner
                continue;
            }

            points[i] = candidate;
        }
    }
    for (int d = 0; d < DIM; ++d) force[d].clear();
    valence.clear();
}

// ~~~~~~~~~~~~~~~~~

// Compute valence for each point and return sorted unique edge list
// The edge list uses local indices into the points vector
template <int DIM>
static void ComputeValenceAndEdges(const std::vector<Point<DIM> >& points,
                                   const std::vector<Simplex<DIM> >& triangles,
                                   std::vector<int>& valence,
                                   std::vector<std::pair<int, int>>& unique_edges) {
    size_t n = points.size();
    valence.assign(n, 0);

    const int (*simplex_edges)[2];
    const int num_simplex_edges = get_simplex_edges<DIM>(simplex_edges);

    // Collect all edges as (min_idx, max_idx)
    std::vector<std::pair<int, int>> edges;
    edges.reserve(triangles.size() * num_simplex_edges);

    for (const auto& t : triangles) {
        for (int i = 0; i < num_simplex_edges; ++i) {
            int va = t.v[simplex_edges[i][0]], vb = t.v[simplex_edges[i][1]];
            edges.emplace_back(std::min(va, vb), std::max(va, vb));
        }
    }

    // Sort to find unique edges
    std::sort(edges.begin(), edges.end());

    // Extract unique edges and count valence
    unique_edges.clear();
    unique_edges.reserve(edges.size() / 2); // Approximate: most edges shared by 2 triangles

    if (!edges.empty()) {
        unique_edges.push_back(edges[0]);
        valence[edges[0].first]++;
        valence[edges[0].second]++;

        for (size_t i = 1; i < edges.size(); ++i) {
            if (edges[i] != edges[i-1]) {
                unique_edges.push_back(edges[i]);
                valence[edges[i].first]++;
                valence[edges[i].second]++;
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~

// // Helper to remove duplicates from cloud
// static void remove_duplicates(std::vector<Point>& points) {
//     if (points.empty()) return;

//     // Use strict lexicographical sort.
//     std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) {
//         if (a.x != b.x) return a.x < b.x;
//         if (a.y != b.y) return a.y < b.y;
//         return a.unique_hash_id < b.unique_hash_id; // Tie-breaker for absolute stability
//     });

//     std::vector<Point> unique_points;
//     unique_points.reserve(points.size());
//     unique_points.push_back(points[0]);

//     for (size_t i = 1; i < points.size(); ++i) {
//         const Point& prev = unique_points.back();
//         const Point& curr = points[i];

//         // Check distance with tolerance
//         double dist_sq = (prev.x - curr.x)*(prev.x - curr.x) + (prev.y - curr.y)*(prev.y - curr.y);

//         // Only keep if distance is physically significant relative to mesh size
//         if (dist_sq > TOL_LEN_SQ) {
//             unique_points.push_back(curr);
//         }
//     }
//     points = unique_points;
// }

// ~~~~~~~~~~~~~~~~~

// Boundary features (corners, walls, and in 3D edges and faces) generated explicitly, in
// generation order. Each entry gives, per axis, whether the feature sits on the low wall,
// the high wall or is free (spans the axis). Returns the number of features
enum { AXIS_LOW = 0, AXIS_HIGH = 1, AXIS_FREE = 2 };

template <int DIM>
static int get_boundary_features(const int (*&features)[DIM]);

template <>
int get_boundary_features<2>(const int (*&features)[2]) {
    static const int table[8][2] = {
        // 1. Corners: (0,0), (1,0), (0,1), (1,1)
        {AXIS_LOW, AXIS_LOW}, {AXIS_HIGH, AXIS_LOW}, {AXIS_LOW, AXIS_HIGH}, {AXIS_HIGH, AXIS_HIGH},
        // 2. Walls: Left (x=0), Right (x=DOMAIN_SIZE[0]), Bottom (y=0), Top (y=DOMAIN_SIZE[1])
        {AXIS_LOW, AXIS_FREE}, {AXIS_HIGH, AXIS_FREE}, {AXIS_FREE, AXIS_LOW}, {AXIS_FREE, AXIS_HIGH}};
    features = table;
    return 8;
}

// Builds a tile, creates points and triangulates
template <int DIM>
static void process_tile(MPI_Comm comm, int final_smooth_its, const int *tile,
                  std::vector<Point<DIM> >& points_on_owned_triangles_and_orphans,
                  std::vector<Simplex<DIM> >& triangles_owned) {

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    double tile_s[DIM], t_min[DIM], t_max[DIM];
    for (int d = 0; d < DIM; ++d) {
        tile_s[d] = DOMAIN_SIZE[d] / TILE_DIM[d];
        t_min[d] = tile[d] * tile_s[d]; t_max[d] = (tile[d] + 1) * tile_s[d];
    }

    // UNIFIED PADDING LOGIC
    // We generate a halo large enough to absorb the boundary effects of annealing.
    // The distortion from the boundary travels approx 1 edge per iter.
    // We add +8 for safety
    double pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8);

    // Safety Check: Ensure the required halo doesn't exceed the tile size.
    // In a domain decomposition, needing a halo larger than the subdomain
    // implies we need data from neighbors-of-neighbors, which is inefficient/complex.
    // Only axes split between ranks matter: along an axis with a single tile there are
    // no neighbours, the halo is cut off by the domain walls, and ResolveBoundaryOwnership
    // just treats every point as a boundary candidate along that axis (extra communication
    // only). With a single tile on every axis there is nothing to check.
    bool any_split = false;
    double min_dim = 0.0;
    for (int d = 0; d < DIM; ++d) {
        if (TILE_DIM[d] > 1) {
            if (!any_split) min_dim = tile_s[d];
            else min_dim = std::min(min_dim, tile_s[d]);
            any_split = true;
        }
    }
    if (any_split) {
        if (pad > min_dim/2.0) {
            std::cerr << "Error: Annealing iters (" << ANNEAL_ITERS << ") require a halo of "
                      << pad << ", which is more than half the tile size of " << min_dim << " along an axis split between ranks.\n"
                      << "Reduce ANNEAL_ITERS or increase tile size (by having fewer tiles or more points per tile).\n";
            std::exit(EXIT_FAILURE);
        }
    }

    double search_min[DIM], search_max[DIM], interior_min[DIM], interior_max[DIM];
    for (int d = 0; d < DIM; ++d) {
        search_min[d] = t_min[d] - pad; search_max[d] = t_max[d] + pad;
        interior_min[d] = t_min[d] + pad; interior_max[d] = t_max[d] - pad;
    }

    std::vector<Point<DIM> > points_with_halos;

    // 1. EXPLICIT CORNERS and 2. EXPLICIT BOUNDARY GENERATION (Edges only)
    // Walk the boundary feature table: corners first, then the walls.
    // Add points if they fall within search box.
    // We use type=1 (Boundary) and grid indices 0 on the low wall and max_idx on the high wall
    // (0,0) -> 0,0
    // (1,0) -> Max,0
    // (0,1) -> 0,Max
    // (1,1) -> Max,Max
    // Wall points exclude corners (EPSILON checks) to avoid duplication with explicit corners.
    int max_idx = far_wall_grid_index<DIM>(); // Just a large number for the "1.0" side

    // Wall points are spaced evenly, round(length / TARGET_EDGE_LENGTH) pieces per wall, so the
    // last one is a full spacing from the far corner. Stepping by TARGET_EDGE_LENGTH could put a
    // wall point arbitrarily close to that corner (0.01 L for L = 0.0099), where
    // boundary_move_valid won't let it move out, leaving a sliver triangle. When the wall length
    // is a whole multiple of TARGET_EDGE_LENGTH the spacing (and so the mesh) is unchanged.
    double wall_d[DIM];
    for (int d = 0; d < DIM; ++d) {
        wall_d[d] = DOMAIN_SIZE[d] / std::max(1.0, std::round(DOMAIN_SIZE[d] / TARGET_EDGE_LENGTH));
    }

    const int (*features)[DIM];
    const int num_features = get_boundary_features<DIM>(features);
    for (int f = 0; f < num_features; ++f) {
        // Is the fixed part of the feature within the search box
        bool in_box = true;
        for (int d = 0; d < DIM; ++d) {
            if (features[f][d] == AXIS_LOW) {
                if (!(search_min[d] <= EPSILON && search_max[d] >= -EPSILON)) in_box = false;
            } else if (features[f][d] == AXIS_HIGH) {
                if (!(search_min[d] <= DOMAIN_SIZE[d] + EPSILON && search_max[d] >= DOMAIN_SIZE[d] - EPSILON)) in_box = false;
            }
        }
        if (!in_box) continue;

        // Range of wall point indices along the free axes, a single position on the fixed ones
        int lo[DIM], hi[DIM], idx[DIM];
        for (int d = 0; d < DIM; ++d) {
            if (features[f][d] == AXIS_FREE) {
                lo[d] = floor(search_min[d] / wall_d[d]);
                hi[d] = ceil(search_max[d] / wall_d[d]);
            } else {
                lo[d] = 0; hi[d] = 0;
            }
            idx[d] = lo[d];
        }
        do {
            double x[DIM];
            int grid_idx[DIM];
            bool keep = true;
            for (int d = 0; d < DIM; ++d) {
                if (features[f][d] == AXIS_LOW) {
                    x[d] = 0.0; grid_idx[d] = 0;
                } else if (features[f][d] == AXIS_HIGH) {
                    x[d] = DOMAIN_SIZE[d]; grid_idx[d] = max_idx;
                } else {
                    x[d] = idx[d] * wall_d[d];
                    grid_idx[d] = idx[d];
                    if (!(x[d] > EPSILON && x[d] < DOMAIN_SIZE[d] - EPSILON)) keep = false;
                }
            }
            if (keep) {
                points_with_halos.push_back(create_point_with_unique_hash_id<DIM>(x, grid_idx, 1));
            }
        } while (next_index<DIM>(idx, lo, hi));
    }

    // 3. INTERIOR GENERATION
    int min_i[DIM], max_i[DIM];
    bool interior_empty = false;
    for (int d = 0; d < DIM; ++d) {
        min_i[d] = floor(search_min[d] / TARGET_EDGE_LENGTH);
        max_i[d] = ceil(search_max[d] / TARGET_EDGE_LENGTH);
        if (min_i[d] > max_i[d]) interior_empty = true;
    }

    // Distance from wall to reject interior points
    // Must be larger than max jitter to prevent collision with boundary
    // Max jitter is START_JITTER * TARGET_EDGE_LENGTH.
    // We need a larger safety margin to prevent slivers.
    double exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.25);

    // Let's just make sure we don't have any hash collisions
    //std::set<uint64_t> existing_hashes;

    // Visits the cells with the last axis outermost and x innermost
    int cell[DIM];
    for (int d = 0; d < DIM; ++d) cell[d] = min_i[d];
    if (!interior_empty) do {
        // Deterministic RNG based on global grid index
        RngState rng = {interior_seed<DIM>(cell)};

        // Restrict random range to [0.1, 0.9]
        // This keeps points centered in their cells and guarantees separation.
        double r[DIM];
        for (int d = 0; d < DIM; ++d) r[d] = 0.1 + next_double(rng) * 0.8;

        double x[DIM];
        for (int d = 0; d < DIM; ++d) x[d] = (cell[d] + r[d]) * TARGET_EDGE_LENGTH;

        // Rejection Sampling for Exclusion Zone
        bool rejected = false;
        for (int d = 0; d < DIM; ++d) {
            if (x[d] < exclusion) rejected = true;
            if (x[d] > DOMAIN_SIZE[d] - exclusion) rejected = true;
        }
        if (rejected) continue;

        Point<DIM> p = create_point_with_unique_hash_id<DIM>(x, cell, 0);

        // // If hash is unique
        // if (existing_hashes.find(p.unique_hash_id) == existing_hashes.end())
        // {
        //    existing_hashes.insert(p.unique_hash_id);
        // }
        // else
        // {
        //    std::cerr << "Warning: Hash collision detected for point (" << cx << ", " << cy << ") ID: " << p.unique_hash_id << "\n";
        //    MPI_Abort(comm, EXIT_FAILURE);
        // }

        // Only add point if strictly away from boundaries
        // Type 0 (Interior)
        points_with_halos.push_back(p);
    } while (next_index<DIM>(cell, min_i, max_i));
    //existing_hashes.clear();

    // Remove any accidental duplicates (e.g. from corner/edge overlaps or precision issues)
    // remove_duplicates(points_with_halos);

    // 4. ITERATIONS
    // We relax all points that are strictly inside the generated cloud.
    // The outer hull acts as a fixed boundary condition.
    // We must freeze the outer rim of the generated cloud.
    // If we relax the very edge, it collapses inward due to lack of outer neighbors.
    // This collapse propagates errors inward.
    // We freeze a strip of width ~ 1.5 * spacing.

    // UPDATE: To prevent drift between ranks, we synchronize ALL points that are allowed to move.
    // The "Frozen Zone" is the outer rim of the halo (width ~1.5*L).
    // The "Active Zone" is everything inside that.
    // We set the sync_margin to cover the entire Active Zone.
    // This ensures that if a point moves, its position is synchronized across ranks.

    double frozen_width = TARGET_EDGE_LENGTH * 1.5;
    double sync_margin = pad - frozen_width;

    double s_min[DIM], s_max[DIM];
    for (int d = 0; d < DIM; ++d) {
        s_min[d] = t_min[d] - sync_margin;
        s_max[d] = t_max[d] + sync_margin;
    }

    std::vector<Simplex<DIM> > triangles_with_halos;
    double current_jitter = START_JITTER;

    // Jitter + triangulate + smooth loop
    for (int iter = 0; iter < ANNEAL_ITERS; ++iter) {
        apply_jitter(points_with_halos, current_jitter, iter);
        triangles_with_halos = triangulation<DIM>(points_with_halos);
        relax_points_lloyd(points_with_halos, triangles_with_halos, s_min, s_max);
        relax_points_spring(points_with_halos, triangles_with_halos, s_min, s_max);

        // Sync boundary points immediately to prevent divergence
        // We pass the calculated sync_margin to the function so it knows how far to look
        ResolveBoundaryOwnership(comm, points_with_halos, s_min, s_max, interior_min, interior_max, pad);
    }
    // Final smooth iterations without jitter
    for(int k=0; k<final_smooth_its; ++k) {
        triangles_with_halos = triangulation<DIM>(points_with_halos);
        relax_points_lloyd(points_with_halos, triangles_with_halos, s_min, s_max);
        relax_points_spring(points_with_halos, triangles_with_halos, s_min, s_max);

        // Sync boundary points immediately to prevent divergence
        ResolveBoundaryOwnership(comm, points_with_halos, s_min, s_max, interior_min, interior_max, pad);
    }

    // Final mesh
    triangles_with_halos = triangulation<DIM>(points_with_halos);

    // Resolve ownership before filtering
    ResolveBoundaryOwnership(comm, points_with_halos, s_min, s_max, interior_min, interior_max, pad);

    // Pre-allocate remapping array.
    // -1 indicates the point hasn't been added to the owned list yet.
    std::vector<int> local_to_owned_idx(points_with_halos.size(), -1);

    // We only keep triangles that are geometrically "owned" by this tile
    for (const auto& tri : triangles_with_halos) {

        // Robust Ownership Rule:
        // A triangle is owned by the rank that owns the triangle's "lowest" point.
        // We define "lowest" using the unique hash ID to ensure all ranks agree.
        // If multiple points are on the same rank, that rank definitely owns it.
        // If points are on different ranks, the one with the smallest ID decides.
        // Find which point has the smallest ID (they are unique)
        const Point<DIM> *min_p = &points_with_halos[tri.v[0]];
        for (int a = 1; a <= DIM; ++a) {
            const Point<DIM> *pa = &points_with_halos[tri.v[a]];
            if (pa->unique_hash_id < min_p->unique_hash_id) min_p = pa;
        }

        // Check if WE own this determining point
        int owner = get_owner_rank(*min_p, comm_size);

        if (owner == comm_rank) {
            Simplex<DIM> new_t;
            // Map local halo indices to owned indices directly
            for(int k=0; k<=DIM; ++k) {
                int local_idx = tri.v[k];
                if (local_to_owned_idx[local_idx] == -1) {
                    int new_idx = points_on_owned_triangles_and_orphans.size();
                    points_on_owned_triangles_and_orphans.push_back(points_with_halos[local_idx]);
                    local_to_owned_idx[local_idx] = new_idx;
                }
                new_t.v[k] = local_to_owned_idx[local_idx];
            }
            triangles_owned.push_back(new_t);
        }
    }
    triangles_with_halos.clear();
    // Explicitly delete memory
    std::vector<Simplex<DIM> >().swap(triangles_with_halos);

    // Explicitly add "orphaned" points that we own spatially.
    // It is possible to own a point spatially (it's in our tile) but NOT own any of the
    // triangles connected to it (due to the min_id rule giving them to neighbors).
    // We must still track this point so we can assign it a Global ID and answer requests.
    for (size_t i = 0; i < points_with_halos.size(); ++i) {
        const auto& p = points_with_halos[i];
        if (get_owner_rank(p, comm_size) == comm_rank) {
             if (local_to_owned_idx[i] == -1) {
                 int new_idx = points_on_owned_triangles_and_orphans.size();
                 points_on_owned_triangles_and_orphans.push_back(p);
                 local_to_owned_idx[i] = new_idx;
             }
        }
    }
    local_to_owned_idx.clear();
    points_with_halos.clear();
}

// ~~~~~~~~~~~~~~~~~

// Return a PETSc DM for the points and triangles passed in
template <int DIM>
static DM CreateDM(MPI_Comm comm, const std::vector<Point<DIM> >& points_on_owned_triangles_and_orphans, const std::vector<Simplex<DIM> >& triangles_owned) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    PetscInt neg_one = -1;

    PetscInt num_points_on_owned_triangles_and_orphans = points_on_owned_triangles_and_orphans.size();
    std::vector<PetscInt> global_ids(num_points_on_owned_triangles_and_orphans, neg_one);
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
        if (global_ids[i] == neg_one) {
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
    send_ids.clear();
    // Explicitly delete memory
    std::vector<MPI_Request>().swap(requests);
    std::vector<std::vector<uint64_t>>().swap(send_ids);

    // ---------------------------------------------------------
    // PROCESSING: Lookup Global IDs
    // ---------------------------------------------------------

    // We need a map for fast lookup between the unique hash id and the global id
    std::unordered_map<uint64_t, PetscInt> points_owned_l2g_map;
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
                // If we don't have the point the ranks disagree about the geometry,
                // error rather than silently handing back a bogus global id
                auto it = points_owned_l2g_map.find(recv_ids[r][k]);
                if (it == points_owned_l2g_map.end()) {
                    std::cerr << "Error: [Rank " << comm_rank << "] rank " << r << " asked for the global id of point "
                              << recv_ids[r][k] << ", which this rank does not own.\n";
                    MPI_Abort(comm, EXIT_FAILURE);
                }
                send_answers[r][k] = it->second;
            }
        }
    }
    recv_ids.clear();
    points_owned_l2g_map.clear();
    // Explicitly delete memory
    std::vector<std::vector<uint64_t>>().swap(recv_ids);
    std::unordered_map<uint64_t, PetscInt>().swap(points_owned_l2g_map);

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
    send_req_indices.clear();
    send_counts.clear();
    recv_counts.clear();
    recv_answers.clear();
    // Explicitly delete memory
    std::vector<std::vector<int>>().swap(send_req_indices);
    std::vector<int>().swap(send_counts);
    std::vector<int>().swap(recv_counts);
    std::vector<std::vector<PetscInt>>().swap(recv_answers);

    // 5. Build DMPlex
    PetscInt num_tris_owned = triangles_owned.size();
    std::vector<PetscInt> cells(num_tris_owned * (DIM + 1));
    // Global ids for all points on owned triangles
    for(int i=0; i<num_tris_owned; ++i) {
        for (int k = 0; k <= DIM; ++k) {
            cells[i*(DIM + 1) + k] = global_ids[triangles_owned[i].v[k]];
        }
    }
    global_ids.clear();
    // Explicitly delete memory
    std::vector<PetscInt>().swap(global_ids);

    // Prepare coordinates for DMPlexCreateFromCellListParallelPetsc
    // points_on_owned_triangles_and_orphans contains ghosts (points of owned triangles that are owned by neighbors).
    // PETSc expects 'numPoints' to be the count of locally owned points,
    // and 'coords' to be the coordinates of those specific points
    std::vector<PetscReal> coords_points_owned;
    coords_points_owned.reserve(num_points_owned * DIM);

    for(int i=0; i<num_points_on_owned_triangles_and_orphans; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            for (int d = 0; d < DIM; ++d) coords_points_owned.push_back(points_on_owned_triangles_and_orphans[i].c[d]);
        }
    }

    // Build the DM - DMPlexCreateFromCellListParallelPetsc creates it, so we
    // must not DMCreate one ourselves first or it leaks
    DM dm;
    PetscErrorCode ierr;

    PetscInt dim = DIM;
    PetscInt num_corners = DIM + 1;
    ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, num_tris_owned, num_points_owned, PETSC_DECIDE, \
         num_corners, PETSC_TRUE, cells.data(), dim, coords_points_owned.data(), NULL, NULL, &dm);

    // The DM is already distributed, we don't want to call parmetis (or equivalent)
    // by default as it's very memory heavy
    ierr = DMPlexDistributeSetDefault(dm, PETSC_FALSE);
    // Have to include or -dm_view doesn't work on command line
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");
    (void)ierr;

    return dm;
}

// ~~~~~~~~~~~~~~~~~

// The domain size is stored on each DM we create (and on every refinement of it),
// so the refinement hook labels against that mesh's domain rather than the globals,
// which are overwritten by any later call to GenerateBoxMeshDM
struct BoxDomain {
    int dim;
    double size[3];
};
static const char BOX_DOMAIN_KEY[] = "BoxMeshDM_domain";

static PetscErrorCode SetBoxDomain(DM dm, int dim, const double *size) {
    BoxDomain *domain;
    PetscFunctionBeginUser;
    PetscCall(PetscNew(&domain));
    domain->dim = dim;
    for (int d = 0; d < 3; ++d) domain->size[d] = (d < dim) ? size[d] : 0.0;
    // The container (and domain) are freed when the DM is destroyed
    PetscCall(PetscObjectContainerCompose((PetscObject)dm, BOX_DOMAIN_KEY, domain, PetscCtxDestroyDefault));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Number of walls, which are label values 1..num_walls
template <int DIM>
static PetscInt num_walls();

template <>
PetscInt num_walls<2>() {
    return 4;
}

// Label value of the wall a point lies on (0 if none)
// Values: 1=Bottom, 2=Right, 3=Top, 4=Left
template <int DIM>
static PetscInt classify_point_wall(const double *xy, const double *domain_size);

template <>
PetscInt classify_point_wall<2>(const double *xy, const double *domain_size) {
    double x = xy[0];
    double y = xy[1];
    double domain_width = domain_size[0];
    double domain_height = domain_size[1];

    PetscInt val = 0;
    // Priority for corners: Bottom > Right > Top > Left
    if (std::abs(y) < EPSILON) val = 1;              // Bottom
    else if (std::abs(x - domain_width) < EPSILON) val = 2; // Right
    else if (std::abs(y - domain_height) < EPSILON) val = 3; // Top
    else if (std::abs(x) < EPSILON) val = 4;         // Left
    return val;
}

// Label value of the wall a facet (height 1 point) lies on (0 if none)
template <int DIM>
static PetscErrorCode facet_wall_value(DM dm, PetscInt e, PetscSection coordSection, const PetscScalar *coords,
                                       const double *domain_size, PetscInt *val);

template <>
PetscErrorCode facet_wall_value<2>(DM dm, PetscInt e, PetscSection coordSection, const PetscScalar *coords,
                                   const double *domain_size, PetscInt *val) {
    PetscInt num_points;
    const PetscInt *points;
    PetscFunctionBeginUser;
    PetscCall(DMPlexGetConeSize(dm, e, &num_points));
    PetscCall(DMPlexGetCone(dm, e, &points));

    // Compute centroid of edge to determine boundary
    double cx = 0, cy = 0;
    int count = 0;

    for(int i=0; i<num_points; ++i) {
        PetscInt v = points[i];
        PetscInt off, dof;
        PetscCall(PetscSectionGetDof(coordSection, v, &dof));
        if (dof > 0) {
            PetscCall(PetscSectionGetOffset(coordSection, v, &off));
            cx += coords[off];
            cy += coords[off+1];
            count++;
        }
    }

    *val = 0;
    if (count > 0) {
        cx /= count;
        cy /= count;

        double centroid[2] = {cx, cy};
        *val = classify_point_wall<2>(centroid, domain_size);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Label boundary faces and vertices based on geometric location
template <int DIM>
static void LabelBoundaries(DM dm, const double *domain_size) {

    // Create or get "Face Sets" label (standard name for boundary markers)
    // Values: 1=Bottom, 2=Right, 3=Top, 4=Left
    DMLabel label;
    PetscCallVoid(DMGetLabel(dm, "Face Sets", &label));
    if (!label) {
        PetscCallVoid(DMCreateLabel(dm, "Face Sets"));
        PetscCallVoid(DMGetLabel(dm, "Face Sets", &label));
    } else {
        // Clear existing strata to avoid stale labels
        for (PetscInt w = 1; w <= num_walls<DIM>(); ++w) {
            PetscCallVoid(DMLabelClearStratum(label, w));
        }
    }

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
            double x[DIM];
            for (int d = 0; d < DIM; ++d) x[d] = coords[off + d];

            PetscInt val = classify_point_wall<DIM>(x, domain_size);

            if (val != 0) PetscCallVoid(DMLabelSetValue(label, v, val));
        }
    }
    PetscCallVoid(DMPlexLabelComplete(dm, label));

    // 2. Label Edges/Facets (Height 1 in 2D = codimension-1 = facets)
    // Need to label to BOTH "Face Sets" and "markers"
    PetscInt eStart, eEnd;
    PetscCallVoid(DMPlexGetHeightStratum(dm, 1, &eStart, &eEnd));

    for (PetscInt e = eStart; e < eEnd; ++e) {
        PetscInt val;
        PetscCallVoid(facet_wall_value<DIM>(dm, e, coordSection, coords, domain_size, &val));
        if (val != 0) {
            // Label to Face Sets (for boundary conditions)
            PetscCallVoid(DMLabelSetValue(label, e, val));
        }
    }
    PetscCallVoid(DMPlexLabelComplete(dm, label));

    // Also label edges to "markers" label
    PetscCallVoid(DMCreateLabel(dm, "markers"));
    DMLabel markersLabel;
    PetscCallVoid(DMGetLabel(dm, "markers", &markersLabel));

    for (PetscInt e = eStart; e < eEnd; ++e) {
        PetscInt val;
        PetscCallVoid(facet_wall_value<DIM>(dm, e, coordSection, coords, domain_size, &val));
        if (val != 0) {
            // Label to markers
            PetscCallVoid(DMLabelSetValue(markersLabel, e, val));
        }
    }

    PetscCallVoid(VecRestoreArrayRead(coordsVec, &coords));
    PetscCallVoid(DMPlexLabelComplete(dm, markersLabel));
}

// ~~~~~~~~~~~~~~~~~

// Refinement hook, re-labels the boundaries of the refined mesh
static PetscErrorCode RefineHook_LabelBoundaries(DM dm, DM dmf, void *ctx) {
    BoxDomain *domain = NULL;
    PetscInt dim;
    PetscFunctionBeginUser;
    (void)ctx;
    PetscCall(PetscObjectContainerQuery((PetscObject)dm, BOX_DOMAIN_KEY, &domain));
    PetscCheck(domain, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM has no BoxMeshDM domain attached");
    // Carry the domain down so further refinements can find it
    PetscCall(SetBoxDomain(dmf, domain->dim, domain->size));
    // Label the fine (refined) mesh
    PetscCall(DMGetDimension(dmf, &dim));
    if (dim == 2) LabelBoundaries<2>(dmf, domain->size);
    else if (dim == 3) LabelBoundaries<3>(dmf, domain->size);
    else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "BoxMeshDM refine hook on a DM of dimension %" PetscInt_FMT, dim);
    // Also add the hook to the refined mesh so further refinements work
    PetscCall(DMRefineHookAdd(dmf, RefineHook_LabelBoundaries, NULL, NULL));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// ~~~~~~~~~~~~~~~~~

// Check every ghost point (a vertex of an owned triangle that a neighbour owns) has
// bit-identical coordinates to the owner's copy. Each rank builds its owned triangles
// from its own copies of the ghost points, but only the owner's coordinates go into
// the DM, so a mismatch means the connectivity was built against geometry that is not
// in the DM. Returns the number of mismatches found on this rank.
template <int DIM>
static long CheckGhostCoordinates(MPI_Comm comm, const std::vector<Point<DIM> >& points_on_owned_triangles_and_orphans) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    size_t num_points = points_on_owned_triangles_and_orphans.size();

    // Ask the owner of each ghost point for its coordinates, identified by the unique hash id
    std::vector<std::vector<uint64_t>> send_ids(comm_size);
    std::vector<std::vector<int>>      send_req_indices(comm_size);
    for (size_t i = 0; i < num_points; ++i) {
        int owner = get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size);
        if (owner != comm_rank) {
            send_ids[owner].push_back(points_on_owned_triangles_and_orphans[i].unique_hash_id);
            send_req_indices[owner].push_back((int)i);
        }
    }

    // Exchange counts
    std::vector<int> send_counts(comm_size), recv_counts(comm_size);
    for(int r=0; r<comm_size; ++r) send_counts[r] = send_ids[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    // Exchange hash ids (requests)
    std::vector<std::vector<uint64_t>> recv_ids(comm_size);
    std::vector<MPI_Request> requests;
    requests.reserve(comm_size * 2);
    for(int r=0; r<comm_size; ++r) {
        if (recv_counts[r] > 0) {
            recv_ids[r].resize(recv_counts[r]);
            MPI_Request req;
            MPI_Irecv(recv_ids[r].data(), recv_counts[r], MPI_UINT64_T, r, 102, comm, &req);
            requests.push_back(req);
        }
    }
    for(int r=0; r<comm_size; ++r) {
        if (send_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_ids[r].data(), send_counts[r], MPI_UINT64_T, r, 102, comm, &req);
            requests.push_back(req);
        }
    }
    if (!requests.empty()) MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();
    // Explicitly delete memory
    std::vector<std::vector<uint64_t>>().swap(send_ids);

    // Answer with the coordinates of our owned points
    std::unordered_map<uint64_t, size_t> points_owned_map;
    for (size_t i = 0; i < num_points; ++i) {
        if (get_owner_rank(points_on_owned_triangles_and_orphans[i], comm_size) == comm_rank) {
            points_owned_map[points_on_owned_triangles_and_orphans[i].unique_hash_id] = i;
        }
    }

    std::vector<std::vector<double>> send_coords(comm_size);
    for(int r=0; r<comm_size; ++r) {
        if (recv_counts[r] > 0) {
            send_coords[r].resize(DIM * (size_t)recv_counts[r]);
            for(int k=0; k<recv_counts[r]; ++k) {
                auto it = points_owned_map.find(recv_ids[r][k]);
                if (it == points_owned_map.end()) {
                    std::cerr << "Error: [Rank " << comm_rank << "] rank " << r << " asked for the coordinates of point "
                              << recv_ids[r][k] << ", which this rank does not own.\n";
                    MPI_Abort(comm, EXIT_FAILURE);
                }
                for (int d = 0; d < DIM; ++d) {
                    send_coords[r][DIM * k + d] = points_on_owned_triangles_and_orphans[it->second].c[d];
                }
            }
        }
    }
    // Explicitly delete memory
    std::vector<std::vector<uint64_t>>().swap(recv_ids);
    std::unordered_map<uint64_t, size_t>().swap(points_owned_map);

    // Exchange coordinates (answers)
    std::vector<std::vector<double>> recv_coords(comm_size);
    for(int r=0; r<comm_size; ++r) {
        if (send_counts[r] > 0) {
            recv_coords[r].resize(DIM * (size_t)send_counts[r]);
            MPI_Request req;
            MPI_Irecv(recv_coords[r].data(), DIM * send_counts[r], MPI_DOUBLE, r, 103, comm, &req);
            requests.push_back(req);
        }
    }
    for(int r=0; r<comm_size; ++r) {
        if (recv_counts[r] > 0) {
            MPI_Request req;
            MPI_Isend(send_coords[r].data(), DIM * recv_counts[r], MPI_DOUBLE, r, 103, comm, &req);
            requests.push_back(req);
        }
    }
    if (!requests.empty()) MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    // Explicitly delete memory
    std::vector<MPI_Request>().swap(requests);
    std::vector<std::vector<double>>().swap(send_coords);

    // Compare bitwise - ranks must produce bit-identical coordinates, so no tolerance
    long mismatch_count = 0;
    int mismatch_print_count = 0;
    const int MAX_MISMATCH_PRINTS = 5;
    for(int r=0; r<comm_size; ++r) {
        for(int k=0; k<send_counts[r]; ++k) {
            const Point<DIM>& p = points_on_owned_triangles_and_orphans[send_req_indices[r][k]];
            double owner_c[DIM], local_c[DIM];
            for (int d = 0; d < DIM; ++d) {
                owner_c[d] = recv_coords[r][DIM * k + d];
                local_c[d] = p.c[d];
            }
            if (std::memcmp(owner_c, local_c, sizeof(local_c)) != 0) {
                mismatch_count++;
                if (mismatch_print_count < MAX_MISMATCH_PRINTS) {
                    // Full precision so a 1 ulp difference is visible, then restore the stream
                    std::streamsize old_precision = std::cout.precision(17);
                    std::cout << "[Rank " << comm_rank << "] GHOST COORDINATE MISMATCH: local (";
                    print_coords<DIM>(std::cout, local_c);
                    std::cout << ") vs owner (";
                    print_coords<DIM>(std::cout, owner_c);
                    std::cout << ") Owner: " << r
                              << " ID: " << p.unique_hash_id << "\n";
                    std::cout.precision(old_precision);
                    mismatch_print_count++;
                }
            }
        }
    }
    // Explicitly delete memory
    std::vector<std::vector<int>>().swap(send_req_indices);
    std::vector<std::vector<double>>().swap(recv_coords);
    std::vector<int>().swap(send_counts);
    std::vector<int>().swap(recv_counts);

    return mismatch_count;
}

// ~~~~~~~~~~~~~~~~~

// Local (and then global) sums/maxima gathered by CheckMeshIntegrity
struct IntegrityAccum {
    double total_volume;        // area in 2D
    double boundary_measure;    // boundary length in 2D
    long boundary_facet_count;  // boundary edges in 2D
    long bad_edge_count;
    double max_edge_len;
    double max_badness;         // largest cosine of any triangle angle in 2D, i.e. the smallest angle
    int bad_print_count;
};

// Integrity check thresholds
const double MAX_EDGE_RATIO = 3.0;
// Deliberately loose - smoothed meshes have minimum angles of ~15-30 degrees, this
// only catches slivers (e.g. a wall point stuck next to a corner)
const double MIN_ANGLE_DEG = 5.0;
const int MAX_BAD_PRINTS = 5;

// Accumulate the integrity measures of one owned simplex (p in simplex order)
template <int DIM>
static void integrity_accumulate_simplex(const Point<DIM> *const *p, int rank, int size, IntegrityAccum& acc);

template <>
void integrity_accumulate_simplex<2>(const Point<2> *const *p, int rank, int size, IntegrityAccum& acc) {
    const Point<2>& p0 = *p[0];
    const Point<2>& p1 = *p[1];
    const Point<2>& p2 = *p[2];
    const double THRESHOLD_LEN = TARGET_EDGE_LENGTH * MAX_EDGE_RATIO;

    // Edge Lengths
    double d01_sq = std::pow(p1.c[0]-p0.c[0], 2) + std::pow(p1.c[1]-p0.c[1], 2);
    double d12_sq = std::pow(p2.c[0]-p1.c[0], 2) + std::pow(p2.c[1]-p1.c[1], 2);
    double d20_sq = std::pow(p0.c[0]-p2.c[0], 2) + std::pow(p0.c[1]-p2.c[1], 2);

    double d01 = std::sqrt(d01_sq);
    double d12 = std::sqrt(d12_sq);
    double d20 = std::sqrt(d20_sq);

    acc.max_edge_len = std::max({acc.max_edge_len, d01, d12, d20});
    acc.max_badness = std::max(acc.max_badness, get_max_cosine_tri(p0, p1, p2));

    if (d01 > THRESHOLD_LEN || d12 > THRESHOLD_LEN || d20 > THRESHOLD_LEN) {
        acc.bad_edge_count++;
        if (acc.bad_print_count < MAX_BAD_PRINTS) {
            std::cout << "[Rank " << rank << "] BAD TRIANGLE: Edge len "
                      << std::max({d01, d12, d20}) << " vs target " << TARGET_EDGE_LENGTH << "\n";
            std::cout << "   P0: (" << p0.c[0] << ", " << p0.c[1] << ") Owner: " << get_owner_rank(p0, size) << " ID: " << p0.unique_hash_id << "\n";
            std::cout << "   P1: (" << p1.c[0] << ", " << p1.c[1] << ") Owner: " << get_owner_rank(p1, size) << " ID: " << p1.unique_hash_id << "\n";
            std::cout << "   P2: (" << p2.c[0] << ", " << p2.c[1] << ") Owner: " << get_owner_rank(p2, size) << " ID: " << p2.unique_hash_id << "\n";
            acc.bad_print_count++;
        }
    }

    // Area
    double volume = 0.5 * std::abs((p1.c[0] - p0.c[0])*(p2.c[1] - p0.c[1]) - (p1.c[1] - p0.c[1])*(p2.c[0] - p0.c[0]));
    acc.total_volume += volume;

    // Boundary Consistency
    Point<2> pts[3] = {p0, p1, p2};
    for (int i = 0; i < 3; i++) {
        Point<2>& ep1 = pts[i];
        Point<2>& ep2 = pts[(i + 1) % 3];

        bool is_bdy = false;
        if (std::abs(ep1.c[0]) < EPSILON && std::abs(ep2.c[0]) < EPSILON) is_bdy = true;
        else if (std::abs(ep1.c[0] - DOMAIN_SIZE[0]) < EPSILON && std::abs(ep2.c[0] - DOMAIN_SIZE[0]) < EPSILON) is_bdy = true;
        else if (std::abs(ep1.c[1]) < EPSILON && std::abs(ep2.c[1]) < EPSILON) is_bdy = true;
        else if (std::abs(ep1.c[1] - DOMAIN_SIZE[1]) < EPSILON && std::abs(ep2.c[1] - DOMAIN_SIZE[1]) < EPSILON) is_bdy = true;

        if (is_bdy) {
            double len = std::sqrt(std::pow(ep1.c[0]-ep2.c[0], 2) + std::pow(ep1.c[1]-ep2.c[1], 2));
            acc.boundary_measure += len;
            acc.boundary_facet_count++;
        }
    }
}

// On rank 0: test the reduced integrity measures (Euler characteristic, area, perimeter,
// edge lengths, angles, ghost coordinates) and print any failures. Returns true on success
template <int DIM>
static bool evaluate_integrity(long num_tris_owned_global, long num_points_owned_global,
                               const IntegrityAccum& global, long global_ghost_mismatch_count);

template <>
bool evaluate_integrity<2>(long num_tris_owned_global, long num_points_owned_global,
                           const IntegrityAccum& global, long global_ghost_mismatch_count) {
    // Calculate expected values based on domain dimensions
    double expected_area = DOMAIN_SIZE[0] * DOMAIN_SIZE[1];
    double expected_perimeter = 2.0 * (DOMAIN_SIZE[0] + DOMAIN_SIZE[1]);

    double global_total_area = global.total_volume;
    double global_boundary_len = global.boundary_measure;
    long global_boundary_edge_count = global.boundary_facet_count;
    long global_bad_edge_count = global.bad_edge_count;
    double global_max_edge_len = global.max_edge_len;
    double global_max_cosine = global.max_badness;

    bool success = true;
    long long total_degrees = (long long)num_tris_owned_global * 3;
    long long double_edges = total_degrees + global_boundary_edge_count;
    long long E_total = double_edges / 2;
    long long V_total = num_points_owned_global;
    long long F_total = num_tris_owned_global;
    long long euler = V_total - E_total + F_total;

    // Tolerances are relative so the check is independent of the domain scale.
    // On the unit square they equal the previous absolute tolerances of 1e-6 and 1e-4
    bool area_pass = std::abs(global_total_area - expected_area) < 1e-6 * expected_area;
    bool perim_pass = std::abs(global_boundary_len - expected_perimeter) < 2.5e-5 * expected_perimeter;
    bool euler_pass = (euler == 1);
    bool edge_pass = (global_bad_edge_count == 0);
    double global_min_angle = std::acos(clamp_val(global_max_cosine)) * 180.0 / 3.14159265358979323846;
    bool angle_pass = (global_min_angle >= MIN_ANGLE_DEG);
    bool ghost_pass = (global_ghost_mismatch_count == 0);

    if (!area_pass || !perim_pass || !euler_pass || !edge_pass || !angle_pass || !ghost_pass) {
        success = false;
        std::cout << "\n!!! MESH INTEGRITY CHECK FAILED !!!\n";
        if (!area_pass) std::cout << "  [FAIL] Total Area: " << std::fixed << std::setprecision(6) << global_total_area << " (Expected " << expected_area << ")\n";
        if (!perim_pass) std::cout << "  [FAIL] Boundary Perimeter: " << global_boundary_len << " (Expected " << expected_perimeter << ")\n";
        if (!euler_pass) std::cout << "  [FAIL] Euler Characteristic: " << euler << " (Expected 1)\n";
        if (!edge_pass) {
            std::cout << "  [FAIL] Bad Edges: " << global_bad_edge_count << " edges > " << MAX_EDGE_RATIO << "x target.\n";
            std::cout << "         Max Edge: " << global_max_edge_len << "\n";
        }
        if (!angle_pass) std::cout << "  [FAIL] Min Angle: " << global_min_angle << " deg (Expected >= " << MIN_ANGLE_DEG << " deg)\n";
        if (!ghost_pass) std::cout << "  [FAIL] Ghost coordinates: " << global_ghost_mismatch_count << " mismatches\n";
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    }
    return success;
}

// Perform rigorous checks on mesh topology and geometry
template <int DIM>
static bool CheckMeshIntegrity(MPI_Comm comm,
                               const std::vector<Point<DIM> >& points_on_owned_triangles_and_orphans,
                               const std::vector<Simplex<DIM> >& triangles_owned,
                               std::vector<int>& valence,
                               std::vector<std::pair<int, int>>& unique_edges) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    (void)unique_edges;

    // Build a set of point indices that appear in owned triangles
    std::vector<bool> appears_in_triangle(points_on_owned_triangles_and_orphans.size(), false);
    for (const auto& t : triangles_owned) {
        for (int a = 0; a <= DIM; ++a) appears_in_triangle[t.v[a]] = true;
    }

    for(size_t i=0; i<points_on_owned_triangles_and_orphans.size(); ++i) {
        const auto& p = points_on_owned_triangles_and_orphans[i];
        // Only check points owned by this rank that appear in a triangle
        // Orphan points (owned point but not in any owned triangle on this rank)
        // legitimately have valence 0 here
        if (valence[i] == 0 && appears_in_triangle[i] && get_owner_rank(p, size) == rank) {
            std::cout << "[Rank " << rank << "] UNCONNECTED POINT: (";
            print_coords<DIM>(std::cout, p.c);
            std::cout << ") ID: " << p.unique_hash_id << " Valence: " << valence[i] << "\n";
        }
        // On a single rank every point should be in a triangle - an orphan here
        // means a duplicate/dropped vertex and shows up as an Euler failure below
        if (size == 1 && !appears_in_triangle[i]) {
            std::cout << "[Rank " << rank << "] ORPHAN POINT ON SINGLE RANK: (";
            print_coords<DIM>(std::cout, p.c);
            std::cout << ") ID: " << p.unique_hash_id << "\n";
        }
    }
    appears_in_triangle.clear();
    // Explicitly delete memory
    std::vector<bool>().swap(appears_in_triangle);

    // 1. Count Owned Points (needed for Euler)
    long num_points_owned = 0;
    for (const auto& p : points_on_owned_triangles_and_orphans) {
        if (get_owner_rank(p, size) == rank) {
            num_points_owned++;
        }
    }

    // 2. Accumulate Local Stats
    IntegrityAccum local;
    local.total_volume = 0.0;
    local.boundary_measure = 0.0;
    local.boundary_facet_count = 0;
    local.bad_edge_count = 0;
    local.max_edge_len = 0.0;
    local.max_badness = -1.0;
    local.bad_print_count = 0;

    for (const auto& t : triangles_owned) {
        const Point<DIM> *p[DIM + 1];
        for (int a = 0; a <= DIM; ++a) p[a] = &points_on_owned_triangles_and_orphans[t.v[a]];
        integrity_accumulate_simplex<DIM>(p, rank, size, local);
    }

    // Ghost points must match their owners bit for bit
    long local_ghost_mismatch_count = CheckGhostCoordinates(comm, points_on_owned_triangles_and_orphans);

    // 3. Global Reductions
    long num_tris_owned = triangles_owned.size();
    long num_tris_owned_global, num_points_owned_global;
    IntegrityAccum global;
    global.bad_print_count = 0;
    long global_ghost_mismatch_count;

    MPI_Reduce(&num_tris_owned, &num_tris_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&num_points_owned, &num_points_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local.total_volume, &global.total_volume, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local.boundary_measure, &global.boundary_measure, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local.boundary_facet_count, &global.boundary_facet_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local.bad_edge_count, &global.bad_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_ghost_mismatch_count, &global_ghost_mismatch_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local.max_edge_len, &global.max_edge_len, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.max_badness, &global.max_badness, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int success = 1;
    if (rank == 0) {
        if (!evaluate_integrity<DIM>(num_tris_owned_global, num_points_owned_global, global, global_ghost_mismatch_count)) {
            success = 0;
        }
    }

    MPI_Bcast(&success, 1, MPI_INT, 0, comm);
    return (success == 1);
}

// ~~~~~~~~~~~~~~~~~

// Local (and then global) minima/maxima of the simplex quality printed in the stats
struct SimplexQuality {
    double min_volume, max_volume;
    double min_angle, max_angle;   // degrees
};

// Update q with one owned simplex (p in simplex order)
template <int DIM>
static void stats_accumulate_simplex(const Point<DIM> *const *p, SimplexQuality& q);

template <>
void stats_accumulate_simplex<2>(const Point<2> *const *p, SimplexQuality& q) {
    const Point<2>& p0 = *p[0];
    const Point<2>& p1 = *p[1];
    const Point<2>& p2 = *p[2];

    double d01_sq = std::pow(p1.c[0]-p0.c[0], 2) + std::pow(p1.c[1]-p0.c[1], 2);
    double d12_sq = std::pow(p2.c[0]-p1.c[0], 2) + std::pow(p2.c[1]-p1.c[1], 2);
    double d20_sq = std::pow(p0.c[0]-p2.c[0], 2) + std::pow(p0.c[1]-p2.c[1], 2);

    double d01 = std::sqrt(d01_sq);
    double d12 = std::sqrt(d12_sq);
    double d20 = std::sqrt(d20_sq);

    // volume
    double volume = 0.5 * std::abs((p1.c[0] - p0.c[0])*(p2.c[1] - p0.c[1]) - (p1.c[1] - p0.c[1])*(p2.c[0] - p0.c[0]));
    if (volume < q.min_volume) q.min_volume = volume;
    if (volume > q.max_volume) q.max_volume = volume;

    // Angles
    if (d01 > 1e-14 && d12 > 1e-14 && d20 > 1e-14) {
        double a0 = std::acos(clamp_val((d01_sq + d20_sq - d12_sq) / (2.0*d01*d20))) * 180.0 / 3.14159265358979323846;
        double a1 = std::acos(clamp_val((d01_sq + d12_sq - d20_sq) / (2.0*d01*d12))) * 180.0 / 3.14159265358979323846;
        double a2 = std::acos(clamp_val((d12_sq + d20_sq - d01_sq) / (2.0*d12*d20))) * 180.0 / 3.14159265358979323846;

        q.min_angle = std::min({q.min_angle, a0, a1, a2});
        q.max_angle = std::max({q.max_angle, a0, a1, a2});
    }
}

// Orientation histogram bin (of NUM_ORIENTATION_BINS) of the edge vector d
template <int DIM>
static int edge_orientation_bin(const double *d);

template <>
int edge_orientation_bin<2>(const double *d) {
    double dx = d[0];
    double dy = d[1];

    // Angle in [0, 180)
    double angle = std::atan2(dy, dx) * 180.0 / 3.14159265358979323846;
    if (angle < 0) angle += 180.0;
    if (angle >= 180.0) angle -= 180.0;

    int bin = static_cast<int>(angle / 10.0);
    if (bin < 0) bin = 0;
    if (bin >= 18) bin = 17;
    return bin;
}

// On rank 0: print the simplex, edge length and edge orientation part of the stats
template <int DIM>
static void print_simplex_stats(long num_tris_owned_global, const SimplexQuality& global,
                                double global_total_edge_len, long global_edge_count,
                                const std::vector<long>& global_bins);

template <>
void print_simplex_stats<2>(long num_tris_owned_global, const SimplexQuality& global,
                            double global_total_edge_len, long global_edge_count,
                            const std::vector<long>& global_bins) {
    double global_min_volume = global.min_volume, global_max_volume = global.max_volume;
    double global_min_angle = global.min_angle, global_max_angle = global.max_angle;

    std::cout << "Triangles:\n";
    std::cout << "  Total: " << num_tris_owned_global << "\n";
    std::cout << "  Volume Min: " << std::scientific << global_min_volume << "\n";
    std::cout << "  Volume Max: " << std::scientific << global_max_volume << "\n";

    std::cout << std::defaultfloat << std::setprecision(16);
    std::cout << "  Volume Ratio: " << (global_min_volume > 0 ? global_max_volume / global_min_volume : -1.0) << "\n";
    std::cout << "  Angle Min: " << global_min_angle << " deg\n";
    std::cout << "  Angle Max: " << global_max_angle << " deg\n";

    // Print average edge length
    std::cout << "  Avg Edge Len: " << (global_edge_count > 0 ? global_total_edge_len / global_edge_count : 0.0) << "\n";

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
}

// Print mesh statistics on rank 0
template <int DIM>
static void ComputeAndPrintStats(MPI_Comm comm, int final_smooth_its,
                                 const std::vector<Point<DIM> >& points_on_owned_triangles_and_orphans,
                                 const std::vector<Simplex<DIM> >& triangles_owned,
                                 const std::vector<int>& valence,
                                 const std::vector<std::pair<int, int>>& unique_edges) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Compute load imbalance & Connectivity
    long points_owned = 0;
    const int MAX_CONN = 30;
    long local_conn_bins[MAX_CONN] = {0};

    // Bin the valences
    for (size_t i = 0; i < points_on_owned_triangles_and_orphans.size(); ++i) {
        const auto& p = points_on_owned_triangles_and_orphans[i];
        if (get_owner_rank(p, size) == rank) {
            points_owned++;

            // Bin Connectivity using pre-calculated valence
            int degree = valence[i];
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

    SimplexQuality local;
    local.min_volume = 1e30; local.max_volume = -1.0;
    local.min_angle = 360.0; local.max_angle = -1.0;

    // 3. Edge Orientation Statistics - use the pre-computed edge list
    std::vector<long> local_bins(NUM_ORIENTATION_BINS, 0);

    // Accumulators for edge length stats
    double local_total_edge_len = 0.0;
    long local_edge_count = 0;

    // First pass: compute triangle stats
    for (const auto& t : triangles_owned) {
        const Point<DIM> *p[DIM + 1];
        for (int a = 0; a <= DIM; ++a) p[a] = &points_on_owned_triangles_and_orphans[t.v[a]];
        stats_accumulate_simplex<DIM>(p, local);
    }

    // Second pass: use pre-computed unique edges for edge stats
    for (const auto& edge : unique_edges) {
        int idx1 = edge.first;
        int idx2 = edge.second;

        const Point<DIM>& ep1 = points_on_owned_triangles_and_orphans[idx1];
        const Point<DIM>& ep2 = points_on_owned_triangles_and_orphans[idx2];

        // Check ownership of edge midpoint
        Point<DIM> midpoint;
        for (int d = 0; d < DIM; ++d) midpoint.c[d] = (ep1.c[d] + ep2.c[d]) * 0.5;

        if (get_owner_rank(midpoint, size) == rank) {
            double delta[DIM];
            for (int d = 0; d < DIM; ++d) delta[d] = ep2.c[d] - ep1.c[d];

            // Accumulate length
            double len = std::sqrt(dist_sq<DIM>(delta));
            local_total_edge_len += len;
            local_edge_count++;

            local_bins[edge_orientation_bin<DIM>(delta)]++;
        }
    }

    if (triangles_owned.empty()) {
        local.min_volume = 1e30; local.max_volume = -1.0;
        local.min_angle = 360.0; local.max_angle = -1.0;
    }

    SimplexQuality global;

    // Global reduction for edge stats
    double global_total_edge_len;
    long global_edge_count;
    MPI_Reduce(&local_total_edge_len, &global_total_edge_len, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_edge_count, &global_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);

    MPI_Reduce(&local.min_volume, &global.min_volume, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local.max_volume, &global.max_volume, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local.min_angle, &global.min_angle, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local.max_angle, &global.max_angle, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    std::vector<long> global_bins(NUM_ORIENTATION_BINS);
    MPI_Reduce(local_bins.data(), global_bins.data(), NUM_ORIENTATION_BINS, MPI_LONG, MPI_SUM, 0, comm);
    local_bins.clear();
    // Explicitly delete memory
    std::vector<long>().swap(local_bins);

    // Output stats to rank 0
    if (rank == 0) {
        std::cout << "\n=== Mesh Statistics ===\n";
        std::cout << "Final Smooth Iterations: " << final_smooth_its << "\n";
        std::cout << "Points:\n";
        std::cout << "  Total: " << num_points_owned_global << "\n";
        std::cout << "  Min per Rank: " << min_points_owned_global << "\n";
        std::cout << "  Max per Rank: " << max_points_owned_global << "\n";
        std::cout << "  Imbalance Ratio (Max/Avg): " << (double)max_points_owned_global / ((double)num_points_owned_global / size) << "\n";

        std::cout << "Connectivity (Valence):\n";
        // Start from 1 to skip any orphan points that falsely report valence 0
        for(int i=1; i<MAX_CONN; ++i) {
            if (global_conn_bins[i] > 0) {
                double pct = 100.0 * global_conn_bins[i] / num_points_owned_global;
                std::cout << "  Degree " << std::setw(2) << i << ": "
                          << std::setw(8) << global_conn_bins[i]
                          << " (" << std::fixed << std::setprecision(2) << pct << "%)\n";
            }
        }

        print_simplex_stats<DIM>(num_tris_owned_global, global, global_total_edge_len, global_edge_count, global_bins);
        std::cout << "=======================\n";
    }
    global_bins.clear();
}

// ~~~~~~~~~~~~~~~~~

// Factorize n into a grid of dims[0] x dims[1] (x dims[2]) tiles with that product
template <int DIM>
static void factorize_min_cut(int n, const double *size, int *dims);

// Factorize n into M x N such that M*N = n
// (Tries to match the W x H aspect ratio as closely as possible
// by finding the minimum total halo surface area possible)
template <>
void factorize_min_cut<2>(int n, const double *size, int *dims) {
    double W = size[0];
    double H = size[1];

    int best_m = 1;
    int best_n = n;
    double best_surface_area = 1e30;

    // Find all factor pairs of n
    for (int m = 1; m * m <= n; ++m) {
        if (n % m == 0) {
            int nn = n / m;

            // Calculate the total internal interface length for this decomposition.
            // In 2D this is the communication cut length, not a surface area.
            // There are (M - 1) vertical cuts of height H and
            // (N - 1) horizontal cuts of width W.
            double surface_area = (m - 1) * H + (nn - 1) * W;

            if (surface_area < best_surface_area) {
                best_surface_area = surface_area;
                best_m = m;
                best_n = nn;
            }

            // Also try the swapped decomposition (n x m) to ensure symmetry
            // This handles cases where the domain is rotated (width vs height swapped)
            double swapped_surface_area = (nn - 1) * H + (m - 1) * W;

            if (swapped_surface_area < best_surface_area) {
                best_surface_area = swapped_surface_area;
                best_m = nn;
                best_n = m;
            }
        }
    }

    dims[0] = best_m;
    dims[1] = best_n;
}

// ~~~~~~~~~~~~~~~~~

// Check the target edge length, domain size and smoothing count, aborting on bad input
template <int DIM>
static void validate_inputs(MPI_Comm comm, double target_edge_length, const double *size, int final_smooth_its);

template <>
void validate_inputs<2>(MPI_Comm comm, double target_edge_length, const double *size, int final_smooth_its) {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    double domain_width = size[0];
    double domain_height = size[1];

    // Validate the inputs
    if (!(target_edge_length > 0.0) || !std::isfinite(target_edge_length) ||
        !(domain_width > 0.0) || !std::isfinite(domain_width) ||
        !(domain_height > 0.0) || !std::isfinite(domain_height) || final_smooth_its < 0) {
        if (comm_rank == 0) {
            std::cerr << "ERROR: Target edge length (" << target_edge_length << "), domain width (" << domain_width
                      << ") and domain height (" << domain_height << ") must be positive and finite, "
                      << "and final smooth iterations (" << final_smooth_its << ") must be non-negative.\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // Ensure the grid indices fit in the unique hash id. The largest index is at the far
    // edge of the halo around the domain, which is at most max(width, height) + pad from
    // the origin, plus a couple of cells for rounding. This bounds the number of points
    // along a side of the domain, the distributed mesh itself can be larger
    double max_grid_idx = std::max(domain_width, domain_height) / target_edge_length
                          + (ANNEAL_ITERS + final_smooth_its + 8) + 2;
    if (max_grid_idx >= MAX_GRID_IDX) {
        if (comm_rank == 0) {
            std::cerr << "ERROR: Target edge length " << target_edge_length
                      << " is too small for the domain size, it needs grid indices up to " << max_grid_idx
                      << ", beyond the limit of " << MAX_GRID_IDX << " in the 31-bit index hashing scheme.\n"
                      << "Rewrite create_point_with_unique_hash_id to go further.\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

// Tolerance below which a point's star volume counts as degenerate
template <int DIM>
static double volume_tolerance();

template <>
double volume_tolerance<2>() {
    return TOL_LEN_SQ * 1e-2;
}

// On rank 0: the first lines of the stats header, describing the domain
template <int DIM>
static void print_domain_header();

template <>
void print_domain_header<2>() {
    std::cout << "Generating Unstructured Mesh of 2D box...\n";
    std::cout << "Target Edge Length: " << TARGET_EDGE_LENGTH << "\n";
    std::cout << "Domain Width: " << DOMAIN_SIZE[0] << ", Domain Height: " << DOMAIN_SIZE[1] << "\n";
}

// ~~~~~~~~~~~~~~~~~

// DIM=3 stubs. They fix the interfaces the 3D implementation has to fill in, and let the
// generic pipeline be compiled for DIM=3, but none of them is reachable yet

template <>
Point<3> create_point_with_unique_hash_id<3>(const double *x, const int *idx, int type) {
    (void)idx; (void)type;
    AbortNotImplemented3D("create_point_with_unique_hash_id");
    return Point<3>(x, 0);
}

template <>
int far_wall_grid_index<3>() {
    AbortNotImplemented3D("far_wall_grid_index");
    return 0;
}

template <>
uint64_t interior_seed<3>(const int *idx) {
    (void)idx;
    AbortNotImplemented3D("interior_seed");
    return 0;
}

template <>
std::vector<Simplex<3> > triangulation<3>(const std::vector<Point<3> >& points) {
    (void)points;
    AbortNotImplemented3D("The Delaunay backend");
    return std::vector<Simplex<3> >();
}

template <>
double simplex_volume<3>(const Point<3> *const *p) {
    (void)p;
    AbortNotImplemented3D("simplex_volume");
    return 0.0;
}

template <>
void simplex_centroid<3>(const Point<3> *const *p, double *centroid) {
    (void)p; (void)centroid;
    AbortNotImplemented3D("simplex_centroid");
}

template <>
double dist_sq<3>(const double *d) {
    (void)d;
    AbortNotImplemented3D("dist_sq");
    return 0.0;
}

template <>
double star_simplex_badness<3>(const Point<3>& p, int i, const Simplex<3>& tri, const std::vector<Point<3> >& points) {
    (void)p; (void)i; (void)tri; (void)points;
    AbortNotImplemented3D("star_simplex_badness");
    return 0.0;
}

template <>
int get_simplex_edges<3>(const int (*&edges)[2]) {
    edges = NULL;
    AbortNotImplemented3D("get_simplex_edges");
    return 0;
}

template <>
int get_boundary_features<3>(const int (*&features)[3]) {
    features = NULL;
    AbortNotImplemented3D("get_boundary_features");
    return 0;
}

template <>
PetscInt num_walls<3>() {
    AbortNotImplemented3D("num_walls");
    return 0;
}

template <>
PetscInt classify_point_wall<3>(const double *xyz, const double *domain_size) {
    (void)xyz; (void)domain_size;
    AbortNotImplemented3D("classify_point_wall");
    return 0;
}

template <>
PetscErrorCode facet_wall_value<3>(DM dm, PetscInt e, PetscSection coordSection, const PetscScalar *coords,
                                   const double *domain_size, PetscInt *val) {
    (void)dm; (void)e; (void)coordSection; (void)coords; (void)domain_size;
    *val = 0;
    AbortNotImplemented3D("facet_wall_value");
    return PETSC_SUCCESS;
}

template <>
void integrity_accumulate_simplex<3>(const Point<3> *const *p, int rank, int size, IntegrityAccum& acc) {
    (void)p; (void)rank; (void)size; (void)acc;
    AbortNotImplemented3D("integrity_accumulate_simplex");
}

template <>
bool evaluate_integrity<3>(long num_tris_owned_global, long num_points_owned_global,
                           const IntegrityAccum& global, long global_ghost_mismatch_count) {
    (void)num_tris_owned_global; (void)num_points_owned_global; (void)global; (void)global_ghost_mismatch_count;
    AbortNotImplemented3D("evaluate_integrity");
    return false;
}

template <>
void stats_accumulate_simplex<3>(const Point<3> *const *p, SimplexQuality& q) {
    (void)p; (void)q;
    AbortNotImplemented3D("stats_accumulate_simplex");
}

template <>
int edge_orientation_bin<3>(const double *d) {
    (void)d;
    AbortNotImplemented3D("edge_orientation_bin");
    return 0;
}

template <>
void print_simplex_stats<3>(long num_tris_owned_global, const SimplexQuality& global,
                            double global_total_edge_len, long global_edge_count,
                            const std::vector<long>& global_bins) {
    (void)num_tris_owned_global; (void)global; (void)global_total_edge_len; (void)global_edge_count; (void)global_bins;
    AbortNotImplemented3D("print_simplex_stats");
}

template <>
void factorize_min_cut<3>(int n, const double *size, int *dims) {
    (void)n; (void)size;
    dims[0] = dims[1] = dims[2] = 1;
    AbortNotImplemented3D("factorize_min_cut");
}

template <>
void validate_inputs<3>(MPI_Comm comm, double target_edge_length, const double *size, int final_smooth_its) {
    (void)comm; (void)target_edge_length; (void)size; (void)final_smooth_its;
    AbortNotImplemented3D("GenerateBoxMeshDM");
}

template <>
double volume_tolerance<3>() {
    AbortNotImplemented3D("volume_tolerance");
    return 0.0;
}

template <>
void print_domain_header<3>() {
    AbortNotImplemented3D("print_domain_header");
}

// ~~~~~~~~~~~~~~~~~

// Generate the mesh of the box [0,size[0]] x ... x [0,size[DIM-1]]
template <int DIM>
static DM GenerateBoxMeshDMImpl(MPI_Comm comm, double target_edge_length, const double *size, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats, int agglomeration_factor) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    // Validate the inputs
    validate_inputs<DIM>(comm, target_edge_length, size, final_smooth_its);

    // 1. Setup Globals
    TARGET_EDGE_LENGTH = target_edge_length;
    for (int d = 0; d < DIM; ++d) DOMAIN_SIZE[d] = size[d];

    TOL_LEN = TARGET_EDGE_LENGTH * 1e-4;
    TOL_LEN_SQ = TOL_LEN * TOL_LEN;
    TOL_VOLUME = volume_tolerance<DIM>();

    // Validate the agglomeration factor - the fine grid has to split evenly
    // into comm_size/agglomeration_factor coarse groups
    if (agglomeration_factor < 1) {
        if (comm_rank == 0) {
            std::cerr << "ERROR: Agglomeration factor " << agglomeration_factor
                      << " must be at least 1.\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }
    if (agglomeration_factor > 1 && comm_size % agglomeration_factor != 0) {
        if (comm_rank == 0) {
            std::cerr << "ERROR: Number of MPI ranks " << comm_size
                      << " is not divisible by the agglomeration factor " << agglomeration_factor << ".\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    AGG_FACTOR = agglomeration_factor;

    // Build the coarse grid the same way an ordinary run on comm_size/k ranks would,
    // then split each coarse tile into a compact sub-block of k fine tiles. The fine
    // grid is therefore a refinement of the coarse one by construction.
    factorize_min_cut<DIM>(comm_size / AGG_FACTOR, DOMAIN_SIZE, COARSE_DIM);
    // Factorize the sub-block against the coarse tile aspect ratio so we get
    // compact blocks (e.g. 3x2 for k=6) rather than degenerate 1xk strips
    double coarse_tile_size[DIM];
    for (int d = 0; d < DIM; ++d) coarse_tile_size[d] = DOMAIN_SIZE[d] / COARSE_DIM[d];
    factorize_min_cut<DIM>(AGG_FACTOR, coarse_tile_size, SUB_DIM);

    for (int d = 0; d < DIM; ++d) TILE_DIM[d] = COARSE_DIM[d] * SUB_DIM[d];

    if (comm_rank == 0 && print_stats) {
        print_domain_header<DIM>();
        std::cout << "Running on " << comm_size << " MPI ranks with decomposition ";
        print_dims<DIM>(std::cout, TILE_DIM);
        std::cout << ".\n";
        if (AGG_FACTOR > 1) {
            std::cout << "Agglomeration factor " << AGG_FACTOR << ": coarse grid ";
            print_dims<DIM>(std::cout, COARSE_DIM);
            std::cout << " of ";
            print_dims<DIM>(std::cout, SUB_DIM);
            std::cout << " sub-blocks -> fine grid ";
            print_dims<DIM>(std::cout, TILE_DIM);
            std::cout << ".\n";
        }
    }

    std::vector<Point<DIM> > points_on_owned_triangles_and_orphans;
    std::vector<Simplex<DIM> > triangles_owned;

    // 2. Distribute tiles (1 per rank)
    // Since we have exactly comm_size tiles, the rank maps directly to a tile
    int my_tile[DIM];
    rank_to_tile<DIM>(comm_rank, my_tile);
    process_tile<DIM>(comm, final_smooth_its, my_tile, points_on_owned_triangles_and_orphans, triangles_owned);

    //if (print_stats) std::cout << "Rank " << comm_rank << " generated " << triangles_owned.size() << " triangles_owned.\n";

    // 3. Check Integrity
    std::vector<int> valence;
    std::vector<std::pair<int, int>> unique_edges;
    // Compute valence and edges once here
    if (print_stats || integrity_check)
    {
      ComputeValenceAndEdges(points_on_owned_triangles_and_orphans, triangles_owned, valence, unique_edges);
    }

    if (integrity_check)
    {
      if (!CheckMeshIntegrity(comm, points_on_owned_triangles_and_orphans, triangles_owned, valence, unique_edges)) {
         return NULL;
      }
   }

    // 4. Print stats
    if (print_stats) ComputeAndPrintStats(comm, final_smooth_its, points_on_owned_triangles_and_orphans, triangles_owned, valence, unique_edges);

    // Free the valence and edge data now that we're done with stats
    valence.clear();
    unique_edges.clear();
    // Explicitly delete memory
    std::vector<int>().swap(valence);
    std::vector<std::pair<int,int>>().swap(unique_edges);

    // 5. Create the DM
    if (comm_rank == 0 && print_stats) std::cout << "Creating DM...\n";
    DM dm = CreateDM(comm, points_on_owned_triangles_and_orphans, triangles_owned);

    points_on_owned_triangles_and_orphans.clear();
    triangles_owned.clear();
    // Explicitly delete memory
    std::vector<Point<DIM> >().swap(points_on_owned_triangles_and_orphans);
    std::vector<Simplex<DIM> >().swap(triangles_owned);

    PetscErrorCode ierr;
    ierr = PetscObjectSetName((PetscObject)dm, "Mesh");

    // 6. Label boundaries
    ierr = SetBoxDomain(dm, DIM, DOMAIN_SIZE);
    LabelBoundaries<DIM>(dm, DOMAIN_SIZE);

    // 7. Add refinement hook so labels are applied after any refinement
    ierr = DMRefineHookAdd(dm, RefineHook_LabelBoundaries, NULL, NULL);
    (void)ierr;

    return dm;
}

// Compile the whole pipeline for DIM=3 against the stubs above, so the generic code is
// kept 3D-clean. Nothing calls it yet
template DM GenerateBoxMeshDMImpl<3>(MPI_Comm, double, const double *, int, PetscBool, PetscBool, int);

// ~~~~~~~~~~~~~~~~~

PETSC_EXTERN DM GenerateBoxMeshDMAgglom(MPI_Comm comm, double target_edge_length, double domain_width, double domain_height, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats, int agglomeration_factor) {
    double size[2] = {domain_width, domain_height};
    return GenerateBoxMeshDMImpl<2>(comm, target_edge_length, size, final_smooth_its, integrity_check, print_stats, agglomeration_factor);
}

// ~~~~~~~~~~~~~~~~~

// Original entry point - equivalent to an agglomeration factor of 1
PETSC_EXTERN DM GenerateBoxMeshDM(MPI_Comm comm, double target_edge_length, double domain_width, double domain_height, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats) {
    return GenerateBoxMeshDMAgglom(comm, target_edge_length, domain_width, domain_height, final_smooth_its, integrity_check, print_stats, 1);
}
