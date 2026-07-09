#include "BoxMeshDM_3D.h"
#include "petscsystypes.h"
#include "tetgen.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mpi.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscviewerhdf5.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

const double EPSILON = 1e-13;
const double START_JITTER = 0.30;
const int ANNEAL_ITERS = 3;

static int TILE_DIM_X = 1;
static int TILE_DIM_Y = 1;
static int TILE_DIM_Z = 1;

double DOMAIN_WIDTH = 1.0;
double DOMAIN_HEIGHT = 1.0;
double DOMAIN_DEPTH = 1.0;
double TARGET_EDGE_LENGTH = 0.02;

struct Face3D {
    int v0, v1, v2;
    bool operator<(const Face3D &o) const {
        if (v0 != o.v0)
            return v0 < o.v0;
        if (v1 != o.v1)
            return v1 < o.v1;
        return v2 < o.v2;
    }
    bool operator==(const Face3D &o) const { return v0 == o.v0 && v1 == o.v1 && v2 == o.v2; }
};

// Pack: [Type: 2 bits] [ix: 20 bits] [iy: 20 bits] [iz: 20 bits]
uint64_t create_point_with_unique_hash_id_3D(int ix, int iy, int iz, int type) {
    uint64_t id = ((uint64_t)(type & 0x3) << 60) | ((uint64_t)(ix & 0xFFFFF) << 40) | ((uint64_t)(iy & 0xFFFFF) << 20) |
                  ((uint64_t)(iz & 0xFFFFF));
    return id;
}

// Simple splitmix64 for deterministic RNG
uint64_t splitmix64_3D(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

double next_double_3D(uint64_t &state) { return (splitmix64_3D(state) >> 11) * (1.0 / 9007199254740992.0); }

// Determines which rank owns a point based on spatial location in 3D
int get_owner_rank_3D(const Point3D &p) {
    double tile_s_x = DOMAIN_WIDTH / TILE_DIM_X;
    double tile_s_y = DOMAIN_HEIGHT / TILE_DIM_Y;
    double tile_s_z = DOMAIN_DEPTH / TILE_DIM_Z;

    int tx = std::floor(p.x / tile_s_x);
    int ty = std::floor(p.y / tile_s_y);
    int tz = std::floor(p.z / tile_s_z);

    if (tx < 0)
        tx = 0;
    if (tx >= TILE_DIM_X)
        tx = TILE_DIM_X - 1;
    if (ty < 0)
        ty = 0;
    if (ty >= TILE_DIM_Y)
        ty = TILE_DIM_Y - 1;
    if (tz < 0)
        tz = 0;
    if (tz >= TILE_DIM_Z)
        tz = TILE_DIM_Z - 1;

    return (tz * TILE_DIM_Y * TILE_DIM_X) + (ty * TILE_DIM_X) + tx;
}

// Clamps to prevent acos domain errors
static double clamp_val_3D(double v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); }

// Calculates the dihedral angles of a given tetrahedron
static void get_tet_dihedral_angles(const Point3D &p0, const Point3D &p1, const Point3D &p2, const Point3D &p3,
                                    double angles[6]) {
    auto normal = [](const Point3D &a, const Point3D &b, const Point3D &c) {
        double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        return Point3D(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    };

    Point3D n[4] = {normal(p1, p2, p3), normal(p0, p3, p2), normal(p0, p1, p3), normal(p0, p2, p1)};

    auto mag = [](const Point3D &vec) { return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z); };
    auto dot = [](const Point3D &a, const Point3D &b) { return a.x * b.x + a.y * b.y + a.z * b.z; };

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            double m1 = mag(n[i]), m2 = mag(n[j]);
            if (m1 < 1e-14 || m2 < 1e-14) {
                angles[idx++] = 0.0;
                continue;
            }
            double cos_theta = dot(n[i], n[j]) / (m1 * m2);
            angles[idx++] = std::acos(clamp_val_3D(-cos_theta)) * 180.0 / 3.14159265358979323846;
        }
    }
}

// Calculates the square mean ratio of a given tetrahedron
// Maximizing this metric prevents slivers
static double get_square_mean_ratio(const Point3D &p0, const Point3D &p1, const Point3D &p2, const Point3D &p3) {
    double v1x = p1.x - p0.x, v1y = p1.y - p0.y, v1z = p1.z - p0.z;
    double v2x = p2.x - p0.x, v2y = p2.y - p0.y, v2z = p2.z - p0.z;
    double v3x = p3.x - p0.x, v3y = p3.y - p0.y, v3z = p3.z - p0.z;

    // Cross product (p2-p0) x (p3-p0)
    double cx = v2y * v3z - v2z * v3y;
    double cy = v2z * v3x - v2x * v3z;
    double cz = v2x * v3y - v2y * v3x;

    // 6x Signed volume
    double vol6 = v1x * cx + v1y * cy + v1z * cz;

    // If volume is negative or effectively zero, it's inverted/degenerate
    if (vol6 <= 1e-14)
        return -1.0;

    // Squared edge lengths
    double d01 = v1x * v1x + v1y * v1y + v1z * v1z;
    double d02 = v2x * v2x + v2y * v2y + v2z * v2z;
    double d03 = v3x * v3x + v3y * v3y + v3z * v3z;
    double d12 = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y) + (p2.z - p1.z) * (p2.z - p1.z);
    double d23 = (p3.x - p2.x) * (p3.x - p2.x) + (p3.y - p2.y) * (p3.y - p2.y) + (p3.z - p2.z) * (p3.z - p2.z);
    double d31 = (p1.x - p3.x) * (p1.x - p3.x) + (p1.y - p3.y) * (p1.y - p3.y) + (p1.z - p3.z) * (p1.z - p3.z);

    double sum_sq = d01 + d02 + d03 + d12 + d23 + d31;

    // V^2 / S^3 metric. Higher is better.
    return (vol6 * vol6) / (sum_sq * sum_sq * sum_sq);
}

// Fast Edge and Valence Computation
static void ComputeValenceAndEdges_3D(const std::vector<Tetrahedron> &tets, int num_points,
                                      std::vector<std::pair<int, int>> &unique_edges, std::vector<int> &valence) {
    unique_edges.clear();
    unique_edges.reserve(tets.size() * 6);

    for (const auto &t : tets) {
        int e[6][2] = {{t.v0, t.v1}, {t.v0, t.v2}, {t.v0, t.v3}, {t.v1, t.v2}, {t.v1, t.v3}, {t.v2, t.v3}};
        for (int i = 0; i < 6; ++i) {
            int u = e[i][0];
            int v = e[i][1];
            if (u > v)
                std::swap(u, v);
            unique_edges.push_back({u, v});
        }
    }

    std::sort(unique_edges.begin(), unique_edges.end());
    unique_edges.erase(std::unique(unique_edges.begin(), unique_edges.end()), unique_edges.end());

    valence.assign(num_points, 0);
    for (const auto &edge : unique_edges) {
        valence[edge.first]++;
        valence[edge.second]++;
    }
}

// Deterministically resolve ownership of boundary nodes via MPI
static void ResolveBoundaryOwnership_3D(MPI_Comm comm, std::vector<Point3D> &points, double min_x, double min_y, double min_z,
                                        double max_x, double max_y, double max_z, double interior_min_x, double interior_min_y,
                                        double interior_min_z, double interior_max_x, double interior_max_y,
                                        double interior_max_z, double pad) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    struct Claim3D {
        uint64_t id;
        int geo_rank;
        double x, y, z;
    };

    MPI_Datatype MPI_CLAIM_3D;
    {
        int blocklengths[4] = {1, 1, 3};
        MPI_Aint displacements[3];
        MPI_Datatype types[3] = {MPI_UINT64_T, MPI_INT, MPI_DOUBLE};

        Claim3D dummy;
        MPI_Aint base_addr;
        MPI_Get_address(&dummy, &base_addr);
        MPI_Get_address(&dummy.id, &displacements[0]);
        MPI_Get_address(&dummy.geo_rank, &displacements[1]);
        MPI_Get_address(&dummy.x, &displacements[2]);

        for (int i = 0; i < 3; i++)
            displacements[i] -= base_addr;

        MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_CLAIM_3D);
        MPI_Type_commit(&MPI_CLAIM_3D);
    }

    std::vector<std::vector<Claim3D>> send_buffers(size);
    std::unordered_set<uint64_t> involved_ids;

    int tx = rank % TILE_DIM_X;
    int ty = (rank / TILE_DIM_X) % TILE_DIM_Y;
    int tz = rank / (TILE_DIM_X * TILE_DIM_Y);

    for (auto &p : points) {
        if (p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y && p.z > min_z && p.z < max_z &&
            (p.x <= interior_min_x || p.x >= interior_max_x || p.y <= interior_min_y || p.y >= interior_max_y ||
             p.z <= interior_min_z || p.z >= interior_max_z)) {

            involved_ids.insert(p.unique_hash_id);
            int my_geo_rank = get_owner_rank_3D(p);

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0)
                            continue;
                        int nx = tx + dx;
                        int ny = ty + dy;
                        int nz = tz + dz;
                        if (nx >= 0 && nx < TILE_DIM_X && ny >= 0 && ny < TILE_DIM_Y && nz >= 0 && nz < TILE_DIM_Z) {
                            double tile_s_x = DOMAIN_WIDTH / TILE_DIM_X;
                            double tile_s_y = DOMAIN_HEIGHT / TILE_DIM_Y;
                            double tile_s_z = DOMAIN_DEPTH / TILE_DIM_Z;

                            double n_min_x = nx * tile_s_x;
                            double n_max_x = (nx + 1) * tile_s_x;
                            double n_min_y = ny * tile_s_y;
                            double n_max_y = (ny + 1) * tile_s_y;
                            double n_min_z = nz * tile_s_z;
                            double n_max_z = (nz + 1) * tile_s_z;
                            double safe_pad = pad * 1.2;

                            if (p.x >= n_min_x - safe_pad && p.x <= n_max_x + safe_pad && p.y >= n_min_y - safe_pad &&
                                p.y <= n_max_y + safe_pad && p.z >= n_min_z - safe_pad && p.z <= n_max_z + safe_pad) {
                                int n_rank = (nz * TILE_DIM_Y * TILE_DIM_X) + (ny * TILE_DIM_X) + nx;
                                send_buffers[n_rank].push_back({p.unique_hash_id, my_geo_rank, p.x, p.y, p.z});
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> send_counts(size), recv_counts(size);
    for (int r = 0; r < size; ++r)
        send_counts[r] = send_buffers[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<std::vector<Claim3D>> recv_buffers(size);
    std::vector<MPI_Request> requests;

    for (int r = 0; r < size; ++r) {
        if (recv_counts[r] > 0) {
            recv_buffers[r].resize(recv_counts[r]);
            MPI_Irecv(recv_buffers[r].data(), recv_counts[r], MPI_CLAIM_3D, r, 999, comm, &requests.emplace_back());
        }
    }
    for (int r = 0; r < size; ++r) {
        if (send_counts[r] > 0) {
            MPI_Isend(send_buffers[r].data(), send_counts[r], MPI_CLAIM_3D, r, 999, comm, &requests.emplace_back());
        }
    }
    if (!requests.empty())
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    MPI_Type_free(&MPI_CLAIM_3D);

    struct ResolutionData {
        int best_rank = 999999;
        double best_x = 0, best_y = 0, best_z = 0;
    };
    std::unordered_map<uint64_t, ResolutionData> resolution_map;

    for (const auto &p : points) {
        if (involved_ids.count(p.unique_hash_id)) {
            auto &data = resolution_map[p.unique_hash_id];
            data.best_rank = rank;
            data.best_x = p.x;
            data.best_y = p.y;
            data.best_z = p.z;
        }
    }

    for (int r = 0; r < size; ++r) {
        for (const auto &claim : recv_buffers[r]) {
            involved_ids.insert(claim.id);
            auto &data = resolution_map[claim.id];
            if (r < data.best_rank) {
                data.best_rank = r;
                data.best_x = claim.x;
                data.best_y = claim.y;
                data.best_z = claim.z;
            }
        }
    }

    for (auto &p : points) {
        if (involved_ids.count(p.unique_hash_id)) {
            const auto &data = resolution_map[p.unique_hash_id];
            p.x = data.best_x;
            p.y = data.best_y;
            p.z = data.best_z;
        }
    }

    // Memory cleanup
    send_buffers.clear();
    recv_buffers.clear();
    std::vector<std::vector<Claim3D>>().swap(send_buffers);
    std::vector<std::vector<Claim3D>>().swap(recv_buffers);
    involved_ids.clear();
    resolution_map.clear();
}

bool apply_boundary_constraint_3D(Point3D &p, double &dx, double &dy, double &dz) {
    bool on_boundary = false;

    // X Planes (Left/Right)
    if (std::abs(p.x) < EPSILON) {
        p.x = 0.0;
        dx = 0.0;
        on_boundary = true;
    } else if (std::abs(p.x - DOMAIN_WIDTH) < EPSILON) {
        p.x = DOMAIN_WIDTH;
        dx = 0.0;
        on_boundary = true;
    }

    // Y Planes (Bottom/Top)
    if (std::abs(p.y) < EPSILON) {
        p.y = 0.0;
        dy = 0.0;
        on_boundary = true;
    } else if (std::abs(p.y - DOMAIN_HEIGHT) < EPSILON) {
        p.y = DOMAIN_HEIGHT;
        dy = 0.0;
        on_boundary = true;
    }

    // Z Planes (Front/Back)
    if (std::abs(p.z) < EPSILON) {
        p.z = 0.0;
        dz = 0.0;
        on_boundary = true;
    } else if (std::abs(p.z - DOMAIN_DEPTH) < EPSILON) {
        p.z = DOMAIN_DEPTH;
        dz = 0.0;
        on_boundary = true;
    }

    return on_boundary;
}

void keep_interior_point_inside_3D(Point3D &p) {
    if (p.x < 0.0)
        p.x = -p.x;
    else if (p.x > DOMAIN_WIDTH)
        p.x = DOMAIN_WIDTH - (p.x - DOMAIN_WIDTH);

    if (p.y < 0.0)
        p.y = -p.y;
    else if (p.y > DOMAIN_HEIGHT)
        p.y = DOMAIN_HEIGHT - (p.y - DOMAIN_HEIGHT);

    if (p.z < 0.0)
        p.z = -p.z;
    else if (p.z > DOMAIN_DEPTH)
        p.z = DOMAIN_DEPTH - (p.z - DOMAIN_DEPTH);

    // Enforce strict interiority
    if (p.x <= EPSILON)
        p.x = EPSILON * 2.0;
    if (p.x >= DOMAIN_WIDTH - EPSILON)
        p.x = DOMAIN_WIDTH - EPSILON * 2.0;
    if (p.y <= EPSILON)
        p.y = EPSILON * 2.0;
    if (p.y >= DOMAIN_HEIGHT - EPSILON)
        p.y = DOMAIN_HEIGHT - EPSILON * 2.0;
    if (p.z <= EPSILON)
        p.z = EPSILON * 2.0;
    if (p.z >= DOMAIN_DEPTH - EPSILON)
        p.z = DOMAIN_DEPTH - EPSILON * 2.0;
}

void apply_jitter_3D(std::vector<Point3D> &points, double amount, int iter) {
    for (auto &p : points) {
        uint64_t h = p.unique_hash_id;
        h ^= iter + 0x9e3779b9 + (h << 6) + (h >> 2); // Seed modifier

        double jx = (next_double_3D(h) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;
        double jy = (next_double_3D(h) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;
        double jz = (next_double_3D(h) - 0.5) * 2.0 * amount * TARGET_EDGE_LENGTH;

        bool was_boundary = apply_boundary_constraint_3D(p, jx, jy, jz);

        p.x += jx;
        p.y += jy;
        p.z += jz;

        if (!was_boundary) {
            keep_interior_point_inside_3D(p);
        } else {
            // Strict float drift clamping
            if (p.x < 0)
                p.x = 0;
            if (p.x > DOMAIN_WIDTH)
                p.x = DOMAIN_WIDTH;
            if (p.y < 0)
                p.y = 0;
            if (p.y > DOMAIN_HEIGHT)
                p.y = DOMAIN_HEIGHT;
            if (p.z < 0)
                p.z = 0;
            if (p.z > DOMAIN_DEPTH)
                p.z = DOMAIN_DEPTH;
        }
    }
}

// Tetrahedralizes a point cloud using TetGen
void TetrahedralizePointCloud(const std::vector<Point3D> &point_cloud, std::vector<Tetrahedron> &out_tetrahedra) {
    tetgenio in, out;

    in.numberofpoints = static_cast<int>(point_cloud.size());
    in.pointlist = new double[in.numberofpoints * 3];

    for (size_t i = 0; i < point_cloud.size(); ++i) {
        in.pointlist[i * 3 + 0] = point_cloud[i].x;
        in.pointlist[i * 3 + 1] = point_cloud[i].y;
        in.pointlist[i * 3 + 2] = point_cloud[i].z;
    }

    tetgenbehavior behavior;
    char switches[] = "Qz";
    behavior.parse_commandline(switches);
    tetrahedralize(&behavior, &in, &out, nullptr, nullptr);

    out_tetrahedra.clear();
    out_tetrahedra.reserve(out.numberoftetrahedra);

    for (int i = 0; i < out.numberoftetrahedra; ++i) {
        Tetrahedron tet;
        tet.v0 = out.tetrahedronlist[i * 4 + 0];
        tet.v1 = out.tetrahedronlist[i * 4 + 1];
        tet.v2 = out.tetrahedronlist[i * 4 + 2];
        tet.v3 = out.tetrahedronlist[i * 4 + 3];
        out_tetrahedra.push_back(tet);
    }
}

double calculate_tet_volume(const Point3D &p0, const Point3D &p1, const Point3D &p2, const Point3D &p3) {
    // Volume = 1/6 * |(p1-p0) . ((p2-p0) x (p3-p0))|
    double v1x = p1.x - p0.x, v1y = p1.y - p0.y, v1z = p1.z - p0.z;
    double v2x = p2.x - p0.x, v2y = p2.y - p0.y, v2z = p2.z - p0.z;
    double v3x = p3.x - p0.x, v3y = p3.y - p0.y, v3z = p3.z - p0.z;

    // Cross product (p2-p0) x (p3-p0)
    double cx = v2y * v3z - v2z * v3y;
    double cy = v2z * v3x - v2x * v3z;
    double cz = v2x * v3y - v2y * v3x;

    // Dot product with (p1-p0)
    return std::abs(v1x * cx + v1y * cy + v1z * cz) / 6.0;
}

double calculate_signed_tet_volume(const Point3D &p0, const Point3D &p1, const Point3D &p2, const Point3D &p3) {
    // Volume = 1/6 * |(p1-p0) . ((p2-p0) x (p3-p0))|
    double v1x = p1.x - p0.x, v1y = p1.y - p0.y, v1z = p1.z - p0.z;
    double v2x = p2.x - p0.x, v2y = p2.y - p0.y, v2z = p2.z - p0.z;
    double v3x = p3.x - p0.x, v3y = p3.y - p0.y, v3z = p3.z - p0.z;

    // Cross product (p2-p0) x (p3-p0)
    double cx = v2y * v3z - v2z * v3y;
    double cy = v2z * v3x - v2x * v3z;
    double cz = v2x * v3y - v2y * v3x;

    // Dot product with (p1-p0)
    return (v1x * cx + v1y * cy + v1z * cz) / 6.0;
}

// 3D Lloyd-smoothing
// Moves vertices towards the volume-weighted centroid of surrounding tets
// Optimising element shape and aspect ratios
void relax_points_lloyd_3D(std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets, double factor, double min_x,
                           double min_y, double min_z, double max_x, double max_y, double max_z) {
    int n = points.size();
    std::vector<double> wx(n, 0.0), wy(n, 0.0), wz(n, 0.0), w_sum(n, 0.0);

    // Compressed Sparse Row (CSR) Setup for fast adjacency lookups
    std::vector<int> tet_count(n, 0);
    for (const auto &t : tets) {
        tet_count[t.v0]++;
        tet_count[t.v1]++;
        tet_count[t.v2]++;
        tet_count[t.v3]++;
    }

    std::vector<int> tet_offset(n + 1, 0);
    for (int i = 0; i < n; ++i)
        tet_offset[i + 1] = tet_offset[i] + tet_count[i];
    std::vector<int> tet_data(tet_offset[n]);
    std::fill(tet_count.begin(), tet_count.end(), 0);

    // Loops through all tets and calculates the volume-weighted centroid
    for (size_t k = 0; k < tets.size(); ++k) {
        const auto &t = tets[k];
        const Point3D &p0 = points[t.v0], &p1 = points[t.v1], &p2 = points[t.v2], &p3 = points[t.v3];
        double cx = (p0.x + p1.x + p2.x + p3.x) * 0.25;
        double cy = (p0.y + p1.y + p2.y + p3.y) * 0.25;
        double cz = (p0.z + p1.z + p2.z + p3.z) * 0.25;
        double vol = calculate_tet_volume(p0, p1, p2, p3);

        int v[4] = {t.v0, t.v1, t.v2, t.v3};
        for (int i = 0; i < 4; ++i) {
            wx[v[i]] += vol * cx;
            wy[v[i]] += vol * cy;
            wz[v[i]] += vol * cz;
            w_sum[v[i]] += vol;
            tet_data[tet_offset[v[i]] + tet_count[v[i]]++] = k;
        }
    }

    std::vector<double> candidate_factors = {0.1, 0.3, 0.5, 0.7, 0.9};

    // Loops through all points and apply the spring forces
    for (int i = 0; i < n; ++i) {
        if (w_sum[i] < 1e-12)
            continue;
        if (points[i].x > min_x && points[i].x < max_x && points[i].y > min_y && points[i].y < max_y && points[i].z > min_z &&
            points[i].z < max_z) {
            double tx = wx[i] / w_sum[i];
            double ty = wy[i] / w_sum[i];
            double tz = wz[i] / w_sum[i];

            // Scales the full displacement by the provided factor
            double full_dx = (tx - points[i].x) * factor;
            double full_dy = (ty - points[i].y) * factor;
            double full_dz = (tz - points[i].z) * factor;

            int start = tet_offset[i], end = tet_offset[i + 1];
            Point3D best_candidate = points[i];
            double best_min_quality = -1.0;

            // Find the baseline minimum quality before moving
            for (int j = start; j < end; ++j) {
                const auto &t = tets[tet_data[j]];
                double q = get_square_mean_ratio(points[t.v0], points[t.v1], points[t.v2], points[t.v3]);
                if (best_min_quality < 0 || q < best_min_quality)
                    best_min_quality = q;
            }

            // Test displacement factors
            for (double cf : candidate_factors) {
                Point3D candidate = points[i];
                double dx = full_dx * cf, dy = full_dy * cf, dz = full_dz * cf;
                bool was_boundary = apply_boundary_constraint_3D(candidate, dx, dy, dz);
                candidate.x += dx;
                candidate.y += dy;
                candidate.z += dz;

                if (!was_boundary) {
                    keep_interior_point_inside_3D(candidate);
                } else {
                    if (candidate.x < 0)
                        candidate.x = 0;
                    if (candidate.x > DOMAIN_WIDTH)
                        candidate.x = DOMAIN_WIDTH;
                    if (candidate.y < 0)
                        candidate.y = 0;
                    if (candidate.y > DOMAIN_HEIGHT)
                        candidate.y = DOMAIN_HEIGHT;
                    if (candidate.z < 0)
                        candidate.z = 0;
                    if (candidate.z > DOMAIN_DEPTH)
                        candidate.z = DOMAIN_DEPTH;
                }

                double candidate_min_quality = 1e30;
                bool inverted = false;
                for (int j = start; j < end; ++j) {
                    const auto &t = tets[tet_data[j]];
                    Point3D p0 = (t.v0 == i) ? candidate : points[t.v0];
                    Point3D p1 = (t.v1 == i) ? candidate : points[t.v1];
                    Point3D p2 = (t.v2 == i) ? candidate : points[t.v2];
                    Point3D p3 = (t.v3 == i) ? candidate : points[t.v3];

                    // Combined inversion/degeneracy and quality check in one call
                    double q = get_square_mean_ratio(p0, p1, p2, p3);
                    if (q < 0) {
                        inverted = true;
                        break;
                    }
                    if (q < candidate_min_quality)
                        candidate_min_quality = q;
                }

                if (!inverted && candidate_min_quality > best_min_quality) {
                    best_min_quality = candidate_min_quality;
                    best_candidate = candidate;
                }
            }
            points[i] = best_candidate;
        }
    }
}

// 3D Spring-Force Relaxation
// Attempts to equalize all edges to TARGET_EDGE_LENGTH
void relax_points_spring_3D(std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets, double dt, double min_x,
                            double min_y, double min_z, double max_x, double max_y, double max_z) {
    int n = points.size();
    std::vector<double> force_x(n, 0.0), force_y(n, 0.0), force_z(n, 0.0);

    std::vector<std::pair<int, int>> unique_edges;
    std::vector<int> valence;
    ComputeValenceAndEdges_3D(tets, n, unique_edges, valence);

    for (const auto &edge : unique_edges) {
        int idx1 = edge.first, idx2 = edge.second;
        double dx = points[idx2].x - points[idx1].x;
        double dy = points[idx2].y - points[idx1].y;
        double dz = points[idx2].z - points[idx1].z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1e-14)
            continue;

        double force_mag = (dist - TARGET_EDGE_LENGTH);
        double fx = force_mag * (dx / dist), fy = force_mag * (dy / dist), fz = force_mag * (dz / dist);

        force_x[idx1] += fx;
        force_y[idx1] += fy;
        force_z[idx1] += fz;
        force_x[idx2] -= fx;
        force_y[idx2] -= fy;
        force_z[idx2] -= fz;
    }

    // Integrate forces and update coordinates
    for (int i = 0; i < n; ++i) {
        if (valence[i] == 0)
            continue;
        if (points[i].x > min_x && points[i].x < max_x && points[i].y > min_y && points[i].y < max_y && points[i].z > min_z &&
            points[i].z < max_z) {
            double dx = (force_x[i] / valence[i]) * dt;
            double dy = (force_y[i] / valence[i]) * dt;
            double dz = (force_z[i] / valence[i]) * dt;

            Point3D temp_p = points[i];
            bool was_boundary = apply_boundary_constraint_3D(temp_p, dx, dy, dz);
            temp_p.x += dx;
            temp_p.y += dy;
            temp_p.z += dz;

            if (!was_boundary) {
                keep_interior_point_inside_3D(temp_p);
            } else {
                if (temp_p.x < 0)
                    temp_p.x = 0;
                if (temp_p.x > DOMAIN_WIDTH)
                    temp_p.x = DOMAIN_WIDTH;
                if (temp_p.y < 0)
                    temp_p.y = 0;
                if (temp_p.y > DOMAIN_HEIGHT)
                    temp_p.y = DOMAIN_HEIGHT;
                if (temp_p.z < 0)
                    temp_p.z = 0;
                if (temp_p.z > DOMAIN_DEPTH)
                    temp_p.z = DOMAIN_DEPTH;
            }
            points[i] = temp_p;
        }
    }
}

// PETSc DMPlex Boundary Labelling
static void LabelBoundaries_3D(DM dm) {
    DMLabel label;
    DMGetLabel(dm, "Face Sets", &label);
    if (!label) {
        DMCreateLabel(dm, "Face Sets");
        DMGetLabel(dm, "Face Sets", &label);
    } else {
        DMLabelClearStratum(label, 1);
        DMLabelClearStratum(label, 2);
        DMLabelClearStratum(label, 3);
        DMLabelClearStratum(label, 4);
        DMLabelClearStratum(label, 5);
        DMLabelClearStratum(label, 6);
    }

    Vec coordsVec;
    DMGetCoordinatesLocal(dm, &coordsVec);
    PetscSection coordSection;
    DMGetCoordinateSection(dm, &coordSection);

    const PetscScalar *coords;
    VecGetArrayRead(coordsVec, &coords);

    // Label Vertices (Depth 0)
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

    for (PetscInt v = vStart; v < vEnd; ++v) {
        PetscInt dof, off;
        PetscSectionGetDof(coordSection, v, &dof);
        if (dof > 0) {
            PetscSectionGetOffset(coordSection, v, &off);
            double x = coords[off], y = coords[off + 1], z = coords[off + 2];

            PetscInt val = 0;
            // 1=Bottom(Z=0), 2=Top(Z=D), 3=Left(X=0), 4=Right(X=W), 5=Front(Y=0), 6=Back(Y=H)
            if (std::abs(z) < EPSILON)
                val = 1;
            else if (std::abs(z - DOMAIN_DEPTH) < EPSILON)
                val = 2;
            else if (std::abs(x) < EPSILON)
                val = 3;
            else if (std::abs(x - DOMAIN_WIDTH) < EPSILON)
                val = 4;
            else if (std::abs(y) < EPSILON)
                val = 5;
            else if (std::abs(y - DOMAIN_HEIGHT) < EPSILON)
                val = 6;

            if (val != 0)
                DMLabelSetValue(label, v, val);
        }
    }
    DMPlexLabelComplete(dm, label);

    // Label Faces (Height 1 in 3D = faces)
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);

    // Get the valid chart limits of the coordinate section to prevent SEGVs
    PetscInt cStart, cEnd;
    PetscSectionGetChart(coordSection, &cStart, &cEnd);

    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscInt num_edges;
        const PetscInt *edges;

        // Traverse Face -> Edges
        DMPlexGetConeSize(dm, f, &num_edges);
        DMPlexGetCone(dm, f, &edges);

        double cx = 0, cy = 0, cz = 0;
        int count = 0;

        for (PetscInt i = 0; i < num_edges; ++i) {
            PetscInt e = edges[i];
            PetscInt num_verts;
            const PetscInt *verts;

            // Traverse Edge -> Vertices
            DMPlexGetConeSize(dm, e, &num_verts);
            DMPlexGetCone(dm, e, &verts);

            for (PetscInt j = 0; j < num_verts; ++j) {
                PetscInt p = verts[j];

                // Defensively check if the point is within the coordinate section's chart range
                if (p >= cStart && p < cEnd) {
                    PetscInt dof, off;
                    PetscSectionGetDof(coordSection, p, &dof);

                    // Only pull coordinates if the point actually holds degrees of freedom
                    if (dof > 0) {
                        PetscSectionGetOffset(coordSection, p, &off);
                        cx += coords[off];
                        cy += coords[off + 1];
                        cz += coords[off + 2];
                        count++;
                    }
                }
            }
        }

        // "count" will normally be 6 for a triangle (3 edges * 2 vertices).
        // Dividing the sum by 6 mathematically yields the exact centroid.
        if (count > 0) {
            cx /= count;
            cy /= count;
            cz /= count;
            PetscInt val = 0;

            if (std::abs(cz) < EPSILON)
                val = 1;
            else if (std::abs(cz - DOMAIN_DEPTH) < EPSILON)
                val = 2;
            else if (std::abs(cx) < EPSILON)
                val = 3;
            else if (std::abs(cx - DOMAIN_WIDTH) < EPSILON)
                val = 4;
            else if (std::abs(cy) < EPSILON)
                val = 5;
            else if (std::abs(cy - DOMAIN_HEIGHT) < EPSILON)
                val = 6;

            if (val != 0)
                DMLabelSetValue(label, f, val);
        }
    }

    VecRestoreArrayRead(coordsVec, &coords);
    DMPlexLabelComplete(dm, label);
}

static PetscErrorCode RefineHook_LabelBoundaries_3D(DM /*dm*/, DM dmf, void * /*ctx*/) {
    LabelBoundaries_3D(dmf);
    DMRefineHookAdd(dmf, RefineHook_LabelBoundaries_3D, NULL, NULL);
    return 0;
}

static bool CheckMeshIntegrity_3D(MPI_Comm comm, const std::vector<Point3D> &points_local,
                                  const std::vector<Tetrahedron> &tets_local) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double local_total_volume = 0.0;
    long local_bad_edge_count = 0;
    double local_max_edge_len = 0.0;

    const double MAX_EDGE_RATIO = 3.5;
    const double THRESHOLD_LEN = TARGET_EDGE_LENGTH * MAX_EDGE_RATIO;
    int bad_edge_print_count = 0;

    for (const auto &t : tets_local) {
        const Point3D &p0 = points_local[t.v0], &p1 = points_local[t.v1], &p2 = points_local[t.v2], &p3 = points_local[t.v3];

        local_total_volume += calculate_tet_volume(p0, p1, p2, p3);

        double edges[6] = {std::sqrt(std::pow(p1.x - p0.x, 2) + std::pow(p1.y - p0.y, 2) + std::pow(p1.z - p0.z, 2)),
                           std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2) + std::pow(p2.z - p1.z, 2)),
                           std::sqrt(std::pow(p0.x - p2.x, 2) + std::pow(p0.y - p2.y, 2) + std::pow(p0.z - p2.z, 2)),
                           std::sqrt(std::pow(p3.x - p0.x, 2) + std::pow(p3.y - p0.y, 2) + std::pow(p3.z - p0.z, 2)),
                           std::sqrt(std::pow(p3.x - p1.x, 2) + std::pow(p3.y - p1.y, 2) + std::pow(p3.z - p1.z, 2)),
                           std::sqrt(std::pow(p3.x - p2.x, 2) + std::pow(p3.y - p2.y, 2) + std::pow(p3.z - p2.z, 2))};

        for (int i = 0; i < 6; ++i) {
            if (edges[i] > local_max_edge_len)
                local_max_edge_len = edges[i];
            if (edges[i] > THRESHOLD_LEN) {
                local_bad_edge_count++;
                if (bad_edge_print_count < 5) {
                    std::cout << "[Rank " << rank << "] BAD TET EDGE: Len " << edges[i] << " vs target " << TARGET_EDGE_LENGTH
                              << "\n";
                    bad_edge_print_count++;
                }
            }
        }
    }

    // Euler Characteristic Accumulators
    long local_owned_v = 0, local_owned_c = 0, local_owned_e = 0, local_owned_f = 0;

    // Counts Vertices strictly via geometric ownership
    for (const auto &p : points_local) {
        if (get_owner_rank_3D(p) == rank)
            local_owned_v++;
    }

    // Counts Cells (Tetrahedra) strictly via geometric ownership of the min-hash point
    for (const auto &t : tets_local) {
        uint64_t m = std::min({points_local[t.v0].unique_hash_id, points_local[t.v1].unique_hash_id,
                               points_local[t.v2].unique_hash_id, points_local[t.v3].unique_hash_id});
        const Point3D *p_min = &points_local[t.v0];
        if (m == points_local[t.v1].unique_hash_id)
            p_min = &points_local[t.v1];
        else if (m == points_local[t.v2].unique_hash_id)
            p_min = &points_local[t.v2];
        else if (m == points_local[t.v3].unique_hash_id)
            p_min = &points_local[t.v3];

        if (get_owner_rank_3D(*p_min) == rank)
            local_owned_c++;
    }

    // Extracts all unique local edges and faces
    std::vector<std::pair<int, int>> unique_edges;
    std::vector<int> valence;
    ComputeValenceAndEdges_3D(tets_local, points_local.size(), unique_edges, valence);

    std::vector<Face3D> unique_faces;
    unique_faces.reserve(tets_local.size() * 4);
    for (const auto &t : tets_local) {
        int f[4][3] = {{t.v0, t.v1, t.v2}, {t.v0, t.v1, t.v3}, {t.v0, t.v2, t.v3}, {t.v1, t.v2, t.v3}};
        for (int i = 0; i < 4; ++i) {
            std::sort(f[i], f[i] + 3);
            unique_faces.push_back({f[i][0], f[i][1], f[i][2]});
        }
    }
    std::sort(unique_faces.begin(), unique_faces.end());
    unique_faces.erase(std::unique(unique_faces.begin(), unique_faces.end()), unique_faces.end());

    // Distributed Hash Counting via Split MPI Exchanges for Edges and Faces
    auto mix_hash = [](uint64_t x) -> uint64_t {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    };

    std::vector<std::vector<uint64_t>> send_e(size), send_f(size);
    std::unordered_set<uint64_t> local_e_hashes, local_f_hashes;

    // Distribute Edges uniformly by hash
    for (const auto &edge : unique_edges) {
        uint64_t h1 = points_local[edge.first].unique_hash_id;
        uint64_t h2 = points_local[edge.second].unique_hash_id;
        if (h1 > h2)
            std::swap(h1, h2);

        uint64_t eh = mix_hash(h1 ^ mix_hash(h2));
        if (local_e_hashes.insert(eh).second) {
            send_e[eh % size].push_back(eh);
        }
    }

    // Distribute Faces uniformly by hash
    for (const auto &face : unique_faces) {
        uint64_t h[3] = {points_local[face.v0].unique_hash_id, points_local[face.v1].unique_hash_id,
                         points_local[face.v2].unique_hash_id};
        std::sort(h, h + 3);

        uint64_t fh = mix_hash(h[0] ^ mix_hash(h[1] ^ mix_hash(h[2])));
        if (local_f_hashes.insert(fh).second) {
            send_f[fh % size].push_back(fh);
        }
    }

    // Exchange Payload Sizes
    std::vector<int> send_counts_e(size), recv_counts_e(size);
    std::vector<int> send_counts_f(size), recv_counts_f(size);
    for (int r = 0; r < size; ++r) {
        send_counts_e[r] = send_e[r].size();
        send_counts_f[r] = send_f[r].size();
    }

    MPI_Alltoall(send_counts_e.data(), 1, MPI_INT, recv_counts_e.data(), 1, MPI_INT, comm);
    MPI_Alltoall(send_counts_f.data(), 1, MPI_INT, recv_counts_f.data(), 1, MPI_INT, comm);

    // Exchange Hash Payloads using Non-blocking Isend/Irecv
    std::vector<std::vector<uint64_t>> recv_e(size), recv_f(size);
    std::vector<MPI_Request> reqs;

    for (int r = 0; r < size; ++r) {
        if (recv_counts_e[r] > 0) {
            recv_e[r].resize(recv_counts_e[r]);
            MPI_Irecv(recv_e[r].data(), recv_counts_e[r], MPI_UINT64_T, r, 0, comm, &reqs.emplace_back());
        }
        if (recv_counts_f[r] > 0) {
            recv_f[r].resize(recv_counts_f[r]);
            MPI_Irecv(recv_f[r].data(), recv_counts_f[r], MPI_UINT64_T, r, 1, comm, &reqs.emplace_back());
        }
    }

    for (int r = 0; r < size; ++r) {
        if (send_counts_e[r] > 0) {
            MPI_Isend(send_e[r].data(), send_counts_e[r], MPI_UINT64_T, r, 0, comm, &reqs.emplace_back());
        }
        if (send_counts_f[r] > 0) {
            MPI_Isend(send_f[r].data(), send_counts_f[r], MPI_UINT64_T, r, 1, comm, &reqs.emplace_back());
        }
    }

    if (!reqs.empty()) {
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Deduplicate shared elements on designated hash target ranks
    std::unordered_set<uint64_t> owned_e, owned_f;
    for (int r = 0; r < size; ++r) {
        for (uint64_t eh : recv_e[r])
            owned_e.insert(eh);
        for (uint64_t fh : recv_f[r])
            owned_f.insert(fh);
    }

    local_owned_e = owned_e.size();
    local_owned_f = owned_f.size();

    // Gathers metrics to rank 0
    double global_total_volume, global_max_edge_len, global_bad_edge_count;
    long global_v, global_e, global_f, global_c;

    MPI_Reduce(&local_total_volume, &global_total_volume, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_bad_edge_count, &global_bad_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_max_edge_len, &global_max_edge_len, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    MPI_Reduce(&local_owned_v, &global_v, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_owned_e, &global_e, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_owned_f, &global_f, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_owned_c, &global_c, 1, MPI_LONG, MPI_SUM, 0, comm);

    int success = 1;
    if (rank == 0) {
        double expected_volume = DOMAIN_WIDTH * DOMAIN_HEIGHT * DOMAIN_DEPTH;
        long euler_characteristic = global_v - global_e + global_f - global_c;

        bool vol_pass = std::abs(global_total_volume - expected_volume) < 1e-5;
        bool edge_pass = (global_bad_edge_count == 0);
        bool euler_pass = (euler_characteristic == 1); // 1 for solid 3D domain (ball)

        if (!vol_pass || !edge_pass || !euler_pass) {
            success = 0;
            std::cout << "\n!!! 3D MESH INTEGRITY CHECK FAILED !!!\n";
            if (!vol_pass)
                std::cout << "  [FAIL] Total Volume: " << std::fixed << std::setprecision(6) << global_total_volume
                          << " (Expected " << expected_volume << ")\n";
            if (!euler_pass)
                std::cout << "  [FAIL] Topological Validation: V=" << global_v << ", E=" << global_e << ", F=" << global_f
                          << ", C=" << global_c << " -> Chi = " << euler_characteristic << " (Expected 1)\n";
            if (!edge_pass) {
                std::cout << "  [FAIL] Bad Edges: " << global_bad_edge_count << " edges > " << MAX_EDGE_RATIO << "x target.\n";
                std::cout << "         Max Edge: " << global_max_edge_len << "\n";
            }
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        }
    }
    MPI_Bcast(&success, 1, MPI_INT, 0, comm);
    return (success == 1);
}

void ComputeAndPrintStats_3D(MPI_Comm comm, int final_smooth_its, const std::vector<Point3D> &points_local,
                             const std::vector<Tetrahedron> &tets_local) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Save the caller's stream configuration
    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    // Connectivity and Unique Edges extraction
    std::vector<std::pair<int, int>> unique_edges;
    std::vector<int> vertex_valence;
    ComputeValenceAndEdges_3D(tets_local, points_local.size(), unique_edges, vertex_valence);

    long points_owned = 0;
    const int MAX_CONN = 60;
    long local_conn_bins[MAX_CONN] = {0};

    for (size_t i = 0; i < points_local.size(); ++i) {
        if (get_owner_rank_3D(points_local[i]) == rank) {
            points_owned++;
            int degree = vertex_valence[i];
            if (degree >= MAX_CONN)
                degree = MAX_CONN - 1;
            local_conn_bins[degree]++;
        }
    }

    long min_points_owned_global, max_points_owned_global, num_points_owned_global;
    MPI_Reduce(&points_owned, &min_points_owned_global, 1, MPI_LONG, MPI_MIN, 0, comm);
    MPI_Reduce(&points_owned, &max_points_owned_global, 1, MPI_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&points_owned, &num_points_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);

    long global_conn_bins[MAX_CONN] = {0};
    MPI_Reduce(local_conn_bins, global_conn_bins, MAX_CONN, MPI_LONG, MPI_SUM, 0, comm);

    // Tetrahedron Statistics
    long local_tet_count = tets_local.size(); // In distributed context, tets_local are ALL owned by this rank
    double local_min_volume = 1e30, local_max_volume = -1.0;
    double local_min_angle = 360.0, local_max_angle = -1.0;

    for (const auto &t : tets_local) {
        double vol = calculate_tet_volume(points_local[t.v0], points_local[t.v1], points_local[t.v2], points_local[t.v3]);
        if (vol < local_min_volume)
            local_min_volume = vol;
        if (vol > local_max_volume)
            local_max_volume = vol;

        double dihedrals[6];
        get_tet_dihedral_angles(points_local[t.v0], points_local[t.v1], points_local[t.v2], points_local[t.v3], dihedrals);
        for (double a : dihedrals) {
            if (a < local_min_angle)
                local_min_angle = a;
            if (a > local_max_angle)
                local_max_angle = a;
        }
    }

    long num_tets_owned_global;
    MPI_Reduce(&local_tet_count, &num_tets_owned_global, 1, MPI_LONG, MPI_SUM, 0, comm);

    // Edge Orientation Statistics
    std::vector<long> local_bins(18, 0);
    double local_total_edge_len = 0.0;
    long local_edge_count = 0;

    for (const auto &edge : unique_edges) {
        const Point3D &p1 = points_local[edge.first];
        const Point3D &p2 = points_local[edge.second];

        // Ensure edge is only counted by the rank that owns the vertex with the lowest hash id
        uint64_t min_hash = std::min(p1.unique_hash_id, p2.unique_hash_id);
        const Point3D &min_p = (p1.unique_hash_id == min_hash) ? p1 : p2;

        // Ties edge ownership to a hash combination of its two vertices
        if (get_owner_rank_3D(min_p) == rank) {
            double dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
            double len = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (len > 0) {
                local_total_edge_len += len;
                local_edge_count++;

                // Polar elevation angle (0 to 180 deg relative to Z-axis)
                double angle = std::acos(clamp_val_3D(dz / len)) * 180.0 / 3.14159265358979323846;
                int bin = std::min(17, std::max(0, static_cast<int>(angle / 10.0)));
                local_bins[bin]++;
            }
        }
    }

    if (local_tet_count == 0) {
        local_min_volume = 1e30;
        local_max_volume = -1.0;
        local_min_angle = 360.0;
        local_max_angle = -1.0;
    }

    // Global Reductions
    double global_min_volume, global_max_volume;
    double global_min_angle, global_max_angle;
    double global_total_edge_len;
    long global_edge_count;

    MPI_Reduce(&local_total_edge_len, &global_total_edge_len, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&local_edge_count, &global_edge_count, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_min_volume, &global_min_volume, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_max_volume, &global_max_volume, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_min_angle, &global_min_angle, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_max_angle, &global_max_angle, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    std::vector<long> global_bins(18);
    MPI_Reduce(local_bins.data(), global_bins.data(), 18, MPI_LONG, MPI_SUM, 0, comm);

    // Output Aggregated Stats to Rank 0
    if (rank == 0) {
        std::cout << "\n=== Parallel 3D Mesh Statistics ===\n";
        std::cout << "Final Smooth Iterations: " << final_smooth_its << "\n";
        std::cout << "Points:\n";
        std::cout << "  Total: " << num_points_owned_global << "\n";
        std::cout << "  Min per Rank: " << min_points_owned_global << "\n";
        std::cout << "  Max per Rank: " << max_points_owned_global << "\n";
        std::cout << "  Imbalance Ratio (Max/Avg): " << std::fixed << std::setprecision(4)
                  << (num_points_owned_global > 0 ? (double)max_points_owned_global / ((double)num_points_owned_global / size)
                                                  : 0)
                  << "\n";

        std::cout << "Connectivity (Valence - Tets per Vertex):\n";
        for (int i = 1; i < MAX_CONN; ++i) {
            if (global_conn_bins[i] > 0) {
                double pct = 100.0 * global_conn_bins[i] / num_points_owned_global;
                std::cout << "  Degree " << std::setw(2) << i << ": " << std::setw(8) << global_conn_bins[i] << " ("
                          << std::fixed << std::setprecision(2) << pct << "%)\n";
            }
        }

        std::cout << "Tetrahedra:\n";
        std::cout << "  Total: " << num_tets_owned_global << "\n";
        std::cout << "  Volume Min: " << std::scientific << global_min_volume << "\n";
        std::cout << "  Volume Max: " << std::scientific << global_max_volume << "\n";
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "  Volume Ratio: " << (global_min_volume > 0 ? global_max_volume / global_min_volume : -1.0) << "\n";
        std::cout << "  Angle Min: " << global_min_angle << " deg\n";
        std::cout << "  Angle Max: " << global_max_angle << " deg\n";
        std::cout << "  Avg Edge Len: " << (global_edge_count > 0 ? global_total_edge_len / global_edge_count : 0.0) << "\n";

        std::cout << "Edge Polar Orientations (10 deg bins from Z-axis):\n";
        long total_edges = 0;
        for (long c : global_bins)
            total_edges += c;

        if (total_edges > 0) {
            for (int i = 0; i < 18; ++i) {
                double pct = 100.0 * global_bins[i] / total_edges;
                std::cout << "  " << std::setw(3) << (i * 10) << "-" << std::setw(3) << ((i + 1) * 10)
                          << " deg: " << std::fixed << std::setprecision(2) << pct << "%\n";
            }
        }
        std::cout << "=================================\n";
        std::cout.copyfmt(oldState);
    }
}

// Processes a single tile of the domain
// search_min          t_min          interior_min        interior_max        t_max         search_max
//      |----------------|-----------------|------------------|-----------------|----------------|
static void process_tile_3D(MPI_Comm comm, int final_smooth_its, int tile_x, int tile_y, int tile_z, double factor, double dt,
                            std::vector<Point3D> &points_on_owned_tets_and_orphans, std::vector<Tetrahedron> &tets_owned) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    // The size of each tile
    double tile_s_x = DOMAIN_WIDTH / TILE_DIM_X;
    double tile_s_y = DOMAIN_HEIGHT / TILE_DIM_Y;
    double tile_s_z = DOMAIN_DEPTH / TILE_DIM_Z;

    // The tile's local search space
    double t_min_x = tile_x * tile_s_x, t_max_x = (tile_x + 1) * tile_s_x;
    double t_min_y = tile_y * tile_s_y, t_max_y = (tile_y + 1) * tile_s_y;
    double t_min_z = tile_z * tile_s_z, t_max_z = (tile_z + 1) * tile_s_z;

    // Padding for the tile's local search space
    double pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8);

    // The tile's local search space with padding
    double search_min_x = t_min_x - pad, search_max_x = t_max_x + pad;
    double search_min_y = t_min_y - pad, search_max_y = t_max_y + pad;
    double search_min_z = t_min_z - pad, search_max_z = t_max_z + pad;

    // The tile's interior search space with padding
    // (Points that don't need to be ghosted as are safely within the tile)
    double interior_min_x = t_min_x + pad, interior_max_x = t_max_x - pad;
    double interior_min_y = t_min_y + pad, interior_max_y = t_max_y - pad;
    double interior_min_z = t_min_z + pad, interior_max_z = t_max_z - pad;

    // Clamp search indices to valid global ranges to eliminate out-of-bounds ghost points
    int min_ix = std::max(0, (int)std::floor(search_min_x / TARGET_EDGE_LENGTH));
    int max_ix =
        std::min((int)std::floor(DOMAIN_WIDTH / TARGET_EDGE_LENGTH), (int)std::ceil(search_max_x / TARGET_EDGE_LENGTH));
    int min_iy = std::max(0, (int)std::floor(search_min_y / TARGET_EDGE_LENGTH));
    int max_iy =
        std::min((int)std::floor(DOMAIN_HEIGHT / TARGET_EDGE_LENGTH), (int)std::ceil(search_max_y / TARGET_EDGE_LENGTH));
    int min_iz = std::max(0, (int)std::floor(search_min_z / TARGET_EDGE_LENGTH));
    int max_iz =
        std::min((int)std::floor(DOMAIN_DEPTH / TARGET_EDGE_LENGTH), (int)std::ceil(search_max_z / TARGET_EDGE_LENGTH));

    // Global limits to determine domain termination
    int global_max_ix = static_cast<int>(std::floor(DOMAIN_WIDTH / TARGET_EDGE_LENGTH));
    int global_max_iy = static_cast<int>(std::floor(DOMAIN_HEIGHT / TARGET_EDGE_LENGTH));
    int global_max_iz = static_cast<int>(std::floor(DOMAIN_DEPTH / TARGET_EDGE_LENGTH));

    std::vector<Point3D> points_with_halos;
    double exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.25);

    // Loop through the clamped grid nodes
    for (int iz = min_iz; iz <= max_iz; ++iz) {
        for (int iy = min_iy; iy <= max_iy; ++iy) {
            for (int ix = min_ix; ix <= max_ix; ++ix) {
                // Check if this specific grid vertex lies on any of the 6 outer domain faces
                bool is_boundary =
                    (ix == 0 || ix == global_max_ix || iy == 0 || iy == global_max_iy || iz == 0 || iz == global_max_iz);
                if (is_boundary) {
                    double cx = (ix == global_max_ix) ? DOMAIN_WIDTH : ix * TARGET_EDGE_LENGTH;
                    double cy = (iy == global_max_iy) ? DOMAIN_HEIGHT : iy * TARGET_EDGE_LENGTH;
                    double cz = (iz == global_max_iz) ? DOMAIN_DEPTH : iz * TARGET_EDGE_LENGTH;
                    // If it falls within this tile's padded search window, grab it
                    if (cx >= search_min_x && cx <= search_max_x && cy >= search_min_y && cy <= search_max_y &&
                        cz >= search_min_z && cz <= search_max_z) {
                        points_with_halos.emplace_back(cx, cy, cz, create_point_with_unique_hash_id_3D(ix, iy, iz, 1));
                    }
                } else {
                    // Jitter the point slightly to avoid overlapping with other points
                    uint64_t seed = ((uint64_t)ix << 40) | ((uint64_t)iy << 20) | iz;
                    uint64_t h = splitmix64_3D(seed);
                    double cx = (ix + 0.1 + next_double_3D(h) * 0.8) * TARGET_EDGE_LENGTH;
                    double cy = (iy + 0.1 + next_double_3D(h) * 0.8) * TARGET_EDGE_LENGTH;
                    double cz = (iz + 0.1 + next_double_3D(h) * 0.8) * TARGET_EDGE_LENGTH;

                    // Only keep the interior candidate if it's far enough from all boundary faces
                    if (cx >= exclusion && cx <= DOMAIN_WIDTH - exclusion && cy >= exclusion &&
                        cy <= DOMAIN_HEIGHT - exclusion && cz >= exclusion && cz <= DOMAIN_DEPTH - exclusion) {
                        if (cx >= search_min_x && cx <= search_max_x && cy >= search_min_y && cy <= search_max_y &&
                            cz >= search_min_z && cz <= search_max_z) {
                            points_with_halos.emplace_back(cx, cy, cz, create_point_with_unique_hash_id_3D(ix, iy, iz, 0));
                        }
                    }
                }
            }
        }
    }

    // Synchronize the search window with the tile's interior
    double sync_margin = pad - (TARGET_EDGE_LENGTH * 1.5);
    double s_min_x = t_min_x - sync_margin, s_max_x = t_max_x + sync_margin;
    double s_min_y = t_min_y - sync_margin, s_max_y = t_max_y + sync_margin;
    double s_min_z = t_min_z - sync_margin, s_max_z = t_max_z + sync_margin;

    std::vector<Tetrahedron> tets_with_halos;

    for (int iter = 0; iter < ANNEAL_ITERS + final_smooth_its; ++iter) {
        if (iter < ANNEAL_ITERS)
            apply_jitter_3D(points_with_halos, START_JITTER, iter);

        // Sort deterministically before local meshing to prevent rank drift
        std::sort(points_with_halos.begin(), points_with_halos.end(),
                  [](const Point3D &a, const Point3D &b) { return a.unique_hash_id < b.unique_hash_id; });

        TetrahedralizePointCloud(points_with_halos, tets_with_halos);
        relax_points_lloyd_3D(points_with_halos, tets_with_halos, factor, s_min_x, s_min_y, s_min_z, s_max_x, s_max_y,
                              s_max_z);
        relax_points_spring_3D(points_with_halos, tets_with_halos, dt, s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z);

        ResolveBoundaryOwnership_3D(comm, points_with_halos, s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z,
                                    interior_min_x, interior_min_y, interior_min_z, interior_max_x, interior_max_y,
                                    interior_max_z, pad);
    }

    TetrahedralizePointCloud(points_with_halos, tets_with_halos);
    ResolveBoundaryOwnership_3D(comm, points_with_halos, s_min_x, s_min_y, s_min_z, s_max_x, s_max_y, s_max_z, interior_min_x,
                                interior_min_y, interior_min_z, interior_max_x, interior_max_y, interior_max_z, pad);

    std::vector<int> local_to_owned_idx(points_with_halos.size(), -1);

    // Filter to true ownership
    for (const auto &tet : tets_with_halos) {
        const Point3D &p0 = points_with_halos[tet.v0];
        const Point3D &p1 = points_with_halos[tet.v1];
        const Point3D &p2 = points_with_halos[tet.v2];
        const Point3D &p3 = points_with_halos[tet.v3];

        uint64_t min_hash_id = std::min({p0.unique_hash_id, p1.unique_hash_id, p2.unique_hash_id, p3.unique_hash_id});
        const Point3D *min_p = &p0;
        if (p1.unique_hash_id == min_hash_id)
            min_p = &p1;
        if (p2.unique_hash_id == min_hash_id)
            min_p = &p2;
        if (p3.unique_hash_id == min_hash_id)
            min_p = &p3;

        if (get_owner_rank_3D(*min_p) == comm_rank) {
            double vol = calculate_tet_volume(p0, p1, p2, p3);
            if (vol < 1e-13) {
                continue; // Drop degenerate element before it reaches PETSc
            }

            Tetrahedron new_t;
            int *src_idx[4] = {(int *)&tet.v0, (int *)&tet.v1, (int *)&tet.v2, (int *)&tet.v3};
            int *dst_idx[4] = {&new_t.v0, &new_t.v1, &new_t.v2, &new_t.v3};

            for (int k = 0; k < 4; ++k) {
                int local_idx = *src_idx[k];
                if (local_to_owned_idx[local_idx] == -1) {
                    local_to_owned_idx[local_idx] = points_on_owned_tets_and_orphans.size();
                    points_on_owned_tets_and_orphans.push_back(points_with_halos[local_idx]);
                }
                *dst_idx[k] = local_to_owned_idx[local_idx];
            }
            tets_owned.push_back(new_t);
        }
    }

    for (size_t i = 0; i < points_with_halos.size(); ++i) {
        if (get_owner_rank_3D(points_with_halos[i]) == comm_rank && local_to_owned_idx[i] == -1) {
            points_on_owned_tets_and_orphans.push_back(points_with_halos[i]);
        }
    }

    // Free vectors
    points_with_halos.clear();
    tets_with_halos.clear();
    local_to_owned_idx.clear();
    std::vector<Point3D>().swap(points_with_halos);
    std::vector<Tetrahedron>().swap(tets_with_halos);
    std::vector<int>().swap(local_to_owned_idx);
}

// Merges boundary nodes connected by highly compressed edges
static void CollapseBoundarySlivers_3D(std::vector<Point3D> &points, std::vector<Tetrahedron> &tets) {
    double threshold = 0.1 * TARGET_EDGE_LENGTH;
    double thresh_sq = threshold * threshold;
    int n = points.size();

    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i)
        parent[i] = i;

    // Union-find root lookup
    auto find_set = [&](int i) {
        int root = i;
        while (root != parent[root])
            root = parent[root];
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    };

    auto is_boundary = [&](const Point3D &p) {
        return (std::abs(p.x) < EPSILON || std::abs(p.x - DOMAIN_WIDTH) < EPSILON || std::abs(p.y) < EPSILON ||
                std::abs(p.y - DOMAIN_HEIGHT) < EPSILON || std::abs(p.z) < EPSILON || std::abs(p.z - DOMAIN_DEPTH) < EPSILON);
    };

    // Extracts all unique edges from the current tets
    std::vector<std::pair<int, int>> edges;
    edges.reserve(tets.size() * 6);
    for (const auto &t : tets) {
        edges.push_back({t.v0, t.v1});
        edges.push_back({t.v0, t.v2});
        edges.push_back({t.v0, t.v3});
        edges.push_back({t.v1, t.v2});
        edges.push_back({t.v1, t.v3});
        edges.push_back({t.v2, t.v3});
    }

    // Evaluates edges and collapse if both points sit on the boundary and are too close
    for (const auto &e : edges) {
        int u = e.first;
        int v = e.second;
        int pu = find_set(u);
        int pv = find_set(v);

        if (pu != pv) {
            const Point3D &p1 = points[pu];
            const Point3D &p2 = points[pv];

            if (is_boundary(p1) && is_boundary(p2)) {
                double dx = p1.x - p2.x, dy = p1.y - p2.y, dz = p1.z - p2.z;
                if (dx * dx + dy * dy + dz * dz < thresh_sq) {
                    // Makes the merge deterministic across parallel ranks using the unique hash
                    if (p1.unique_hash_id < p2.unique_hash_id) {
                        parent[pv] = pu;
                    } else {
                        parent[pu] = pv;
                    }
                }
            }
        }
    }

    // Updates the tetrahedra vertices and discards any that have collapsed into lines/points
    std::vector<Tetrahedron> new_tets;
    new_tets.reserve(tets.size());
    for (const auto &t : tets) {
        int v0 = find_set(t.v0);
        int v1 = find_set(t.v1);
        int v2 = find_set(t.v2);
        int v3 = find_set(t.v3);

        // Only keep if all 4 vertices remain distinct
        if (v0 != v1 && v0 != v2 && v0 != v3 && v1 != v2 && v1 != v3 && v2 != v3) {
            Tetrahedron nt = {v0, v1, v2, v3};
            new_tets.push_back(nt);
        }
    }

    tets = std::move(new_tets);
}

DM CreateDMPlex3D(MPI_Comm comm, const std::vector<Point3D> &points_on_owned_tets_and_orphans,
                  const std::vector<Tetrahedron> &tets_owned) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    PetscInt neg_one = -1;

    PetscInt num_points = points_on_owned_tets_and_orphans.size();
    std::vector<PetscInt> global_ids(num_points, neg_one);
    PetscInt num_points_owned = 0;

    for (int i = 0; i < num_points; ++i) {
        if (get_owner_rank_3D(points_on_owned_tets_and_orphans[i]) == comm_rank)
            num_points_owned++;
    }

    PetscInt start_id = 0;
    MPI_Exscan(&num_points_owned, &start_id, 1, MPIU_INT, MPI_SUM, comm);
    PetscInt current_id = start_id;

    for (int i = 0; i < num_points; ++i) {
        if (get_owner_rank_3D(points_on_owned_tets_and_orphans[i]) == comm_rank)
            global_ids[i] = current_id++;
    }

    std::vector<std::vector<uint64_t>> send_ids(comm_size);
    std::vector<std::vector<int>> send_req_indices(comm_size);

    for (int i = 0; i < num_points; ++i) {
        if (global_ids[i] == neg_one) {
            int owner = get_owner_rank_3D(points_on_owned_tets_and_orphans[i]);
            send_ids[owner].push_back(points_on_owned_tets_and_orphans[i].unique_hash_id);
            send_req_indices[owner].push_back(i);
        }
    }

    std::vector<int> send_counts(comm_size), recv_counts(comm_size);
    for (int r = 0; r < comm_size; ++r)
        send_counts[r] = send_ids[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<std::vector<uint64_t>> recv_ids(comm_size);
    std::vector<MPI_Request> requests;

    for (int r = 0; r < comm_size; ++r) {
        if (recv_counts[r] > 0) {
            recv_ids[r].resize(recv_counts[r]);
            MPI_Irecv(recv_ids[r].data(), recv_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, comm, &requests.emplace_back());
        }
        if (r != comm_rank && send_counts[r] > 0) {
            MPI_Isend(send_ids[r].data(), send_counts[r] * sizeof(uint64_t), MPI_BYTE, r, 100, comm, &requests.emplace_back());
        }
    }
    if (!requests.empty())
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();

    std::vector<std::pair<uint64_t, PetscInt>> points_owned_l2g_map;
    points_owned_l2g_map.reserve(num_points_owned);
    for (int i = 0; i < num_points; ++i) {
        if (get_owner_rank_3D(points_on_owned_tets_and_orphans[i]) == comm_rank) {
            points_owned_l2g_map.emplace_back(points_on_owned_tets_and_orphans[i].unique_hash_id, global_ids[i]);
        }
    }

    // Sorts to enable O(log N) cache-friendly lookups
    std::sort(
        points_owned_l2g_map.begin(), points_owned_l2g_map.end(),
        [](const std::pair<uint64_t, PetscInt> &a, const std::pair<uint64_t, PetscInt> &b) { return a.first < b.first; });

    std::vector<std::vector<PetscInt>> send_answers(comm_size);
    for (int r = 0; r < comm_size; ++r) {
        if (recv_counts[r] > 0) {
            send_answers[r].resize(recv_counts[r]);
            for (int k = 0; k < recv_counts[r]; ++k) {
                uint64_t target_id = recv_ids[r][k];

                // Binary search
                auto it = std::lower_bound(points_owned_l2g_map.begin(), points_owned_l2g_map.end(), target_id,
                                           [](const std::pair<uint64_t, PetscInt> &p, uint64_t val) { return p.first < val; });

                if (it != points_owned_l2g_map.end() && it->first == target_id) {
                    send_answers[r][k] = it->second;
                } else {
                    send_answers[r][k] = neg_one;
                }
            }
        }
    }

    std::vector<std::vector<PetscInt>> recv_answers(comm_size);
    for (int r = 0; r < comm_size; ++r) {
        if (r != comm_rank && send_counts[r] > 0) {
            recv_answers[r].resize(send_counts[r]);
            MPI_Irecv(recv_answers[r].data(), send_counts[r] * sizeof(PetscInt), MPI_BYTE, r, 101, comm,
                      &requests.emplace_back());
        }
        if (recv_counts[r] > 0) {
            MPI_Isend(send_answers[r].data(), recv_counts[r] * sizeof(PetscInt), MPI_BYTE, r, 101, comm,
                      &requests.emplace_back());
        }
    }
    if (!requests.empty())
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for (int r = 0; r < comm_size; ++r) {
        if (send_counts[r] > 0) {
            for (int k = 0; k < send_counts[r]; ++k)
                global_ids[send_req_indices[r][k]] = recv_answers[r][k];
        }
    }

    PetscInt num_tets_owned = tets_owned.size();
    std::vector<PetscInt> cells(num_tets_owned * 4);
    for (int i = 0; i < num_tets_owned; ++i) {
        cells[i * 4 + 0] = global_ids[tets_owned[i].v0];
        cells[i * 4 + 1] = global_ids[tets_owned[i].v1];
        cells[i * 4 + 2] = global_ids[tets_owned[i].v2];
        cells[i * 4 + 3] = global_ids[tets_owned[i].v3];
    }

    std::vector<PetscReal> coords_points_owned;
    coords_points_owned.reserve(num_points_owned * 3);
    for (int i = 0; i < num_points; ++i) {
        if (get_owner_rank_3D(points_on_owned_tets_and_orphans[i]) == comm_rank) {
            coords_points_owned.push_back(points_on_owned_tets_and_orphans[i].x);
            coords_points_owned.push_back(points_on_owned_tets_and_orphans[i].y);
            coords_points_owned.push_back(points_on_owned_tets_and_orphans[i].z);
        }
    }

    DM dm = nullptr;

    // Provides valid memory addresses for optional PETSc outputs
    PetscSF vertexSF = nullptr;
    PetscInt *verticesAdj = nullptr;

    // Safeguard against empty vectors returning invalid pointers
    PetscInt *cells_ptr = cells.empty() ? nullptr : cells.data();
    PetscReal *coords_ptr = coords_points_owned.empty() ? nullptr : coords_points_owned.data();

    // (Using PetscCallAbort since the function returns a DM, not a PetscErrorCode)
    PetscCallAbort(comm, DMPlexCreateFromCellListParallelPetsc(comm,
                                                               3,                // dim: 3D Mesh
                                                               num_tets_owned,   // numCells
                                                               num_points_owned, // numVertices
                                                               PETSC_DECIDE, // NVertices: Let PETSc compute the global total
                                                               4,            // numCorners
                                                               PETSC_TRUE,   // interpolate
                                                               cells_ptr,    // cells
                                                               3,            // spaceDim
                                                               coords_ptr,   // vertexCoords
                                                               &vertexSF,    // vertexSF: Safe pointer reference
                                                               &verticesAdj, // verticesAdj: Safe pointer reference
                                                               &dm));

    // Clean up optional outputs if PETSc allocated them
    if (vertexSF)
        PetscCallAbort(comm, PetscSFDestroy(&vertexSF));
    if (verticesAdj)
        PetscCallAbort(comm, PetscFree(verticesAdj));

    DMPlexDistributeSetDefault(dm, PETSC_FALSE);

    // Memory reclamation
    send_ids.clear();
    recv_ids.clear();
    send_answers.clear();
    recv_answers.clear();
    std::vector<std::vector<uint64_t>>().swap(send_ids);
    std::vector<std::vector<uint64_t>>().swap(recv_ids);
    std::vector<std::vector<PetscInt>>().swap(send_answers);
    std::vector<std::vector<PetscInt>>().swap(recv_answers);

    return dm;
}

// Generates a structured 3D grid of points
std::vector<Point3D> GenerateStructuredGrid(int nx, int ny, int nz, double spacing) {
    std::vector<Point3D> points;
    points.reserve(nx * ny * nz);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                // Generate a deterministic hash
                uint64_t hash_id = create_point_with_unique_hash_id_3D(i, j, k, 0);

                double x = i * spacing;
                double y = j * spacing;
                double z = k * spacing;

                // Clamp to the domain boundaries to avoid floating point overshoot
                if (x > DOMAIN_WIDTH)
                    x = DOMAIN_WIDTH;
                if (y > DOMAIN_HEIGHT)
                    y = DOMAIN_HEIGHT;
                if (z > DOMAIN_DEPTH)
                    z = DOMAIN_DEPTH;

                points.emplace_back(x, y, z, hash_id);
            }
        }
    }
    return points;
}

// (filename must include .vtu extension)
void WriteMeshVTU(DM dm, const std::string &filename, MPI_Comm comm) {
    PetscViewer viewer;
    PetscViewerVTKOpen(comm, filename.c_str(), FILE_MODE_WRITE, &viewer);
    DMView(dm, viewer);
    PetscViewerDestroy(&viewer);
}

/*
Can view this in paraview with:
/home/sdargavi/projects/dependencies/petsc_main/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5
then using the XDMF reader with:
paraview box_mesh.xmf
(filename must include .h5 extension)
*/
void WriteMeshH5(DM dm, const std::string &filename, MPI_Comm comm) {
    PetscViewer viewer;
    PetscViewerHDF5Open(comm, filename.c_str(), FILE_MODE_WRITE, &viewer);
    DMView(dm, viewer);
    PetscViewerDestroy(&viewer);
}

// Write output if requested (filename must not include an extension)
void WriteMesh(DM dm, const std::string &filename, MPI_Comm comm, PetscInt write_mesh, PetscBool print_stats) {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    if (write_mesh == 1) {
#ifdef PETSC_HAVE_HDF5
        if (comm_rank == 0 && print_stats) {
            std::cout << "Writing out mesh to " + filename + ".h5...\n";
        }
        WriteMeshH5(dm, filename + ".h5", comm);
#else
        if (comm_rank == 0) {
            std::cerr << "-write_mesh=1 (H5 output mode) not available without HDF5 enabled in PETSc. Consider setting "
                         "write_mesh=2 (VTU output mode).\n";
        }
#endif
    }
    // Can directly view the output in Paraview
    if (write_mesh == 2) {
        if (comm_rank == 0 && print_stats) {
            std::cout << "Writing out mesh to " + filename + ".vtu...\n";
        }
        // Using your existing VTU writer
        WriteMeshVTU(dm, filename + ".vtu", comm);
    }
}

DM GenerateBoxMeshDM_3D(MPI_Comm comm, double target_edge_length, double domain_width, double domain_height,
                        double domain_depth, int final_smooth_its, double factor, double dt, PetscBool integrity_check,
                        PetscBool print_stats) {
    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    TARGET_EDGE_LENGTH = target_edge_length;
    DOMAIN_WIDTH = domain_width;
    DOMAIN_HEIGHT = domain_height;
    DOMAIN_DEPTH = domain_depth;

    // 2^20 is the maximum index value for the index
    if (TARGET_EDGE_LENGTH < DOMAIN_WIDTH / 1048575.0) {
        if (comm_rank == 0) {
            std::cerr << "WARNING: Target edge length is too small. It exceeds the 20-bit index limit per dimension.\n";
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    int best_tx = 1, best_ty = 1, best_tz = 1;
    double min_cut_area = 1e30;

    // Brute-force find the optimal 3D processor grid that minimizes internal halo communication surfaces
    for (int x = 1; x <= comm_size; ++x) {
        if (comm_size % x == 0) {
            for (int y = 1; y <= comm_size / x; ++y) {
                if ((comm_size / x) % y == 0) {
                    int z = comm_size / (x * y);

                    // Internal cut area formula for a 3D bounding box topology
                    double cut_area = (x - 1) * domain_height * domain_depth + (y - 1) * domain_width * domain_depth +
                                      (z - 1) * domain_width * domain_height;

                    if (cut_area < min_cut_area) {
                        min_cut_area = cut_area;
                        best_tx = x;
                        best_ty = y;
                        best_tz = z;
                    }
                }
            }
        }
    }

    // Only warns if forced into 1D because the number itself is prime
    if (comm_rank == 0) {
        int ones_count = (best_tx == 1) + (best_ty == 1) + (best_tz == 1);

        if (ones_count >= 2 && comm_size > 2) {
            // Check if comm_size is actually a prime number
            bool is_prime = true;
            for (int i = 2; i * i <= comm_size; ++i) {
                if (comm_size % i == 0) {
                    is_prime = false;
                    break;
                }
            }

            if (is_prime) {
                std::cout << "\n[BoxMeshDM_3D] Note: MPI rank count (" << comm_size << ") is a prime number.\n"
                          << "-> Decomposed into a 1D slice layout: " << best_tx << "x" << best_ty << "x" << best_tz << "\n\n";
            } else {
                std::cout << "\n[BoxMeshDM_3D] Note: Domain geometry constraints favor a 1D slice layout.\n"
                          << "-> Decomposed grid optimized to: " << best_tx << "x" << best_ty << "x" << best_tz << "\n\n";
            }
        }
    }

    TILE_DIM_X = best_tx;
    TILE_DIM_Y = best_ty;
    TILE_DIM_Z = best_tz;

    std::vector<Point3D> points_on_owned_tets_and_orphans;
    std::vector<Tetrahedron> tets_owned;

    // Calculates the 3D tile coordinates for this specific rank
    int tz = comm_rank / (TILE_DIM_X * TILE_DIM_Y);
    int ty = (comm_rank / TILE_DIM_X) % TILE_DIM_Y;
    int tx = comm_rank % TILE_DIM_X;

    process_tile_3D(comm, final_smooth_its, tx, ty, tz, factor, dt, points_on_owned_tets_and_orphans, tets_owned);

    CollapseBoundarySlivers_3D(points_on_owned_tets_and_orphans, tets_owned);

    if (integrity_check) {
        if (!CheckMeshIntegrity_3D(comm, points_on_owned_tets_and_orphans, tets_owned))
            return NULL;
    }

    if (print_stats) {
        ComputeAndPrintStats_3D(comm, final_smooth_its, points_on_owned_tets_and_orphans, tets_owned);
    }

    DM dm = CreateDMPlex3D(comm, points_on_owned_tets_and_orphans, tets_owned);
    PetscObjectSetName((PetscObject)dm, "Mesh3D");

    LabelBoundaries_3D(dm);
    DMRefineHookAdd(dm, RefineHook_LabelBoundaries_3D, NULL, NULL);

    return dm;
}

// Main Driver
// (Generates a 3D mesh using either the values parsed via the commandline, otherwise using default values)
#ifdef STANDALONE_MESH_GEN
int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    // Parse command line options with defaults
    double target_len = 0.02;
    PetscBool set;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-target_edge_length", &target_len, &set));

    PetscInt write_mesh = 0;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-write_mesh", &write_mesh, NULL));

    PetscBool integrity_check = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-integrity_check", &integrity_check, NULL));

    PetscBool print_stats = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_stats", &print_stats, NULL));

    PetscInt final_smooth_its = 4;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-final_smooth_its", &final_smooth_its, &set));

    double domain_width = 1.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_width", &domain_width, &set));

    double domain_height = 1.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_height", &domain_height, &set));

    double domain_depth = 1.0;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_depth", &domain_depth, &set));

    double lloyd_factor = 0.5;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-lloyd_factor", &lloyd_factor, &set));

    double spring_dt = 0.2;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-spring_dt", &spring_dt, &set));

    // Generate the DMPlex for this mesh
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, target_len, domain_width, domain_height, domain_depth, final_smooth_its,
                                 lloyd_factor, spring_dt, integrity_check, print_stats);

    // Check a valid mesh has been generated
    if (dm) {
        // Write output if requested
        WriteMesh(dm, "box_mesh_3D", PETSC_COMM_WORLD, write_mesh, print_stats);
        PetscCall(DMDestroy(&dm));
    } else {
        PetscCall(PetscFinalize());
        return EXIT_FAILURE;
    }

    PetscCall(PetscFinalize());
    return 0;
}
#endif
