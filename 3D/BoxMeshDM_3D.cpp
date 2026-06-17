#include "BoxMeshDM_3D.h"
#include "tetgen.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <petscdmplex.h>
#include <set>
#include <unordered_map>
#include <vector>

const double EPSILON = 1e-13;
const double START_JITTER = 0.30;
const int ANNEAL_ITERS = 3;
const int FINAL_SMOOTH_ITS = 4;
double DOMAIN_WIDTH = 1.0;
double DOMAIN_HEIGHT = 1.0;
double DOMAIN_DEPTH = 1.0;
double TARGET_EDGE_LENGTH = 0.1;

// Pack: [Type: 2 bits] [ix: 20 bits] [iy: 20 bits] [iz: 20 bits]
uint64_t create_point_with_unique_hash_id_3D(int ix, int iy, int iz, int type) {
  uint64_t id = ((uint64_t)(type & 0x3) << 60) |
                ((uint64_t)(ix & 0xFFFFF) << 40) |
                ((uint64_t)(iy & 0xFFFFF) << 20) | ((uint64_t)(iz & 0xFFFFF));
  return id;
}

// Simple splitmix64 for deterministic RNG
uint64_t splitmix64_3D(uint64_t &x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

double next_double_3D(uint64_t &state) {
  return (splitmix64_3D(state) >> 11) * (1.0 / 9007199254740992.0);
}

bool apply_boundary_constraint_3D(Point3D &p, double &dx, double &dy,
                                  double &dz) {
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

std::vector<Point3D> process_tile_3D(int tile_x, int tile_y, int tile_z,
                                     int dim_x, int dim_y, int dim_z,
                                     double pad) {
  double tile_s_x = DOMAIN_WIDTH / dim_x;
  double tile_s_y = DOMAIN_HEIGHT / dim_y;
  double tile_s_z = DOMAIN_DEPTH / dim_z;

  double search_min_x = tile_x * tile_s_x - pad;
  double search_max_x = (tile_x + 1) * tile_s_x + pad;
  double search_min_y = tile_y * tile_s_y - pad;
  double search_max_y = (tile_y + 1) * tile_s_y + pad;
  double search_min_z = tile_z * tile_s_z - pad;
  double search_max_z = (tile_z + 1) * tile_s_z + pad;

  // Global limits to determine domain termination
  int global_max_ix = std::round(DOMAIN_WIDTH / TARGET_EDGE_LENGTH);
  int global_max_iy = std::round(DOMAIN_HEIGHT / TARGET_EDGE_LENGTH);
  int global_max_iz = std::round(DOMAIN_DEPTH / TARGET_EDGE_LENGTH);

  // Clamp search indices to valid global ranges to eliminate out-of-bounds
  // ghost points
  int min_ix = std::max(0, (int)std::floor(search_min_x / TARGET_EDGE_LENGTH));
  int max_ix = std::min(global_max_ix,
                        (int)std::ceil(search_max_x / TARGET_EDGE_LENGTH));
  int min_iy = std::max(0, (int)std::floor(search_min_y / TARGET_EDGE_LENGTH));
  int max_iy = std::min(global_max_iy,
                        (int)std::ceil(search_max_y / TARGET_EDGE_LENGTH));
  int min_iz = std::max(0, (int)std::floor(search_min_z / TARGET_EDGE_LENGTH));
  int max_iz = std::min(global_max_iz,
                        (int)std::ceil(search_max_z / TARGET_EDGE_LENGTH));

  double exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.25);
  std::vector<Point3D> tile_points;

  // Loop through the clamped grid nodes
  for (int iz = min_iz; iz <= max_iz; ++iz) {
    for (int iy = min_iy; iy <= max_iy; ++iy) {
      for (int ix = min_ix; ix <= max_ix; ++ix) {

        // --- BOUNDARY VERTICES ---
        // Check if this specific grid vertex lies on any of the 6 outer domain
        // faces
        bool is_boundary_vertex =
            (ix == 0 || ix == global_max_ix || iy == 0 || iy == global_max_iy ||
             iz == 0 || iz == global_max_iz);

        if (is_boundary_vertex) {
          double cx = ix * TARGET_EDGE_LENGTH;
          double cy = iy * TARGET_EDGE_LENGTH;
          double cz = iz * TARGET_EDGE_LENGTH;

          // If it falls within this tile's padded search window, grab it
          if (cx >= search_min_x && cx <= search_max_x && cy >= search_min_y &&
              cy <= search_max_y && cz >= search_min_z && cz <= search_max_z) {

            uint64_t id = create_point_with_unique_hash_id_3D(
                ix, iy, iz, 1); // type 1 = boundary
            tile_points.emplace_back(cx, cy, cz, id);
          }
        }

        // --- INTERIOR CELL CANDIDATES ---
        // Decouple from boundary check
        // (Every valid cell volume can host an interior point which is then
        // safely filtered by the exclusion distance)
        if (ix < global_max_ix && iy < global_max_iy && iz < global_max_iz) {
          uint64_t h = ((uint64_t)ix << 40) | ((uint64_t)iy << 20) | iz;
          h = splitmix64_3D(h);

          double r1 = 0.1 + next_double_3D(h) * 0.8;
          double r2 = 0.1 + next_double_3D(h) * 0.8;
          double r3 = 0.1 + next_double_3D(h) * 0.8;

          double cx = (ix + r1) * TARGET_EDGE_LENGTH;
          double cy = (iy + r2) * TARGET_EDGE_LENGTH;
          double cz = (iz + r3) * TARGET_EDGE_LENGTH;

          // Only keep the interior candidate if it's far enough from ALL
          // boundary faces
          if (cx >= exclusion && cx <= DOMAIN_WIDTH - exclusion &&
              cy >= exclusion && cy <= DOMAIN_HEIGHT - exclusion &&
              cz >= exclusion && cz <= DOMAIN_DEPTH - exclusion) {

            if (cx >= search_min_x && cx <= search_max_x &&
                cy >= search_min_y && cy <= search_max_y &&
                cz >= search_min_z && cz <= search_max_z) {

              uint64_t id = create_point_with_unique_hash_id_3D(
                  ix, iy, iz, 0); // type 0 = interior
              tile_points.emplace_back(cx, cy, cz, id);
            }
          }
        }
      }
    }
  }

  // Apply deterministic jitter to this tile's point subset
  apply_jitter_3D(tile_points, START_JITTER, 0);

  return tile_points;
}

// Calculates the volume of a tetrahedron
double calculate_tet_volume(const Point3D &p0, const Point3D &p1,
                            const Point3D &p2, const Point3D &p3) {
  // Volume = 1/6 * |(p1-p0) . ((p2-p0) x (p3-p0))|
  double v1x = p1.x - p0.x, v1y = p1.y - p0.y, v1z = p1.z - p0.z;
  double v2x = p2.x - p0.x, v2y = p2.y - p0.y, v2z = p2.z - p0.z;
  double v3x = p3.x - p0.x, v3y = p3.y - p0.y, v3z = p3.z - p0.z;

  // Cross product (p2-p0) x (p3-p0)
  double cx = v2y * v3z - v2z * v3y;
  double cy = v2z * v3x - v2x * v3z;
  double cz = v2x * v3y - v2y * v3x;

  // Dot product with (p1-p0)
  double det = std::abs(v1x * cx + v1y * cy + v1z * cz);
  return det / 6.0;
}

// Clamps to prevent acos domain errors
static double clamp_val_3D(double v) {
  return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
}

// Calculate the dihedral angles of a given tetrahedron
static void get_tet_dihedral_angles(const Point3D &p0, const Point3D &p1,
                                    const Point3D &p2, const Point3D &p3,
                                    double angles[6]) {
  auto normal = [](const Point3D &a, const Point3D &b, const Point3D &c) {
    double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    return Point3D(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  };

  Point3D n[4] = {normal(p1, p2, p3), normal(p0, p3, p2), normal(p0, p1, p3),
                  normal(p0, p2, p1)};

  auto mag = [](const Point3D &vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
  };
  auto dot = [](const Point3D &a, const Point3D &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  };

  int idx = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      double m1 = mag(n[i]), m2 = mag(n[j]);
      if (m1 < 1e-14 || m2 < 1e-14) {
        angles[idx++] = 0.0;
        continue;
      }
      double cos_theta = dot(n[i], n[j]) / (m1 * m2);
      angles[idx++] =
          std::acos(clamp_val_3D(-cos_theta)) * 180.0 / 3.14159265358979323846;
    }
  }
}

void ComputeAndPrintStats_3D(int final_smooth_its,
                             const std::vector<Point3D> &points,
                             const std::vector<Tetrahedron> &tets) {

  // Connectivity and Unique Edges
  std::vector<int> valence(points.size(), 0);
  std::set<std::pair<int, int>> unique_edges;
  int edge_pairs[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};

  for (const auto &t : tets) {
    int v[4] = {t.v0, t.v1, t.v2, t.v3};
    for (auto &pair : edge_pairs) {
      int i1 = v[pair[0]];
      int i2 = v[pair[1]];
      valence[i1]++;
      valence[i2]++;
      unique_edges.insert({std::min(i1, i2), std::max(i1, i2)});
    }
  }

  // Statistics Calculation
  double min_vol = 1e30, max_vol = -1.0;
  double min_angle = 360.0, max_angle = -1.0;
  double total_edge_len = 0.0;
  std::vector<long> bins(18, 0);

  // Tetrahedron loop
  for (const auto &t : tets) {
    double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2],
                                      points[t.v3]);
    min_vol = std::min(min_vol, vol);
    max_vol = std::max(max_vol, vol);

    double dihedrals[6];
    get_tet_dihedral_angles(points[t.v0], points[t.v1], points[t.v2],
                            points[t.v3], dihedrals);
    for (double a : dihedrals) {
      min_angle = std::min(min_angle, a);
      max_angle = std::max(max_angle, a);
    }
  }

  // Edge statistics loop
  for (const auto &edge : unique_edges) {
    const Point3D &p1 = points[edge.first];
    const Point3D &p2 = points[edge.second];
    double dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
    double len = std::sqrt(dx * dx + dy * dy + dz * dz);
    total_edge_len += len;

    // Polar angle [0, 180) from Z-axis
    double angle =
        std::acos(clamp_val_3D(dz / len)) * 180.0 / 3.14159265358979323846;
    int bin = std::min(17, std::max(0, static_cast<int>(angle / 10.0)));
    bins[bin]++;
  }

  // Final Print
  std::cout << "\n=== Serial 3D Mesh Statistics ===\n";
  std::cout << "Iterations: " << final_smooth_its << "\n";
  std::cout << "Points: " << points.size() << " | Tets: " << tets.size()
            << "\n";
  std::cout << "Volume: Min " << min_vol << " / Max " << max_vol << "\n";
  std::cout << "Dihedral Range: [" << min_angle << ", " << max_angle
            << "] deg\n";
  std::cout << "Avg Edge Length: "
            << (unique_edges.empty() ? 0 : total_edge_len / unique_edges.size())
            << "\n";

  std::cout << "Edge Polar Orientations (10 deg bins):\n";
  for (int i = 0; i < 18; ++i) {
    std::cout << "  " << (i * 10) << "-" << ((i + 1) * 10) << ": " << bins[i]
              << "\n";
  }
  std::cout << "=================================\n";
}

// 3D Lloyd-smoothing
// Moves vertices towards the volume-weighted centroid of surrounding tets
// Optimising element shape and aspect ratios
void relax_points_lloyd_3D(std::vector<Point3D> &points,
                           const std::vector<Tetrahedron> &tets,
                           double factor) {
  int n = points.size();
  // wx, wy, wz:  Volume-Weighted Centroid Accumalators
  // w_sum: Sum of Weights / Total Area
  std::vector<double> wx(n, 0.0), wy(n, 0.0), wz(n, 0.0), w_sum(n, 0.0);

  for (const auto &tet : tets) {
    const Point3D &p0 = points[tet.v0];
    const Point3D &p1 = points[tet.v1];
    const Point3D &p2 = points[tet.v2];
    const Point3D &p3 = points[tet.v3];

    // Tetrahedron centroid
    double cx = (p0.x + p1.x + p2.x + p3.x) * 0.25;
    double cy = (p0.y + p1.y + p2.y + p3.y) * 0.25;
    double cz = (p0.z + p1.z + p2.z + p3.z) * 0.25;

    double vol = calculate_tet_volume(p0, p1, p2, p3);

    int v[4] = {tet.v0, tet.v1, tet.v2, tet.v3};
    for (int i = 0; i < 4; ++i) {
      wx[v[i]] += vol * cx;
      wy[v[i]] += vol * cy;
      wz[v[i]] += vol * cz;
      w_sum[v[i]] += vol;
    }
  }

  for (int i = 0; i < n; ++i) {
    if (w_sum[i] < 1e-12)
      continue; // Skip zero-volume anomalies

    // Target Centroid Coordinates
    // I.e. the volume-weighted average centroid
    // of the Vornoi cells surrounding vertex i
    double tx = wx[i] / w_sum[i];
    double ty = wy[i] / w_sum[i];
    double tz = wz[i] / w_sum[i];

    // Final displacements vectors (prior to applying boundary constraints)
    double dx = (tx - points[i].x) * factor;
    double dy = (ty - points[i].y) * factor;
    double dz = (tz - points[i].z) * factor;

    Point3D temp_p = points[i];
    bool was_boundary = apply_boundary_constraint_3D(temp_p, dx, dy, dz);

    temp_p.x += dx;
    temp_p.y += dy;
    temp_p.z += dz;

    if (!was_boundary) {
      keep_interior_point_inside_3D(temp_p);
    } else {
      // Strict clamp to prevent float drift on boundaries
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

// 3D Spring-Force Relaxation
// Attempts to equalize all edges to TARGET_EDGE_LENGTH
void relax_points_spring_3D(std::vector<Point3D> &points,
                            const std::vector<Tetrahedron> &tets, double dt) {
  int n = points.size();
  std::vector<double> force_x(n, 0.0), force_y(n, 0.0), force_z(n, 0.0);
  std::vector<int> valence(n, 0);

  // 6 edges per tetrahedron
  int edge_pairs[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};

  for (const auto &tet : tets) {
    int v[4] = {tet.v0, tet.v1, tet.v2, tet.v3};
    for (int e = 0; e < 6; ++e) {
      int idx1 = v[edge_pairs[e][0]];
      int idx2 = v[edge_pairs[e][1]];

      double dx = points[idx2].x - points[idx1].x;
      double dy = points[idx2].y - points[idx1].y;
      double dz = points[idx2].z - points[idx1].z;
      double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < 1e-14)
        continue;

      double force_mag = (dist - TARGET_EDGE_LENGTH);

      double nx = dx / dist;
      double ny = dy / dist;
      double nz = dz / dist;

      double fx = force_mag * nx;
      double fy = force_mag * ny;
      double fz = force_mag * nz;

      force_x[idx1] += fx;
      force_y[idx1] += fy;
      force_z[idx1] += fz;
      force_x[idx2] -= fx;
      force_y[idx2] -= fy;
      force_z[idx2] -= fz;
      valence[idx1]++;
      valence[idx2]++;
    }
  }

  for (int i = 0; i < n; ++i) {
    if (valence[i] == 0)
      continue;

    // Normalised Force
    double fx = force_x[i] / valence[i];
    double fy = force_y[i] / valence[i];
    double fz = force_z[i] / valence[i];

    double dx = fx * dt;
    double dy = fy * dt;
    double dz = fz * dt;

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

// Tetrahedralizes a point cloud using TetGen
void TetrahedralizePointCloud(const std::vector<Point3D> &point_cloud,
                              std::vector<Tetrahedron> &out_tetrahedra) {
  tetgenio in, out;

  in.numberofpoints = static_cast<int>(point_cloud.size());
  in.pointlist = new double[in.numberofpoints * 3];

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    in.pointlist[i * 3 + 0] = point_cloud[i].x;
    in.pointlist[i * 3 + 1] = point_cloud[i].y;
    in.pointlist[i * 3 + 2] = point_cloud[i].z;
  }

  // C++ interface
  tetgenbehavior behavior;
  char switches[] = "Q";
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

  // delete[] in.pointlist;
}

std::vector<Point3D> GenerateMesh3D(int dim_x, int dim_y, int dim_z,
                                    double factor, double dt) {
  double pad = TARGET_EDGE_LENGTH * 4.0; // arbitrary halo padding for testing
  std::unordered_map<uint64_t, Point3D> global_point_map;

  // Initial Cloud Generation
  for (int tz = 0; tz < dim_z; ++tz) {
    for (int ty = 0; ty < dim_y; ++ty) {
      for (int tx = 0; tx < dim_x; ++tx) {
        std::vector<Point3D> tile_points =
            process_tile_3D(tx, ty, tz, dim_x, dim_y, dim_z, pad);

        // Merge tile points into global map to filter out halo duplicates
        for (const auto &p : tile_points) {
          global_point_map[p.unique_hash_id] = p;
        }
      }
    }
  }

  // Flatten to a standard vector for TetGen
  std::vector<Point3D> final_cloud;
  final_cloud.reserve(global_point_map.size());
  for (const auto &pair : global_point_map) {
    final_cloud.push_back(pair.second);
  }

  // Iterative Smoothing Process
  std::vector<Tetrahedron> tets;

  // Anneal: Jitter -> Triangulate -> Smooth
  for (int iter = 0; iter < ANNEAL_ITERS; ++iter) {
    apply_jitter_3D(final_cloud, START_JITTER, iter);
    TetrahedralizePointCloud(final_cloud, tets);
    relax_points_lloyd_3D(final_cloud, tets, factor);
    relax_points_spring_3D(final_cloud, tets, dt);
  }

  // Final Smooth Loop (No Jitter)
  for (int iter = 0; iter < FINAL_SMOOTH_ITS; ++iter) {
    TetrahedralizePointCloud(final_cloud, tets);
    relax_points_lloyd_3D(final_cloud, tets, factor);
    relax_points_spring_3D(final_cloud, tets, dt);
  }

  return final_cloud;
}

// Generates a structured 3D grid of points
std::vector<Point3D> GenerateStructuredGrid(int nx, int ny, int nz,
                                            double spacing) {
  std::vector<Point3D> points;
  points.reserve(nx * ny * nz);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        points.emplace_back(i * spacing, j * spacing, k * spacing);
      }
    }
  }
  return points;
}

// Writes mesh to VTU file
void WriteTetrahedralMeshVTU(const std::vector<Point3D> &points,
                             const std::vector<Tetrahedron> &tets,
                             const std::string &filename, MPI_Comm comm) {
  DM dm;
  PetscInt dim = 3, numCells = tets.size(), numVertices = points.size();
  std::vector<PetscInt> cells(numCells * 4);
  std::vector<PetscReal> coords(numVertices * 3);

  for (size_t i = 0; i < tets.size(); ++i) {
    cells[i * 4 + 0] = tets[i].v0;
    cells[i * 4 + 1] = tets[i].v1;
    cells[i * 4 + 2] = tets[i].v2;
    cells[i * 4 + 3] = tets[i].v3;
  }
  for (size_t i = 0; i < points.size(); ++i) {
    coords[i * 3 + 0] = points[i].x;
    coords[i * 3 + 1] = points[i].y;
    coords[i * 3 + 2] = points[i].z;
  }

  DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, 4, PETSC_TRUE,
                                cells.data(), dim, coords.data(), &dm);

  PetscViewer viewer;
  PetscViewerVTKOpen(comm, filename.c_str(), FILE_MODE_WRITE, &viewer);
  DMView(dm, viewer);
  PetscViewerDestroy(&viewer);
  DMDestroy(&dm);
}
