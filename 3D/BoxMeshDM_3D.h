#ifndef TET_MESH_GEN_H
#define TET_MESH_GEN_H

#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

extern double DOMAIN_WIDTH;
extern double DOMAIN_HEIGHT;
extern double DOMAIN_DEPTH;
extern double TARGET_EDGE_LENGTH;

struct Point3D {
  double x, y, z;
  uint64_t unique_hash_id = 0;

  Point3D() : x(0), y(0), z(0), unique_hash_id(0) {}
  Point3D(double x_, double y_, double z_, uint64_t id = 0)
      : x(x_), y(y_), z(z_), unique_hash_id(id) {}
};

struct Tetrahedron {
  int v0, v1, v2, v3;
};

// Generates the new serial halo-decomposed point cloud
std::vector<Point3D> GenerateMesh3D_Serial(int dim_x, int dim_y, int dim_z,
                                           double factor = 0.5,
                                           double dt = 0.2);

// Computes pure 3D Delaunay tetrahedralization of a point cloud
void TetrahedralizePointCloud(const std::vector<Point3D> &point_cloud,
                              std::vector<Tetrahedron> &out_tetrahedra);

// Generates a structured 3D grid of points
std::vector<Point3D> GenerateStructuredGrid(int nx, int ny, int nz,
                                            double spacing);

// Writes tetrahedral mesh to VTU file (unstructured grid)
void WriteTetrahedralMeshVTU(const std::vector<Point3D> &points,
                             const std::vector<Tetrahedron> &tets,
                             const std::string &filename,
                             MPI_Comm comm = MPI_COMM_SELF);

// Lloyd-smoothing: Moves vertices towards the volume-weighted centroid
void relax_points_lloyd_3D(std::vector<Point3D> &points,
                           const std::vector<Tetrahedron> &tets,
                           double factor = 0.5);

// Spring-Force Relaxation: Attempts to equalize all edges to TARGET_EDGE_LENGTH
void relax_points_spring_3D(std::vector<Point3D> &points,
                            const std::vector<Tetrahedron> &tets,
                            double dt = 0.2);

#endif // TET_MESH_GEN_H
