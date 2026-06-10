#ifndef TET_MESH_GEN_H
#define TET_MESH_GEN_H

#include <mpi.h>
#include <string>
#include <vector>

struct Point3D {
  double x, y, z;
  Point3D() : x(0), y(0), z(0) {}
  Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

struct Tetrahedron {
  int v0, v1, v2, v3;
};

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

#endif // TET_MESH_GEN_H
