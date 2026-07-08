#ifndef TET_MESH_GEN_H
#define TET_MESH_GEN_H

#include <mpi.h>
#include <petscdm.h>

// C++ only API
#ifdef __cplusplus

#include <cstdint>
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
    Point3D(double x_, double y_, double z_, uint64_t id = 0) : x(x_), y(y_), z(z_), unique_hash_id(id) {}
};

struct Tetrahedron {
    int v0, v1, v2, v3;
};

// Computes pure 3D Delaunay tetrahedralization of a point cloud
void TetrahedralizePointCloud(const std::vector<Point3D> &point_cloud, std::vector<Tetrahedron> &out_tetrahedra);

// Generates a structured 3D grid of points
std::vector<Point3D> GenerateStructuredGrid(int nx, int ny, int nz, double spacing);

// Decoupled PETSc DMPlex generator for unit testing topology
DM CreateDMPlex3D(MPI_Comm comm, const std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets);

// Writes tetrahedral mesh to VTU file (unstructured grid)
void WriteMeshVTU(DM dm, const std::string &filename, MPI_Comm comm);

// Writes tetrahedral mesh to h5 file
void WriteMeshH5(DM dm, const std::string &filename, MPI_Comm comm);

void WriteMesh(DM dm, const std::string &filename, MPI_Comm comm, PetscInt write_mesh, PetscBool print_stats);

// Lloyd-smoothing: Moves vertices towards the volume-weighted centroid
void relax_points_lloyd_3D(std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets, double factor, double min_x,
                           double min_y, double min_z, double max_x, double max_y, double max_z);

// Spring-Force Relaxation: Attempts to equalize all edges to TARGET_EDGE_LENGTH
void relax_points_spring_3D(std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets, double dt, double min_x,
                            double min_y, double min_z, double max_x, double max_y, double max_z);

void ComputeAndPrintStats_3D(MPI_Comm comm, int final_smooth_its, const std::vector<Point3D> &points,
                             const std::vector<Tetrahedron> &tets);

double calculate_tet_volume(const Point3D &p0, const Point3D &p1, const Point3D &p2, const Point3D &p3);

// Exposed for direct boundary condition coverage tracking
bool apply_boundary_constraint_3D(Point3D &p, double &dx, double &dy, double &dz);

int get_owner_rank_3D(const Point3D &p);

#endif /* __cplusplus */

// Public C / C++ API:
// Main generator
PETSC_EXTERN DM GenerateBoxMeshDM_3D(MPI_Comm comm, double target_edge_length, double domain_width, double domain_height,
                                     double domain_depth, int final_smooth_its, double factor, double dt,
                                     PetscBool integrity_check, PetscBool print_stats);

#endif // TET_MESH_GEN_H
