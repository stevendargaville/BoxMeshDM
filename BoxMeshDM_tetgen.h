#ifndef BOXMESHDM_TETGEN_H
#define BOXMESHDM_TETGEN_H

// Private interface between BoxMeshDM.cpp and the TetGen backend in BoxMeshDM_tetgen.cpp.
// Not part of the public API. Kept in its own translation unit so tetgen.h (its REAL macro,
// lowercase macros, global orient3d/exactinit and C++ exceptions) never meets the 2D code.

// Through petscsys.h, which includes mpi.h with the MPI C++ bindings switched off
#include <petscsys.h>
#include <vector>

// Delaunay tetrahedralisation of num_points points (xyz holds x, y, z of each point in turn).
// On return tets holds 4 input point indices per tetrahedron, in TetGen's vertex order: every
// tetrahedron is positively oriented in the right-hand sense, (p1-p0).((p2-p0)x(p3-p0)) > 0.
// Aborts on comm if TetGen fails, if an input point is left out of every tetrahedron (a
// duplicate or near-coincident point), if a tetrahedron is exactly degenerate or has an
// unexpected orientation, or if PETSc was built without TetGen.
// Returns the number of tetrahedra.
int BoxMeshDM_Delaunay3D(MPI_Comm comm, int num_points, const double *xyz, std::vector<int>& tets);

#endif
