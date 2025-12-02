#ifndef BOX_2D_GEN_UNSTRUC_MESH_H
#define BOX_2D_GEN_UNSTRUC_MESH_H

#include <petscdmplex.h>

/**
 * @brief Generates a 2D unstructured mesh of a box [0,1]x[0,1] and returns a PETSc DM
 * 
 * @param comm The MPI communicator to use.
 * @param target_edge_length The target edge length for the mesh.
 * @param print_stats Whether to print mesh generation statistics.
 * @return DM The generated distributed DMPlex.
 */
DM GenerateBoxMeshDM(MPI_Comm comm, double target_edge_length, PetscBool print_stats);

#endif