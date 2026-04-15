#ifndef BOXMESHDM_H
#define BOXMESHDM_H

#include <petscdmplex.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generates a 2D unstructured mesh of a box [0,width]x[0,height] and returns a PETSc DM
 * 
 * @param comm The MPI communicator to use.
 * @param target_edge_length The target edge length for the mesh.
 * @param width The width of the box domain.
 * @param height The height of the box domain.
 * @param final_smooth_its The number of final smoothing iterations to perform.
 * @param integrity_check Whether to perform an integrity check on the mesh.
 * @param print_stats Whether to print mesh generation statistics.
 * @return DM The generated distributed DMPlex.
 */
DM GenerateBoxMeshDM(MPI_Comm comm, double target_edge_length, double width, double height, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats);
#ifdef __cplusplus
}
#endif

#endif