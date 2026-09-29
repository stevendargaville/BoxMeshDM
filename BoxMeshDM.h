#ifndef BOXMESHDM_H
#define BOXMESHDM_H

#include <petscdmplex.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generates a 2D unstructured mesh of a box [0,width]x[0,height] and returns a PETSc DM
 * 
 * The boundary edges are labelled in "Face Sets" and "markers" with 1=Bottom (y=0),
 * 2=Right (x=width), 3=Top (y=height), 4=Left (x=0).
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

/**
 * @brief As GenerateBoxMeshDM, but with control over how the rank grid nests
 *
 * The rank grid is built as a coarse grid of comm_size/agglomeration_factor tiles, each split
 * into a compact sub-block of agglomeration_factor fine tiles. The agglomeration_factor fine
 * ranks making up coarse group j are exactly ranks [j*agglomeration_factor,
 * (j+1)*agglomeration_factor), so merging every agglomeration_factor consecutive ranks (whose
 * global DOFs form one contiguous range) reproduces the decomposition a plain run on
 * comm_size/agglomeration_factor ranks would have chosen - with no repartitioning.
 *
 * @param comm The MPI communicator to use.
 * @param target_edge_length The target edge length for the mesh.
 * @param width The width of the box domain.
 * @param height The height of the box domain.
 * @param final_smooth_its The number of final smoothing iterations to perform.
 * @param integrity_check Whether to perform an integrity check on the mesh.
 * @param print_stats Whether to print mesh generation statistics.
 * @param agglomeration_factor Number of fine ranks per coarse group. Must be at least 1 and
 *        must divide the size of comm. 1 gives exactly the same mesh as GenerateBoxMeshDM.
 * @return DM The generated distributed DMPlex.
 */
DM GenerateBoxMeshDMAgglom(MPI_Comm comm, double target_edge_length, double width, double height, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats, int agglomeration_factor);

/**
 * @brief Generates a 3D unstructured tetrahedral mesh of a box [0,width]x[0,height]x[0,depth]
 *        and returns a PETSc DM
 *
 * Requires PETSc configured with TetGen (--download-tetgen); without it this stops with an
 * error. The boundary faces are labelled in "Face Sets" and "markers" with the values PETSc's
 * DMPlexCreateBoxMesh uses: 1=Bottom (z=0), 2=Top (z=depth), 3=Front (y=0), 4=Back (y=height),
 * 5=Right (x=width), 6=Left (x=0).
 *
 * @param comm The MPI communicator to use.
 * @param target_edge_length The target edge length for the mesh.
 * @param width The width (x extent) of the box domain.
 * @param height The height (y extent) of the box domain.
 * @param depth The depth (z extent) of the box domain.
 * @param final_smooth_its The number of final smoothing iterations to perform.
 * @param integrity_check Whether to perform an integrity check on the mesh.
 * @param print_stats Whether to print mesh generation statistics.
 * @return DM The generated distributed DMPlex.
 */
DM GenerateBoxMeshDM3D(MPI_Comm comm, double target_edge_length, double width, double height, double depth, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats);

/**
 * @brief As GenerateBoxMeshDM3D, but with control over how the rank grid nests
 *
 * As GenerateBoxMeshDMAgglom: the rank grid is a coarse grid of comm_size/agglomeration_factor
 * tiles, each split into a compact sub-block of agglomeration_factor fine tiles, and the fine
 * ranks making up coarse group j are exactly ranks [j*agglomeration_factor,
 * (j+1)*agglomeration_factor).
 *
 * @param comm The MPI communicator to use.
 * @param target_edge_length The target edge length for the mesh.
 * @param width The width (x extent) of the box domain.
 * @param height The height (y extent) of the box domain.
 * @param depth The depth (z extent) of the box domain.
 * @param final_smooth_its The number of final smoothing iterations to perform.
 * @param integrity_check Whether to perform an integrity check on the mesh.
 * @param print_stats Whether to print mesh generation statistics.
 * @param agglomeration_factor Number of fine ranks per coarse group. Must be at least 1 and
 *        must divide the size of comm. 1 gives exactly the same mesh as GenerateBoxMeshDM3D.
 * @return DM The generated distributed DMPlex.
 */
DM GenerateBoxMeshDM3DAgglom(MPI_Comm comm, double target_edge_length, double width, double height, double depth, int final_smooth_its, PetscBool integrity_check, PetscBool print_stats, int agglomeration_factor);
#ifdef __cplusplus
}
#endif

#endif