#include <petscsys.h>
#include "box_2D_gen_unstruc_mesh.h"

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    DM dm;
    PetscErrorCode ierr;

    // Set target mesh edge length
    double target_edge_length = 0.007;
    // Set the number of smoothing iterations
    PetscInt final_smooth_its = 4;
    // Check the integrity of the mesh and error if not valid
    PetscBool integrity_check = PETSC_TRUE;
    // Print global mesh statistics from MPI rank 0
    PetscBool print_stats = PETSC_TRUE;

    // Generate the mesh stored in a parallel PETSc DM on the MPI_Comm PETSC_COMM_WORLD
    dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, target_edge_length, final_smooth_its, integrity_check, print_stats);

    // Enable the use of command line options for this DM
    ierr = DMSetFromOptions(dm);

    // Clean up
    ierr = DMDestroy(&dm);
    (void)ierr;

    PetscCall(PetscFinalize());
    return 0;
}
