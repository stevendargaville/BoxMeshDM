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

    PetscPrintf(PETSC_COMM_WORLD, "=== Testing Square Mesh (1.0 x 1.0) ===\n");
    
    // Generate square mesh
    double square_width = 1.0;
    double square_height = 1.0;
    
    dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, target_edge_length, final_smooth_its, integrity_check, print_stats, square_width, square_height);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Square mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Square mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Rectangular Mesh (2.0 x 1.5) ===\n");
    
    // Generate rectangular mesh
    double rect_width = 2.0;
    double rect_height = 1.5;
    
    dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, target_edge_length, 
                          final_smooth_its, integrity_check, 
                          print_stats, rect_width, rect_height);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscCall(PetscFinalize());
    return 0;
}
