#include "BoxMeshDM_3D.h"
#include <petscsys.h>

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    DM dm;
    PetscErrorCode ierr;

    // Test parameters
    double target_edge_length = 0.25;
    PetscInt final_smooth_its = 4;
    double lloyd_factor = 0.5;
    double spring_dt = 0.2;

    // Explicitly test the print_stats functionality in the primary API
    PetscBool integrity_check = PETSC_TRUE;
    PetscBool print_stats = PETSC_TRUE;

    PetscPrintf(PETSC_COMM_WORLD, "=== Testing 3D Cube Mesh (1.0 x 1.0 x 1.0) ===\n");

    // Generate cube mesh
    double width = 1.0;
    double height = 1.0;
    double depth = 1.0;

    dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, target_edge_length, width, height, depth, final_smooth_its, lloyd_factor,
                              spring_dt, integrity_check, print_stats);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Cube mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Cube mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing 3D Rectangular Prism Mesh (2.0 x 1.5 x 1.0) ===\n");

    // Generate rectangular prism mesh
    double rect_w = 2.0;
    double rect_h = 1.5;
    double rect_d = 1.0;

    dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, target_edge_length, rect_w, rect_h, rect_d, final_smooth_its, lloyd_factor,
                              spring_dt, integrity_check, print_stats);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular prism mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular prism mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscCall(PetscFinalize());
    return 0;
}
