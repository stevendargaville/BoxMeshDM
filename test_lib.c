#include <petscsys.h>
#include "BoxMeshDM.h"

int main(int argc, char** argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    DM dm;
    PetscErrorCode ierr;

    // Set target mesh edge length
    double target_edge_length = 0.007;
    // Set the number of smoothing iterations
    int final_smooth_its = 4;
    // Check the integrity of the mesh and error if not valid
    PetscBool integrity_check = PETSC_TRUE;
    // Print global mesh statistics from MPI rank 0
    PetscBool print_stats = PETSC_TRUE;

    PetscPrintf(PETSC_COMM_WORLD, "=== Testing Square Mesh (1.0 x 1.0) ===\n");
    
    // Generate square mesh
    double square_width = 1.0;
    double square_height = 1.0;

    dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, target_edge_length, square_width, square_height, final_smooth_its, integrity_check, print_stats);

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
    
    dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, target_edge_length, rect_width, rect_height, final_smooth_its, integrity_check, print_stats);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Rectangular mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Agglomerated Square Mesh (1.0 x 1.0) ===\n");

    // Generate a square mesh with all the ranks in a single coarse group. Calling this from a C
    // translation unit also checks the agglomeration entry point has C linkage.
    PetscMPIInt comm_size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &comm_size));

    dm = GenerateBoxMeshDMAgglom(PETSC_COMM_WORLD, target_edge_length, square_width, square_height, final_smooth_its, integrity_check, print_stats, (int)comm_size);

    if (dm) {
        PetscPrintf(PETSC_COMM_WORLD, "Agglomerated mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Agglomerated mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Boundary Labels After Refinement ===\n");

    // Refine a unit square after a mesh of a different size has been generated. The
    // refinement hook must label against the square's own domain, so all four
    // boundaries (1=Bottom, 2=Right, 3=Top, 4=Left) must still have labelled edges
    DM dm_square, dm_other, dm_refined;
    dm_square = GenerateBoxMeshDM(PETSC_COMM_WORLD, 0.01, 1.0, 1.0, 0, PETSC_FALSE, PETSC_FALSE);
    dm_other = GenerateBoxMeshDM(PETSC_COMM_WORLD, 0.01, 2.0, 1.5, 0, PETSC_FALSE, PETSC_FALSE);
    PetscCall(DMRefine(dm_square, PETSC_COMM_WORLD, &dm_refined));

    DMLabel face_sets;
    PetscInt edge_start, edge_end, local_counts[4] = {0, 0, 0, 0}, counts[4];
    PetscCall(DMGetLabel(dm_refined, "Face Sets", &face_sets));
    PetscCall(DMPlexGetHeightStratum(dm_refined, 1, &edge_start, &edge_end));
    for (PetscInt e = edge_start; e < edge_end; ++e) {
        PetscInt val;
        PetscCall(DMLabelGetValue(face_sets, e, &val));
        if (val >= 1 && val <= 4) local_counts[val - 1]++;
    }
    PetscCallMPI(MPI_Allreduce(local_counts, counts, 4, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(DMDestroy(&dm_refined));
    PetscCall(DMDestroy(&dm_other));
    PetscCall(DMDestroy(&dm_square));

    // Each side of the unit square has 1/0.01 = 100 edges, doubled by the refinement
    for (int i = 0; i < 4; ++i) {
        if (counts[i] != 200) {
            PetscPrintf(PETSC_COMM_WORLD, "Boundary %d has %" PetscInt_FMT " labelled edges after refinement, expected 200!\n", i + 1, counts[i]);
            PetscCall(PetscFinalize());
            return 1;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, "Refined boundary labels correct.\n");

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing for Memory Leaks ===\n");

    // Generate and destroy a small mesh twice. The first call can leave behind
    // one-off PETSc allocations (class registration etc), so only the second is
    // checked. Current usage is only tracked in debug builds (or with -malloc_debug),
    // otherwise both readings are zero and this passes trivially.
    PetscLogDouble mem_before, mem_after;
    for (int i = 0; i < 2; ++i) {
        PetscCall(PetscMallocGetCurrentUsage(&mem_before));
        dm = GenerateBoxMeshDM(PETSC_COMM_WORLD, 0.02, 1.0, 1.0, 0, PETSC_FALSE, PETSC_FALSE);
        PetscCall(DMDestroy(&dm));
        PetscCall(PetscMallocGetCurrentUsage(&mem_after));
    }
    PetscLogDouble leaked = mem_after - mem_before, max_leaked;
    PetscCallMPI(MPI_Allreduce(&leaked, &max_leaked, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
    if (max_leaked != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Leaked %g bytes generating and destroying a mesh!\n", max_leaked);
        PetscCall(PetscFinalize());
        return 1;
    }
    PetscPrintf(PETSC_COMM_WORLD, "No memory leaked.\n");

    PetscCall(PetscFinalize());
    return 0;
}
