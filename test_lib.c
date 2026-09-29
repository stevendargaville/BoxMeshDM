#include <petscsys.h>
#include "BoxMeshDM.h"

// Run PETSc's own consistency checks on a generated DM: cone/support symmetry, the skeleton and
// faces of every cell against the reference simplex, cell orientation and volume, the point SF,
// and that the cones of points shared between ranks agree
static PetscErrorCode CheckDMPlex(DM dm)
{
    PetscFunctionBeginUser;
    PetscCall(DMPlexCheckSymmetry(dm));
    PetscCall(DMPlexCheckSkeleton(dm, 0));
    PetscCall(DMPlexCheckFaces(dm, 0));
    PetscCall(DMPlexCheckGeometry(dm));
    PetscCall(DMPlexCheckPointSF(dm, NULL, PETSC_FALSE));
    PetscCall(DMPlexCheckInterfaceCones(dm));
    PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_TETGEN)
// Count the faces carrying each "Face Sets" value 1..num_walls, summed over all ranks. A face on
// the domain boundary has a single cell, so it lives on one rank only and is counted once.
// num_errors counts faces labelled although they are not on the boundary (or on the boundary
// but unlabelled, using PETSc's own DMPlexMarkBoundaryFaces), faces whose "markers" value
// differs from their "Face Sets" value, and values outside 1..num_walls
static PetscErrorCode CountFaceSets(DM dm, PetscInt num_walls, PetscInt *counts, PetscInt *num_errors)
{
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);
    DMLabel face_sets, markers, boundary;
    PetscInt face_start, face_end, local_counts[6] = {0, 0, 0, 0, 0, 0}, local_errors = 0;

    PetscFunctionBeginUser;
    PetscCheck(num_walls <= 6, comm, PETSC_ERR_ARG_OUTOFRANGE, "At most 6 walls");
    PetscCall(DMGetLabel(dm, "Face Sets", &face_sets));
    PetscCall(DMGetLabel(dm, "markers", &markers));
    PetscCheck(face_sets && markers, comm, PETSC_ERR_ARG_WRONG, "Missing \"Face Sets\" or \"markers\" label");
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "boundary", &boundary));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, boundary));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &face_start, &face_end));
    for (PetscInt f = face_start; f < face_end; ++f) {
        PetscInt face_set, marker, on_boundary;
        PetscCall(DMLabelGetValue(face_sets, f, &face_set));
        PetscCall(DMLabelGetValue(markers, f, &marker));
        PetscCall(DMLabelGetValue(boundary, f, &on_boundary));
        if ((face_set >= 1) != (on_boundary == 1)) local_errors++;
        if (face_set != marker) local_errors++;
        if (face_set >= 1 && face_set <= num_walls) local_counts[face_set - 1]++;
        else if (face_set >= 1) local_errors++;
    }
    PetscCall(DMLabelDestroy(&boundary));
    PetscCallMPI(MPI_Allreduce(local_counts, counts, (PetscMPIInt)num_walls, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&local_errors, num_errors, 1, MPIU_INT, MPI_SUM, comm));
    PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

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
        PetscCall(CheckDMPlex(dm));
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
        PetscCall(CheckDMPlex(dm));
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
        PetscCall(CheckDMPlex(dm));
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
    PetscCall(CheckDMPlex(dm_square));
    PetscCall(CheckDMPlex(dm_other));
    PetscCall(DMRefine(dm_square, PETSC_COMM_WORLD, &dm_refined));
    PetscCall(CheckDMPlex(dm_refined));

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
    // PetscMallocGetCurrentUsage only sees PETSc allocations, not TetGen's or the STL's (valgrind was clean at review).
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

#if defined(PETSC_HAVE_TETGEN)
    // 3D meshes, only when PETSc has TetGen. The tests_lib target runs this on 1 and 2 ranks, so
    // every size keeps the halo, target_edge_length * (11 + final_smooth_its), within half a tile
    // on 2 ranks too: there each box is split along its long x axis, the 1 x 0.25 x 0.25 box into
    // two tiles 0.5 long (halo 0.242), the 2 x 0.3 x 0.3 box into two tiles 1 long (halo 0.33)
    // and the small boxes into two tiles 0.5 long. The boxes are thin because the DMPlex checks
    // cost about a quarter of a millisecond per tet in a debug build, and there is no final
    // smoothing, as TetGen is compiled at -O0 in debug builds
    double thin_edge_length = 0.022;
    double thin_width = 1.0, thin_height = 0.25, thin_depth = 0.25;
    double box_edge_length = 0.03;
    double box_width = 2.0, box_height = 0.3, box_depth = 0.3;
    int final_smooth_its_3d = 0;

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Thin Box Mesh (1.0 x 0.25 x 0.25) ===\n");

    dm = GenerateBoxMeshDM3D(PETSC_COMM_WORLD, thin_edge_length, thin_width, thin_height, thin_depth, final_smooth_its_3d, integrity_check, print_stats);

    if (dm) {
        PetscCall(CheckDMPlex(dm));
        PetscPrintf(PETSC_COMM_WORLD, "Thin box mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Thin box mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Box Mesh (2.0 x 0.3 x 0.3) ===\n");

    dm = GenerateBoxMeshDM3D(PETSC_COMM_WORLD, box_edge_length, box_width, box_height, box_depth, final_smooth_its_3d, integrity_check, print_stats);

    if (dm) {
        PetscCall(CheckDMPlex(dm));
        PetscPrintf(PETSC_COMM_WORLD, "Box mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Box mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing 3D Boundary Labels After Refinement ===\n");

    // Refine a 1 x 0.12 x 0.12 box after a 0.12 x 0.12 x 1 box has been generated. The refinement
    // hook must label against the first box's own domain, or its x=1 face (Right) would be
    // missed. Thin boxes (split along their long axis on 2 ranks) keep this quick, as refining
    // a cube takes many minutes in a debug build; for the same reason the DMPlex checks are
    // run on the two boxes but not on the refined mesh. The boundary triangulation is
    // unstructured, so the face counts can't be hard coded: every wall (1=Bottom, 2=Top,
    // 3=Front, 4=Back, 5=Right, 6=Left) must have labelled faces, only boundary faces may be
    // labelled, and refinement splits every boundary triangle into exactly 4
    DM dm_long, dm_tall, dm_long_refined;
    PetscInt long_counts[6], refined_counts[6], label_errors, refined_label_errors;
    dm_long = GenerateBoxMeshDM3D(PETSC_COMM_WORLD, 0.02, 1.0, 0.12, 0.12, 0, PETSC_FALSE, PETSC_FALSE);
    dm_tall = GenerateBoxMeshDM3D(PETSC_COMM_WORLD, 0.02, 0.12, 0.12, 1.0, 0, PETSC_FALSE, PETSC_FALSE);
    PetscCall(CheckDMPlex(dm_long));
    PetscCall(CheckDMPlex(dm_tall));
    PetscCall(CountFaceSets(dm_long, 6, long_counts, &label_errors));
    PetscCall(DMRefine(dm_long, PETSC_COMM_WORLD, &dm_long_refined));
    PetscCall(CountFaceSets(dm_long_refined, 6, refined_counts, &refined_label_errors));
    PetscCall(DMDestroy(&dm_long_refined));
    PetscCall(DMDestroy(&dm_tall));
    PetscCall(DMDestroy(&dm_long));

    if (label_errors != 0 || refined_label_errors != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT " inconsistent face labels before and %" PetscInt_FMT " after refinement!\n", label_errors, refined_label_errors);
        PetscCall(PetscFinalize());
        return 1;
    }
    for (int i = 0; i < 6; ++i) {
        if (long_counts[i] == 0 || refined_counts[i] != 4 * long_counts[i]) {
            PetscPrintf(PETSC_COMM_WORLD, "Boundary %d has %" PetscInt_FMT " labelled faces before and %" PetscInt_FMT " after refinement, expected a non-zero count and 4 times as many after!\n", i + 1, long_counts[i], refined_counts[i]);
            PetscCall(PetscFinalize());
            return 1;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, "Refined 3D boundary labels correct.\n");

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing Agglomerated Box Mesh (1.0 x 0.25 x 0.25) ===\n");

    // All the ranks in a single coarse group, which also checks the 3D agglomeration entry point
    // has C linkage. On a small box to keep these tests quick in a debug build (the executable
    // tests agglomeration on a cube)
    dm = GenerateBoxMeshDM3DAgglom(PETSC_COMM_WORLD, 0.02, 1.0, 0.25, 0.25, final_smooth_its_3d, integrity_check, print_stats, (int)comm_size);

    if (dm) {
        PetscCall(CheckDMPlex(dm));
        PetscPrintf(PETSC_COMM_WORLD, "Agglomerated box mesh generation successful!\n");
        ierr = DMDestroy(&dm);
        (void)ierr;
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Agglomerated box mesh generation failed!\n");
        PetscCall(PetscFinalize());
        return 1;
    }

    PetscPrintf(PETSC_COMM_WORLD, "\n=== Testing for Memory Leaks in 3D ===\n");

    // As the 2D leak check, on the thin 1 x 0.12 x 0.12 box
    for (int i = 0; i < 2; ++i) {
        PetscCall(PetscMallocGetCurrentUsage(&mem_before));
        dm = GenerateBoxMeshDM3D(PETSC_COMM_WORLD, 0.02, 1.0, 0.12, 0.12, 0, PETSC_FALSE, PETSC_FALSE);
        PetscCall(DMDestroy(&dm));
        PetscCall(PetscMallocGetCurrentUsage(&mem_after));
    }
    leaked = mem_after - mem_before;
    PetscCallMPI(MPI_Allreduce(&leaked, &max_leaked, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
    if (max_leaked != 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Leaked %g bytes generating and destroying a 3D mesh!\n", max_leaked);
        PetscCall(PetscFinalize());
        return 1;
    }
    PetscPrintf(PETSC_COMM_WORLD, "No memory leaked in 3D.\n");
#else
    PetscPrintf(PETSC_COMM_WORLD, "\nPETSc was built without TetGen, skipping the 3D tests.\n");
#endif

    PetscCall(PetscFinalize());
    return 0;
}
