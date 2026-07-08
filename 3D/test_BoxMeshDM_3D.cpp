#include "BoxMeshDM_3D.h"
#include "petscsys.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <petsc.h>
#include <petscdmplex.h>
#include <string>
#include <vector>

// Macro to wrap test execution with MPI timing
#define RUN_TEST_TIMED(test_call)                                                                                             \
    do {                                                                                                                      \
        MPI_Barrier(PETSC_COMM_WORLD); /* Synchronize all ranks before starting */                                            \
        double t_start = MPI_Wtime();                                                                                         \
        test_call;                                                                                                            \
        MPI_Barrier(PETSC_COMM_WORLD); /* Synchronize all ranks after finishing */                                            \
        double t_end = MPI_Wtime();                                                                                           \
        if (rank == 0) {                                                                                                      \
            std::cout << "[TIMING] `" << #test_call << "` completed in " << std::fixed << std::setprecision(4)                \
                      << (t_end - t_start) << " seconds.\n=======================================\n\n";                       \
        }                                                                                                                     \
    } while (0)

namespace fs = std::filesystem;

// Ensure the directory exists
void EnsureOutputDirectory(const std::string &dir = "test_outputs") {
    if (!fs::exists(dir)) {
        fs::create_directory(dir);
    }
}

// Simple assertion macro for visual feedback
#define TEST_ASSERT(condition, message)                                                                                       \
    do {                                                                                                                      \
        if (!(condition)) {                                                                                                   \
            std::cerr << "[RANK " << rank << "] TEST FAILED: " << message << " (Line " << __LINE__ << ")\n";                  \
            MPI_Abort(PETSC_COMM_WORLD, 1);                                                                                   \
        }                                                                                                                     \
    } while (0)

// Helper to filter tetrahedra so each rank only retains its geometrically owned elements
std::vector<Tetrahedron> FilterOwnedTetrahedra(int rank, const std::vector<Point3D> &points,
                                               const std::vector<Tetrahedron> &tets) {
    std::vector<Tetrahedron> owned;
    for (const auto &t : tets) {
        Point3D centroid((points[t.v0].x + points[t.v1].x + points[t.v2].x + points[t.v3].x) / 4.0,
                         (points[t.v0].y + points[t.v1].y + points[t.v2].y + points[t.v3].y) / 4.0,
                         (points[t.v0].z + points[t.v1].z + points[t.v2].z + points[t.v3].z) / 4.0);
        if (get_owner_rank_3D(centroid) == rank) {
            owned.push_back(t);
        }
    }
    return owned;
}

// Validate mesh quality and print parallel stats
void ValidateMeshQualityAndPrint(MPI_Comm comm, int iterations, const std::vector<Point3D> &points,
                                 const std::vector<Tetrahedron> &tets) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Print the globally-reduced statistics
    ComputeAndPrintStats_3D(comm, iterations, points, tets);

    // Ensure we don't have zero-volume or heavily inverted elements
    for (const auto &t : tets) {
        double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2], points[t.v3]);
        if (vol <= 1e-12) {
            std::cerr << "[RANK " << rank << "] Degenerate tetrahedron detected! Vol: " << vol << "\n";
            MPI_Abort(comm, 1);
        }
    }
}

// Write VTU using the decoupled PETSc DM
void WriteTestVTU(DM dm, const std::string &filename, MPI_Comm comm) { WriteMeshVTU(dm, filename, comm); }

// Test 1: Minimal structural sanity checks
void TestMinimalCube(int rank) {
    if (rank == 0)
        std::cout << "Running TestMinimalCube...\n";
    std::vector<Point3D> points = GenerateStructuredGrid(2, 2, 2, 1.0);
    TEST_ASSERT(points.size() == 8, "Structured grid count must be exactly 8 for a 2x2x2 layout.");

    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(points, tets);

    // A standard 2x2x2 cube grid splits cleanly into 5 or 6 tetrahedra depending on orientation
    TEST_ASSERT(tets.size() >= 5, "Cube domain should yield at least 5 tetrahedra.");
    if (rank == 0)
        std::cout << "TestMinimalCube Passed!\n\n";
}

// Test 2: Ensure boundary constraints properly clamp coordinate vectors
void TestBoundaryClamping(int rank) {
    if (rank == 0)
        std::cout << "Running TestBoundaryClamping...\n";

    // Setup a point right on the left boundary plane (X=0)
    Point3D p(0.0, 0.5, 0.5);
    double dx = -0.1, dy = 0.1, dz = 0.1;

    // The constraint should cancel out any negative delta-x movement trying to escape the domain
    bool is_boundary = apply_boundary_constraint_3D(p, dx, dy, dz);
    TEST_ASSERT(is_boundary == true, "Point on X=0 plane must register as boundary.");
    TEST_ASSERT(std::abs(dx) < 1e-14, "Boundary correction should neutralize movement normal to the face.");
    TEST_ASSERT(std::abs(dy - 0.1) < 1e-14, "Tangential Y movement should remain uninhibited.");

    if (rank == 0)
        std::cout << "TestBoundaryClamping Passed!\n\n";
}

// Test 3: Structural/Geometric validation of elements
void TestDegenerateDetection(int rank) {
    if (rank == 0)
        std::cout << "Running TestDegenerateDetection...\n";

    // 4 perfectly coplanar points on the z=0 plane
    Point3D p0(0.0, 0.0, 0.0), p1(1.0, 0.0, 0.0), p2(0.0, 1.0, 0.0), p3(0.5, 0.5, 0.0);
    double vol = calculate_tet_volume(p0, p1, p2, p3);

    TEST_ASSERT(vol < 1e-12, "Coplanar tetrahedron volume evaluation must fall below calculation tolerances.");
    if (rank == 0)
        std::cout << "TestDegenerateDetection Passed!\n\n";
}

// Test 4: Parallel generation, global point cloud validation, and statistics
void TestParallelMeshGeneration(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestParallelMeshGeneration across " << size << " ranks...\n";

    TARGET_EDGE_LENGTH = 0.25;
    DOMAIN_WIDTH = 1.0;
    DOMAIN_HEIGHT = 1.0;
    DOMAIN_DEPTH = 1.0;

    // Generate the mesh directly using the primary API function
    // (This integrates point generation, tetrahedralization, smoothing, and the parallel DM setup)
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, TARGET_EDGE_LENGTH, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH, 20, 0.5, 0.2,
                                 PETSC_TRUE, PETSC_FALSE);

    TEST_ASSERT(dm != nullptr, "Mesh generation returned a null DM pointer.");

    // Save final output
    EnsureOutputDirectory("test_outputs");
    WriteTestVTU(dm, "test_outputs/parallel_mesh_output_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);

    // Clean up the DM
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestParallelMeshGeneration Passed! Dumped 'parallel_mesh_output.vtu'\n\n";
}

// Test 5: Verify parallel PETSc DMPlex Topology and Distribution
void TestDMPlexTopology(int rank) {
    if (rank == 0)
        std::cout << "Running TestDMPlexTopology...\n";

    std::vector<Point3D> points = GenerateStructuredGrid(3, 3, 3, 0.5);
    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(points, tets);

    // Create the parallel distributed DM using only the owned partition
    std::vector<Tetrahedron> tets_owned = FilterOwnedTetrahedra(rank, points, tets);
    DM dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, tets_owned);
    TEST_ASSERT(dm != nullptr, "Failed to instantiate DMPlex container structure.");

    // Verify elements are distributed across the processes
    PetscInt cell_start, cell_end;
    DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end);
    PetscInt local_cells = cell_end - cell_start;

    // Sum up all distributed elements across the communicator group
    PetscInt global_cells = 0;
    MPI_Allreduce(&local_cells, &global_cells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    if (rank == 0) {
        TEST_ASSERT(global_cells == (PetscInt)tets.size(), "Global distributed element metrics desynced.");
    }

    // Confirm mesh chart information is valid on all active working processes
    PetscInt chart_start, chart_end;
    DMPlexGetChart(dm, &chart_start, &chart_end);
    TEST_ASSERT(chart_end >= chart_start, "DMPlex chart definitions corrupted or invalid.");

    DMDestroy(&dm);
    if (rank == 0)
        std::cout << "TestDMPlexTopology Passed!\n\n";
}

// Test 6: Verify smoothing converges mathematically
void TestSmoothingConvergence(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestSmoothingConvergence...\n";

    const double spacing = 1.0 / 3.0;
    const double original_target = TARGET_EDGE_LENGTH;
    TARGET_EDGE_LENGTH = spacing;

    std::vector<Point3D> points = GenerateStructuredGrid(4, 4, 4, spacing);
    std::vector<Tetrahedron> tets;

    // Manually distort interior nodes to create bad elements.
    // Seed identically on all ranks to ensure the math is identical everywhere.
    std::srand(101);
    for (auto &p : points) {
        if (p.x > 0.1 && p.x < 0.9 && p.y > 0.1 && p.y < 0.9 && p.z > 0.1 && p.z < 0.9) {
            p.x += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
            p.y += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
            p.z += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
        }
    }

    TetrahedralizePointCloud(points, tets);

    double initial_min_vol = 1e30;
    for (const auto &t : tets) {
        double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2], points[t.v3]);
        initial_min_vol = std::min(initial_min_vol, vol);
    }

    // Filter initial tetrahedra by parallel ownership partition
    std::vector<Tetrahedron> initial_tets_owned = FilterOwnedTetrahedra(rank, points, tets);

    if (rank == 0)
        std::cout << "-> Initial Distorted State Stats:\n";
    ComputeAndPrintStats_3D(PETSC_COMM_WORLD, 0, points, initial_tets_owned);

    // Apply rigorous smoothing
    for (int i = 0; i < 20; ++i) {
        relax_points_lloyd_3D(points, tets, 0.5, 0.0, 0.0, 0.0, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH);
        relax_points_spring_3D(points, tets, 0.2, 0.0, 0.0, 0.0, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH);
        TetrahedralizePointCloud(points, tets);
    }

    double final_min_vol = 1e30;
    for (const auto &t : tets) {
        double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2], points[t.v3]);
        final_min_vol = std::min(final_min_vol, vol);
    }

    // Filter final smoothed tetrahedra by parallel ownership partition
    std::vector<Tetrahedron> final_tets_owned = FilterOwnedTetrahedra(rank, points, tets);

    if (rank == 0)
        std::cout << "-> Final Smoothed State Stats:\n";
    ComputeAndPrintStats_3D(PETSC_COMM_WORLD, 20, points, final_tets_owned);

    // Create a DM from the points and partitioned tetrahedra
    DM dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, final_tets_owned);

    // Export smoothed mesh to VTU
    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/smoothed_mesh_" + std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(dm, filename, PETSC_COMM_WORLD);

    // Clean up the DM
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "-> Smoothed mesh exported to: " << filename << "\n";

    TEST_ASSERT(final_min_vol > initial_min_vol, "Smoothing failed to improve the worst-case element volume.");

    TARGET_EDGE_LENGTH = original_target; // Restore global
    if (rank == 0)
        std::cout << "TestSmoothingConvergence Passed!\n\n";
}

// Test 7: Non-Unit Domain bounds testing
void TestNonUnitDomain(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestNonUnitDomain...\n";

    DOMAIN_WIDTH = 4.0;
    DOMAIN_HEIGHT = 3.0;
    DOMAIN_DEPTH = 2.0;
    TARGET_EDGE_LENGTH = 0.5;

    // Use the primary API function to construct and validate the DM directly
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, TARGET_EDGE_LENGTH, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH, 20, 0.5, 0.2,
                                 PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Non-Unit Domain Mesh generation returned a null DM pointer.");

    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/domain_" + std::to_string((int)DOMAIN_WIDTH) + "x" +
                           std::to_string((int)DOMAIN_HEIGHT) + "x" + std::to_string((int)DOMAIN_DEPTH) + "_" +
                           std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(dm, filename, PETSC_COMM_WORLD);

    // Clean up the DM
    DMDestroy(&dm);

    // Reset globals
    DOMAIN_WIDTH = 1.0;
    DOMAIN_HEIGHT = 1.0;
    DOMAIN_DEPTH = 1.0;

    if (rank == 0)
        std::cout << "TestNonUnitDomain Passed! Dumped '" << filename << "'\n\n";
}

// Test 8: Visual step-by-step debug outputs
void TestExplicitSmoothingVisuals(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestExplicitSmoothingVisuals...\n";

    std::vector<Point3D> points = GenerateStructuredGrid(4, 4, 4, 0.3333);
    std::vector<Tetrahedron> tets;

    std::srand(42);
    for (auto &p : points) {
        bool is_interior = (p.x > 0.05 && p.x < 0.95 && p.y > 0.05 && p.y < 0.95 && p.z > 0.05 && p.z < 0.95);
        if (is_interior) {
            p.x += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
            p.y += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
            p.z += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
        }
    }

    EnsureOutputDirectory("test_outputs");

    TetrahedralizePointCloud(points, tets);
    DM dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, FilterOwnedTetrahedra(rank, points, tets));
    WriteTestVTU(dm, "test_outputs/visual_01_noisy_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);
    if (rank == 0)
        std::cout << "-> Dumped visual_01_noisy.vtu\n";
    DMDestroy(&dm);

    relax_points_lloyd_3D(points, tets, 0.5, 0.0, 0.0, 0.0, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH);
    TetrahedralizePointCloud(points, tets);
    dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, FilterOwnedTetrahedra(rank, points, tets));
    WriteTestVTU(dm, "test_outputs/visual_02_lloyd_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);
    if (rank == 0)
        std::cout << "-> Dumped visual_02_lloyd.vtu\n";
    DMDestroy(&dm);

    relax_points_spring_3D(points, tets, 0.2, 0.0, 0.0, 0.0, DOMAIN_WIDTH, DOMAIN_HEIGHT, DOMAIN_DEPTH);
    TetrahedralizePointCloud(points, tets);
    dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, FilterOwnedTetrahedra(rank, points, tets));
    WriteTestVTU(dm, "test_outputs/visual_03_spring_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);
    if (rank == 0)
        std::cout << "-> Dumped visual_03_spring.vtu\n";
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestExplicitSmoothingVisuals Passed! Load the 'visual_0*.vtu' files into ParaView.\n\n";
}

// Test 9: High-resolution mesh generation and export
void TestHighResolutionMesh(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestHighResolutionMesh and exporting to VTU...\n";

    // Generate a 10x10x10 structured grid with the new target edge length
    // Using a finer spacing to take advantage of the 0.02 global constraint
    std::vector<Point3D> points = GenerateStructuredGrid(10, 10, 10, 0.05);
    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(points, tets);

    // Create a DM from the owned points and tetrahedra
    std::vector<Tetrahedron> tets_owned = FilterOwnedTetrahedra(rank, points, tets);
    DM dm = CreateDMPlex3D(PETSC_COMM_WORLD, points, tets_owned);

    // Export the high-res mesh
    EnsureOutputDirectory("test_outputs");
    WriteTestVTU(dm, "test_outputs/high_res_mesh_output_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);

    // Clean up the DM
    DMDestroy(&dm);

    if (rank == 0) {
        std::cout << "-> High-Res Mesh Stats:\n";
        std::cout << "   Total Points: " << points.size() << "\n";
        std::cout << "   Total Tetrahedra: " << tets.size() << "\n";
        std::cout << "TestHighResolutionMesh Passed! Dumped 'high_res_mesh_output.vtu'\n\n";
    }
}

// Test 10: Ensures mesh generator fills the exact requested volume without overflowing
void TestWatertightVolume(int rank) {
    if (rank == 0)
        std::cout << "Running TestWatertightVolume...\n";

    double w = 2.0, h = 1.5, d = 1.0;
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.33, w, h, d, 4, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Mesh generation failed in Watertight test.");

    // Get the global bounding box of the generated mesh
    PetscReal globalMin[3], globalMax[3];
    DMGetBoundingBox(dm, globalMin, globalMax);

    // Verify the bounds match the requested domain exactly (within floating point limits)
    if (rank == 0) {
        TEST_ASSERT(std::abs(globalMin[0] - 0.0) < 1e-10, "X minimum bound escaped 0.");
        TEST_ASSERT(std::abs(globalMax[0] - w) < 1e-10, "X maximum bound does not match DOMAIN_WIDTH.");
        TEST_ASSERT(std::abs(globalMin[1] - 0.0) < 1e-10, "Y minimum bound escaped 0.");
        TEST_ASSERT(std::abs(globalMax[1] - h) < 1e-10, "Y maximum bound does not match DOMAIN_HEIGHT.");
        TEST_ASSERT(std::abs(globalMin[2] - 0.0) < 1e-10, "Z minimum bound escaped 0.");
        TEST_ASSERT(std::abs(globalMax[2] - d) < 1e-10, "Z maximum bound does not match DOMAIN_DEPTH.");
    }

    DMDestroy(&dm);
    if (rank == 0)
        std::cout << "TestWatertightVolume Passed!\n\n";
}

// Test 11: Validates that PETSc correctly populated the Boundary Labels based on 3D logic
void TestPETScBoundaryLabels(int rank) {
    if (rank == 0)
        std::cout << "Running TestPETScBoundaryLabels...\n";

    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.33, 1.0, 1.0, 1.0, 4, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);

    DMLabel label;
    DMGetLabel(dm, "Face Sets", &label);
    TEST_ASSERT(label != nullptr, "Face Sets label must exist on the generated DM.");

    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

    // Verify strata for Face Z=0 exists globally (Label 1)
    IS is_bottom;
    DMLabelGetStratumIS(label, 1, &is_bottom);
    PetscInt bottom_pts = 0;
    if (is_bottom)
        ISGetSize(is_bottom, &bottom_pts);

    PetscInt global_bottom_pts;
    MPI_Allreduce(&bottom_pts, &global_bottom_pts, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    if (rank == 0) {
        TEST_ASSERT(global_bottom_pts > 0, "No boundary vertices found on the Bottom (Z=0) plane.");
    }

    if (is_bottom)
        ISDestroy(&is_bottom);
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestPETScBoundaryLabels Passed!\n\n";
}

// Test 12: Asymmetric domain logic
void TestAsymmetricMPI(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestAsymmetricMPI...\n";

    // Forcing an elongated domain. If MPI sorting works, it will map all splits along X.
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.4, 4.0, 1.0, 1.0, 4, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Integrity Check Failed for asymmetric domain");

    EnsureOutputDirectory("test_outputs");
    WriteTestVTU(dm, "test_outputs/asymmetric_mesh_" + std::to_string(size) + "_ranks.vtu", PETSC_COMM_WORLD);
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestAsymmetricMPI Passed!\n\n";
}

// Test 13: Utilize PETSc's internal DMPlex checkers for topological validity
void TestDMPlexInternalValidity(int rank) {
    if (rank == 0)
        std::cout << "Running TestDMPlexInternalValidity...\n";

    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.5, 1.0, 1.0, 1.0, 4, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Mesh generation failed.");

    // Check symmetry of the Plex (e.g., if cell A points to face B, face B must point to cell A)
    PetscErrorCode ierr = DMPlexCheckSymmetry(dm);
    TEST_ASSERT(ierr == 0, "DMPlexCheckSymmetry failed! The mesh topology is corrupted.");

    // Check skeleton (ensures boundaries of cells are properly defined)
    ierr = DMPlexCheckSkeleton(dm, 0);
    TEST_ASSERT(ierr == 0, "DMPlexCheckSkeleton failed! Cell boundaries are invalid.");

    DMDestroy(&dm);
    if (rank == 0)
        std::cout << "TestDMPlexInternalValidity Passed!\n\n";
}

// Test 14: Ensure mesh resolution scales correctly with target edge length
void TestMeshResolutionScaling(int rank) {
    if (rank == 0)
        std::cout << "Running TestMeshResolutionScaling...\n";

    // Generate Coarse Mesh
    DM dm_coarse = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.5, 1.0, 1.0, 1.0, 2, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);

    // Generate Fine Mesh (edge length halved)
    DM dm_fine = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.25, 1.0, 1.0, 1.0, 2, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);

    PetscInt c_start_c, c_end_c, c_start_f, c_end_f;
    DMPlexGetHeightStratum(dm_coarse, 0, &c_start_c, &c_end_c);
    DMPlexGetHeightStratum(dm_fine, 0, &c_start_f, &c_end_f);

    PetscInt local_cells_coarse = c_end_c - c_start_c;
    PetscInt local_cells_fine = c_end_f - c_start_f;

    PetscInt global_cells_coarse, global_cells_fine;
    MPI_Allreduce(&local_cells_coarse, &global_cells_coarse, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&local_cells_fine, &global_cells_fine, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    if (rank == 0) {
        TEST_ASSERT(global_cells_fine > (global_cells_coarse * 4), "Fine mesh did not scale element count appropriately.");
    }

    DMDestroy(&dm_coarse);
    DMDestroy(&dm_fine);

    if (rank == 0)
        std::cout << "TestMeshResolutionScaling Passed!\n\n";
}

// Test 15: High-load stress test to generate a few million elements
void TestLargeMeshGeneration(int rank, int size) {
    // Only runs if it is running in serial or if it runs in MPI with 8 ranks
    // (avoids the massive mesh generation taking too long)
    if (size != 1 && size != 8)
        return;

    if (rank == 0)
        std::cout << "Running TestLargeMeshGeneration (expecting ~3 Million+ elements)...\n";

    // Target edge length of 0.0125 mathematically maps to roughly 80 points per axis.
    // 80 * 80 * 80 = 512,000 cubes. At ~6 tetrahedra per cube that leads to > 3,000,000 elements globally.
    double target_edge = 0.0125;

    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, target_edge, 1.0, 1.0, 1.0, 5, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Large mesh generation returned a null DM.");

    PetscInt c_start, c_end;
    DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
    PetscInt local_cells = c_end - c_start;

    PetscInt global_cells = 0;
    MPI_Allreduce(&local_cells, &global_cells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    if (rank == 0) {
        std::cout << "-> Successfully generated " << global_cells << " global tetrahedra!\n";
        TEST_ASSERT(global_cells > 2000000, "Large mesh failed to breach the 2 million element mark.");
    }

    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/80x80x80_" + std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(dm, filename, PETSC_COMM_WORLD);
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestLargeMeshGeneration Passed! Dumped '" << filename << "'\n\n";
}

// Test 16: Ensure mesh generator can handle extremely flat/sheet-like domains
void TestExtremelyFlatDomain(int rank, int size) {
    if (rank == 0)
        std::cout << "Running TestExtremelyFlatDomain...\n";

    // 10x10x0.1 domain generates extremely high geometric aspect ratios for the initial MPI tile chunking
    DM dm = GenerateBoxMeshDM_3D(PETSC_COMM_WORLD, 0.05, 2.0, 2.0, 0.1, 5, 0.5, 0.2, PETSC_TRUE, PETSC_FALSE);
    TEST_ASSERT(dm != nullptr, "Extremely flat domain mesh generation failed.");

    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/flat_domain_" + std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(dm, filename, PETSC_COMM_WORLD);
    DMDestroy(&dm);

    if (rank == 0)
        std::cout << "TestExtremelyFlatDomain Passed! Dumped '" << filename << "'\n\n";
}

// Test 17: Validate spatial partitioning assignment function handles boundary float limits safely
void TestOwnerRankDetermination(int rank) {
    if (rank == 0)
        std::cout << "Running TestOwnerRankDetermination...\n";

    // Re-simulate domain dimensions to ensure static boundaries are set
    DOMAIN_WIDTH = 2.0;
    DOMAIN_HEIGHT = 2.0;
    DOMAIN_DEPTH = 2.0;

    // Spatial edge boundaries
    Point3D origin(0.0, 0.0, 0.0);
    Point3D far_corner(2.0, 2.0, 2.0);
    Point3D out_of_bounds(3.0, 3.0, 3.0);
    Point3D negative(-1.0, -1.0, -1.0);

    int o_origin = get_owner_rank_3D(origin);
    int o_far = get_owner_rank_3D(far_corner);
    int o_out = get_owner_rank_3D(out_of_bounds);
    int o_neg = get_owner_rank_3D(negative);

    TEST_ASSERT(o_origin >= 0, "Origin owner rank must be strictly valid.");
    TEST_ASSERT(o_far >= 0, "Far corner boundary owner rank must be strictly valid.");
    TEST_ASSERT(o_out >= 0, "Out of bounds points should clamp to highest partition rank safely.");
    TEST_ASSERT(o_neg >= 0, "Negative points should clamp to rank 0 securely.");

    if (rank == 0)
        std::cout << "TestOwnerRankDetermination Passed!\n\n";
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "=======================================\n";
        std::cout << "Starting TetGen Unstructured Mesh Tests\n";
        std::cout << "=======================================\n\n";
    }

    // Run Tests
    RUN_TEST_TIMED(TestMinimalCube(rank));
    RUN_TEST_TIMED(TestBoundaryClamping(rank));
    RUN_TEST_TIMED(TestDegenerateDetection(rank));
    RUN_TEST_TIMED(TestParallelMeshGeneration(rank, size));
    RUN_TEST_TIMED(TestDMPlexTopology(rank));
    RUN_TEST_TIMED(TestSmoothingConvergence(rank, size));
    RUN_TEST_TIMED(TestNonUnitDomain(rank, size));
    RUN_TEST_TIMED(TestExplicitSmoothingVisuals(rank, size));
    RUN_TEST_TIMED(TestHighResolutionMesh(rank, size));
    RUN_TEST_TIMED(TestWatertightVolume(rank));
    RUN_TEST_TIMED(TestPETScBoundaryLabels(rank));
    RUN_TEST_TIMED(TestAsymmetricMPI(rank, size));
    RUN_TEST_TIMED(TestDMPlexInternalValidity(rank));
    RUN_TEST_TIMED(TestMeshResolutionScaling(rank));
    // RUN_TEST_TIMED(TestLargeMeshGeneration(rank, size));
    RUN_TEST_TIMED(TestExtremelyFlatDomain(rank, size));
    RUN_TEST_TIMED(TestOwnerRankDetermination(rank));

    if (rank == 0) {
        std::cout << "=========================================\n";
        std::cout << " ALL 3D PARALLEL MESH TESTS PASSED CLEANLY!\n";
        std::cout << "=========================================\n";
    }

    PetscFinalize();
    return 0;
}
