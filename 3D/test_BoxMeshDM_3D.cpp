#include "BoxMeshDM_3D.h"
#include "petscsys.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <petsc.h>
#include <petscdmplex.h>
#include <string>
#include <vector>

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
void WriteTestVTU(MPI_Comm comm, const std::vector<Point3D> &points, const std::vector<Tetrahedron> &tets,
                  const std::string &filename) {
    DM dm = CreateDMPlex3D(points, tets, comm);
    WriteTetrahedralMeshVTU(dm, filename, comm);
    DMDestroy(&dm);
}

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

    // Generate parallel point clouds (collectively synchronizes internally via MPI_Allgatherv)
    std::vector<Point3D> cloud = GenerateMesh3D(2, 2, 2, 0.5, 0.2);

    // Every rank must possess the identical global deduplicated cloud after reconstruction
    size_t local_size = cloud.size();
    std::vector<size_t> all_sizes(size);
    MPI_Allgather(&local_size, 1, MPI_UNSIGNED_LONG, all_sizes.data(), 1, MPI_UNSIGNED_LONG, PETSC_COMM_WORLD);

    for (int i = 0; i < size; ++i) {
        TEST_ASSERT(all_sizes[i] == local_size, "Mismatched global point cloud sizes gathered across ranks.");
    }

    // Ensure every node remains completely bounded inside the geometric envelope
    for (const auto &p : cloud) {
        TEST_ASSERT(p.x >= 0.0 && p.x <= DOMAIN_WIDTH, "Point escaped X boundary limits.");
        TEST_ASSERT(p.y >= 0.0 && p.y <= DOMAIN_HEIGHT, "Point escaped Y boundary limits.");
        TEST_ASSERT(p.z >= 0.0 && p.z <= DOMAIN_DEPTH, "Point escaped Z boundary limits.");
    }

    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(cloud, tets);

    // Print stats
    ValidateMeshQualityAndPrint(PETSC_COMM_WORLD, 20, cloud, tets);

    // Save final output
    EnsureOutputDirectory("test_outputs");
    WriteTestVTU(PETSC_COMM_WORLD, cloud, tets, "test_outputs/parallel_mesh_output_" + std::to_string(size) + "_ranks.vtu");

    if (rank == 0)
        std::cout << "TestParallelMeshGeneration Passed! Dumped 'parallel_mesh_output.vtu'\n\n";
}

// Test 5: Verify parallel PETSc DMPlex Topology and Distribution
void TestDMPlexTopology(int rank) {
    if (rank == 0)
        std::cout << "Running TestDMPlexTopology...\n";

    // Synthesize a localized mesh structure on rank 0
    std::vector<Point3D> points = GenerateStructuredGrid(2, 2, 2, 1.0);
    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(points, tets);

    // Create the parallel distributed DM
    DM dm = CreateDMPlex3D(points, tets, PETSC_COMM_WORLD);
    TEST_ASSERT(dm != nullptr, "Failed to instantiate DMPlex container structure.");

    // Verify elements are distributed across the processes
    PetscInt cell_start, cell_end;
    DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end);
    PetscInt local_cells = cell_end - cell_start;

    // Sum up all distributed elements across the communicator group
    PetscInt global_cells = 0;
    MPI_Allreduce(&local_cells, &global_cells, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

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

    if (rank == 0)
        std::cout << "-> Initial Distorted State Stats:\n";
    ComputeAndPrintStats_3D(PETSC_COMM_WORLD, 0, points, tets);

    // Apply rigorous smoothing
    for (int i = 0; i < 20; ++i) {
        relax_points_lloyd_3D(points, tets, 0.5);
        relax_points_spring_3D(points, tets, 0.2);
        TetrahedralizePointCloud(points, tets);
    }

    double final_min_vol = 1e30;
    for (const auto &t : tets) {
        double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2], points[t.v3]);
        final_min_vol = std::min(final_min_vol, vol);
    }

    if (rank == 0)
        std::cout << "-> Final Smoothed State Stats:\n";
    ComputeAndPrintStats_3D(PETSC_COMM_WORLD, 20, points, tets);

    // Export smoothed mesh to VTU
    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/smoothed_mesh_" + std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, filename);

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

    std::vector<Point3D> points = GenerateMesh3D(4, 3, 2, 0.5, 0.2);
    std::vector<Tetrahedron> tets;
    TetrahedralizePointCloud(points, tets);

    for (const auto &p : points) {
        TEST_ASSERT(p.x >= -1e-9 && p.x <= DOMAIN_WIDTH + 1e-9, "X drift");
        TEST_ASSERT(p.y >= -1e-9 && p.y <= DOMAIN_HEIGHT + 1e-9, "Y drift");
        TEST_ASSERT(p.z >= -1e-9 && p.z <= DOMAIN_DEPTH + 1e-9, "Z drift");
    }

    // Print stats and validate
    ValidateMeshQualityAndPrint(PETSC_COMM_WORLD, 20, points, tets);

    EnsureOutputDirectory("test_outputs");
    std::string filename = "test_outputs/domain_" + std::to_string((int)DOMAIN_WIDTH) + "x" +
                           std::to_string((int)DOMAIN_HEIGHT) + "x" + std::to_string((int)DOMAIN_DEPTH) + "_" +
                           std::to_string(size) + "_ranks.vtu";
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, filename);

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
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, "test_outputs/visual_01_noisy_" + std::to_string(size) + "_ranks.vtu");
    if (rank == 0)
        std::cout << "-> Dumped visual_01_noisy.vtu\n";

    relax_points_lloyd_3D(points, tets, 0.5);
    TetrahedralizePointCloud(points, tets);
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, "test_outputs/visual_02_lloyd_" + std::to_string(size) + "_ranks.vtu");
    if (rank == 0)
        std::cout << "-> Dumped visual_02_lloyd.vtu\n";

    relax_points_spring_3D(points, tets, 0.2);
    TetrahedralizePointCloud(points, tets);
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, "test_outputs/visual_03_spring_" + std::to_string(size) + "_ranks.vtu");
    if (rank == 0)
        std::cout << "-> Dumped visual_03_spring.vtu\n";

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

    // Export the high-res mesh
    EnsureOutputDirectory("test_outputs");
    WriteTestVTU(PETSC_COMM_WORLD, points, tets, "test_outputs/high_res_mesh_output_" + std::to_string(size) + "_ranks.vtu");

    if (rank == 0) {
        std::cout << "-> High-Res Mesh Stats:\n";
        std::cout << "   Total Points: " << points.size() << "\n";
        std::cout << "   Total Tetrahedra: " << tets.size() << "\n";
        std::cout << "TestHighResolutionMesh Passed! Dumped 'high_res_mesh_output.vtu'\n\n";
    }
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

    // Run Test Battery
    TestMinimalCube(rank);
    TestBoundaryClamping(rank);
    TestDegenerateDetection(rank);
    TestParallelMeshGeneration(rank, size);
    TestDMPlexTopology(rank);
    TestSmoothingConvergence(rank, size);
    TestNonUnitDomain(rank, size);
    TestExplicitSmoothingVisuals(rank, size);
    TestHighResolutionMesh(rank, size);

    if (rank == 0) {
        std::cout << "=========================================\n";
        std::cout << " ALL 3D PARALLEL MESH TESTS PASSED CLEANLY!\n";
        std::cout << "=========================================\n";
    }

    PetscFinalize();
    return 0;
}
