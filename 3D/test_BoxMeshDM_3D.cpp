#include "BoxMeshDM_3D.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <petsc.h>
#include <vector>

// Simple assertion macro for visual feedback
#define TEST_ASSERT(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "TEST FAILED: " << message << " (Line " << __LINE__         \
                << ")\n";                                                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void ValidateMeshQuality(const std::vector<Point3D> &points,
                         const std::vector<Tetrahedron> &tets) {
  // Run the existing statistics
  ComputeAndPrintStats_3D(0, points, tets);

  // Ensure we don't have zero-volume or inverted elements
  for (const auto &t : tets) {
    double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2],
                                      points[t.v3]);
    TEST_ASSERT(vol > 1e-12, "Degenerate tetrahedron detected!");
  }
}

// Test 1: Minimal cube (2x2x2 points)
void TestMinimalCube() {
  std::cout << "Running TestMinimalCube...\n";

  std::vector<Point3D> points = GenerateStructuredGrid(2, 2, 2, 1.0);
  TEST_ASSERT(points.size() == 8, "Cube must have exactly 8 vertices.");

  std::vector<Tetrahedron> tets;
  TetrahedralizePointCloud(points, tets);

  TEST_ASSERT(tets.size() >= 5 && tets.size() <= 6,
              "A 2x2x2 cube should produce 5 or 6 tetrahedra.");

  for (const auto &tet : tets) {
    TEST_ASSERT(tet.v0 >= 0 && tet.v0 < 8, "Vertex index out of bounds.");
    TEST_ASSERT(tet.v1 >= 0 && tet.v1 < 8, "Vertex index out of bounds.");
    TEST_ASSERT(tet.v2 >= 0 && tet.v2 < 8, "Vertex index out of bounds.");
    TEST_ASSERT(tet.v3 >= 0 && tet.v3 < 8, "Vertex index out of bounds.");
  }

  // Write VTU file using PETSc's writer (serial communicator)
  WriteTetrahedralMeshVTU(points, tets, "cube_minimal.vtu", MPI_COMM_SELF);

  std::cout << "TestMinimalCube Passed! Generated " << tets.size()
            << " elements.\n\n";
}

// Test 2: Larger structured grid (6x6x6 points)
void TestLargeStructuredGrid() {
  std::cout << "Running TestLargeStructuredGrid...\n";

  int n = 6;
  std::vector<Point3D> points = GenerateStructuredGrid(n, n, n, 0.2);
  TEST_ASSERT(points.size() == 216, "Grid sizing logic failed.");

  std::vector<Tetrahedron> tets;
  TetrahedralizePointCloud(points, tets);

  TEST_ASSERT(tets.size() >= 625,
              "Sub-optimal element generation volume found.");

  WriteTetrahedralMeshVTU(points, tets, "grid_6x6x6.vtu", MPI_COMM_SELF);

  std::cout << "TestLargeStructuredGrid Passed! Generated " << tets.size()
            << " elements.\n\n";
}

// Test 3: New 3D Jittered Unstructured Mesh via Serial Tile Halos
void TestHaloMesh3D() {
  std::cout << "Running TestHaloMesh3D...\n";

  // Split the domain into a 2x2x2 tile layout layout
  int dim_x = 2;
  int dim_y = 2;
  int dim_z = 2;

  std::vector<Point3D> points = GenerateMesh3D(dim_x, dim_y, dim_z);

  std::cout << "-> Generated " << points.size()
            << " unique points across all tiles (halos removed).\n";
  TEST_ASSERT(!points.empty(), "Point cloud should not be empty.");

  std::vector<Tetrahedron> tets;
  TetrahedralizePointCloud(points, tets);

  ValidateMeshQuality(points, tets);

  WriteTetrahedralMeshVTU(points, tets, "unstructured_halo_3d.vtu",
                          MPI_COMM_SELF);

  std::cout << "TestHaloMesh3D Passed!\n\n";
}

// Test 4: Verify the automated pipeline keeps points inside the domain
void TestSmoothingQuality3D() {
  std::cout << "Running TestSmoothingQuality3D...\n";

  // Use a coarse grid for quick testing
  std::vector<Point3D> points = GenerateMesh3D(2, 2, 2);
  std::vector<Tetrahedron> tets;

  // Tetrahedralize after the automatic internal smoothing
  TetrahedralizePointCloud(points, tets);

  TEST_ASSERT(tets.size() > 0, "Smoothing produced no tetrahedra.");

  // Ensure points haven't drifted outside the domain on ANY axis
  for (const auto &p : points) {
    TEST_ASSERT(p.x >= -1e-9 && p.x <= DOMAIN_WIDTH + 1e-9,
                "Point drifted outside X bounds");
    TEST_ASSERT(p.y >= -1e-9 && p.y <= DOMAIN_HEIGHT + 1e-9,
                "Point drifted outside Y bounds");
    TEST_ASSERT(p.z >= -1e-9 && p.z <= DOMAIN_DEPTH + 1e-9,
                "Point drifted outside Z bounds");
  }

  WriteTetrahedralMeshVTU(points, tets, "smoothed_mesh_3d.vtu", MPI_COMM_SELF);
  std::cout << "TestSmoothingQuality3D Passed!\n\n";
}

// Test 5: Explicitly isolate and visually dump the smoothing progression
void TestExplicitSmoothingVisuals() {
  std::cout << "Running TestExplicitSmoothingVisuals...\n";

  // Create a structured grid (4x4x4)
  std::vector<Point3D> points = GenerateStructuredGrid(4, 4, 4, 0.3333);
  std::vector<Tetrahedron> tets;

  // Manually distort interior nodes to simulate a highly irregular starting
  // mesh
  std::srand(42); // Deterministic seed for repeatable test
  for (auto &p : points) {
    bool is_interior =
        (p.x > 0.05 && p.x < DOMAIN_WIDTH - 0.05 && p.y > 0.05 &&
         p.y < DOMAIN_HEIGHT - 0.05 && p.z > 0.05 && p.z < DOMAIN_DEPTH - 0.05);
    if (is_interior) {
      p.x += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
      p.y += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
      p.z += ((std::rand() % 100) / 100.0 - 0.5) * 0.15;
    }
  }

  // Output the distorted baseline
  TetrahedralizePointCloud(points, tets);
  WriteTetrahedralMeshVTU(points, tets, "visual_01_noisy.vtu", MPI_COMM_SELF);
  std::cout << "-> Dumped visual_01_noisy.vtu\n";

  // Apply Lloyd Smoothing and output
  relax_points_lloyd_3D(points, tets, 0.5);
  TetrahedralizePointCloud(points, tets);
  WriteTetrahedralMeshVTU(points, tets, "visual_02_lloyd.vtu", MPI_COMM_SELF);
  std::cout << "-> Dumped visual_02_lloyd.vtu\n";

  // Apply Spring Smoothing and output
  relax_points_spring_3D(points, tets, 0.2);
  TetrahedralizePointCloud(points, tets);
  WriteTetrahedralMeshVTU(points, tets, "visual_03_spring.vtu", MPI_COMM_SELF);
  std::cout << "-> Dumped visual_03_spring.vtu\n";

  std::cout << "TestExplicitSmoothingVisuals Passed! Load the 'visual_0*.vtu' "
               "files into ParaView to see the smoothing happen.\n\n";
}

// Test 6: Test whether the quality of the mesh has improved*
void TestSmoothingConvergence() {
  std::cout << "Running TestSmoothingConvergence...\n";

  // Match TARGET_EDGE_LENGTH to the grid spacing
  const double spacing = 1.0 / 3.0;
  const double original_target = TARGET_EDGE_LENGTH;
  TARGET_EDGE_LENGTH = spacing;

  std::vector<Point3D> points = GenerateStructuredGrid(4, 4, 4, spacing);
  std::vector<Tetrahedron> tets;

  // Manually distort interior nodes to create bad element quality initially
  std::srand(101);
  for (auto &p : points) {
    if (p.x > 0.1 && p.x < 0.9 && p.y > 0.1 && p.y < 0.9 && p.z > 0.1 &&
        p.z < 0.9) {
      p.x += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
      p.y += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
      p.z += ((std::rand() % 100) / 100.0 - 0.5) * 0.25;
    }
  }

  TetrahedralizePointCloud(points, tets);

  // Capture initial worst-case volume
  double initial_min_vol = 1e30;
  for (const auto &t : tets) {
    double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2],
                                      points[t.v3]);
    initial_min_vol = std::min(initial_min_vol, vol);
  }

  std::cout << "-> Initial State Stats:\n";
  ComputeAndPrintStats_3D(0, points, tets);

  // Run iterations of smoothing
  for (int i = 0; i < 5; ++i) {
    relax_points_lloyd_3D(points, tets, 0.5);
    relax_points_spring_3D(points, tets, 0.2);
    TetrahedralizePointCloud(points, tets);
  }

  // Capture final worst-case volume
  double final_min_vol = 1e30;
  for (const auto &t : tets) {
    double vol = calculate_tet_volume(points[t.v0], points[t.v1], points[t.v2],
                                      points[t.v3]);
    final_min_vol = std::min(final_min_vol, vol);
  }

  std::cout << "-> Final State Stats:\n";
  ComputeAndPrintStats_3D(5, points, tets);

  // Explicit programmatic check to ensure the algorithm *did its job*
  TEST_ASSERT(final_min_vol > initial_min_vol,
              "Smoothing failed to improve the worst-case element volume.");

  std::cout << "TestSmoothingConvergence Passed!\n\n";

  // Restores the global value
  TARGET_EDGE_LENGTH = original_target;
}

// Test 7: Generates a cuboid with differing side lengths
void TestNonUnitDomain() {
  std::cout << "Running TestNonUnitDomain...\n";

  // Set non-unit globals
  DOMAIN_WIDTH = 2.0;
  DOMAIN_HEIGHT = 1.0;
  DOMAIN_DEPTH = 0.5;

  std::vector<Point3D> points = GenerateMesh3D(4, 2, 1);
  std::vector<Tetrahedron> tets;
  TetrahedralizePointCloud(points, tets);

  for (const auto &p : points) {
    TEST_ASSERT(p.x >= -1e-9 && p.x <= DOMAIN_WIDTH + 1e-9, "X drift");
    TEST_ASSERT(p.y >= -1e-9 && p.y <= DOMAIN_HEIGHT + 1e-9, "Y drift");
    TEST_ASSERT(p.z >= -1e-9 && p.z <= DOMAIN_DEPTH + 1e-9, "Z drift");
  }

  ValidateMeshQuality(points, tets);

  // Write out VTU as requested
  WriteTetrahedralMeshVTU(points, tets, "non_unit_domain.vtu", MPI_COMM_SELF);

  // Reset globals for any tests that might theoretically follow
  DOMAIN_WIDTH = 1.0;
  DOMAIN_HEIGHT = 1.0;
  DOMAIN_DEPTH = 1.0;

  std::cout << "TestNonUnitDomain Passed! Dumped non_unit_domain.vtu\n\n";
}

// Test 8: Create a "bad" mesh to test whether the catching logic works
void TestDegenerateDetection() {
  std::cout << "Running TestDegenerateDetection...\n";

  // Create 4 perfectly coplanar points (z = 0 for all)
  std::vector<Point3D> points = {
      {0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.0, 0.1, 0.0}, {0.05, 0.05, 0.0}};
  std::vector<Tetrahedron> tets = {{0, 1, 2, 3}};

  double vol = calculate_tet_volume(points[tets[0].v0], points[tets[0].v1],
                                    points[tets[0].v2], points[tets[0].v3]);

  // Direct mathematical check
  TEST_ASSERT(vol < 1e-12, "Failed to accurately flag a coplanar/degenerate "
                           "tetrahedron as zero-volume!");

  std::cout << "TestDegenerateDetection Passed!\n\n";
}

int main(int argc, char **argv) {
  // Initialize PETSc (and MPI)
  PetscInitialize(&argc, &argv, NULL, NULL);

  std::cout << "=======================================\n";
  std::cout << "Starting TetGen Unstructured Mesh Tests\n";
  std::cout << "=======================================\n\n";

  TestMinimalCube();
  TestLargeStructuredGrid();
  TestHaloMesh3D();
  TestSmoothingQuality3D();
  TestExplicitSmoothingVisuals();
  TestSmoothingConvergence();
  TestNonUnitDomain();
  TestDegenerateDetection();
  std::cout << "All tests passed successfully!\n";

  PetscFinalize();
  return 0;
}
