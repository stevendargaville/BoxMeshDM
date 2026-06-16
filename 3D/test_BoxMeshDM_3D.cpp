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
void TestSerialHaloMesh3D() {
  std::cout << "Running TestSerialHaloMesh3D...\n";

  // Split the domain into a 2x2x2 tile layout layout
  int dim_x = 2;
  int dim_y = 2;
  int dim_z = 2;

  std::vector<Point3D> points = GenerateMesh3D_Serial(dim_x, dim_y, dim_z);

  std::cout << "-> Generated " << points.size()
            << " unique points across all tiles (halos removed).\n";
  TEST_ASSERT(!points.empty(), "Point cloud should not be empty.");

  std::vector<Tetrahedron> tets;
  TetrahedralizePointCloud(points, tets);

  std::cout << "-> Tetrahedralized into " << tets.size() << " elements.\n";
  TEST_ASSERT(!tets.empty(), "TetGen failed to produce any tetrahedra.");

  WriteTetrahedralMeshVTU(points, tets, "unstructured_halo_3d.vtu",
                          MPI_COMM_SELF);

  std::cout << "TestSerialHaloMesh3D Passed!\n\n";
}

// Test 4: Verify the automated pipeline keeps points inside the domain
void TestSmoothingQuality3D() {
  std::cout << "Running TestSmoothingQuality3D...\n";

  // Use a coarse grid for quick testing
  std::vector<Point3D> points = GenerateMesh3D_Serial(2, 2, 2);
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

int main(int argc, char **argv) {
  // Initialize PETSc (and MPI)
  PetscInitialize(&argc, &argv, NULL, NULL);

  std::cout << "=======================================\n";
  std::cout << "Starting TetGen Unstructured Mesh Tests\n";
  std::cout << "=======================================\n\n";

  TestMinimalCube();
  TestLargeStructuredGrid();
  TestSerialHaloMesh3D();
  TestSmoothingQuality3D();       // Wired this up!
  TestExplicitSmoothingVisuals(); // Added direct coverage for the algorithms

  std::cout << "All tests passed successfully!\n";

  PetscFinalize();
  return 0;
}
