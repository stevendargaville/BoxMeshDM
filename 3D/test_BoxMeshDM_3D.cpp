#include "BoxMeshDM_3D.h"
#include <cassert>
#include <cmath>
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

int main(int argc, char **argv) {
  // Initialize PETSc (and MPI)
  PetscInitialize(&argc, &argv, NULL, NULL);

  std::cout << "=======================================\n";
  std::cout << "Starting TetGen Unstructured Mesh Tests\n";
  std::cout << "=======================================\n\n";

  TestMinimalCube();
  TestLargeStructuredGrid();

  std::cout << "All tests passed successfully!\n";

  PetscFinalize();
  return 0;
}
