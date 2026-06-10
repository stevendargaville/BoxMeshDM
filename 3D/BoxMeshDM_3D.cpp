#include "BoxMeshDM_3D.h"
#include "tetgen.h"
#include <petscdmplex.h>

// Generates a stuctured 3D grid of points
std::vector<Point3D> GenerateStructuredGrid(int nx, int ny, int nz,
                                            double spacing) {
  std::vector<Point3D> points;
  points.reserve(nx * ny * nz);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        points.emplace_back(i * spacing, j * spacing, k * spacing);
      }
    }
  }
  return points;
}

// Tetrahedralises a point cloud using TetGen
void TetrahedralizePointCloud(const std::vector<Point3D> &point_cloud,
                              std::vector<Tetrahedron> &out_tetrahedra) {
  tetgenio in, out;

  in.numberofpoints = static_cast<int>(point_cloud.size());
  in.pointlist = new double[in.numberofpoints * 3];

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    in.pointlist[i * 3 + 0] = point_cloud[i].x;
    in.pointlist[i * 3 + 1] = point_cloud[i].y;
    in.pointlist[i * 3 + 2] = point_cloud[i].z;
  }

  // C++ interface
  tetgenbehavior behavior;
  char switches[] = "Q";
  behavior.parse_commandline(switches);
  tetrahedralize(&behavior, &in, &out, nullptr, nullptr);

  out_tetrahedra.clear();
  out_tetrahedra.reserve(out.numberoftetrahedra);

  for (int i = 0; i < out.numberoftetrahedra; ++i) {
    Tetrahedron tet;
    tet.v0 = out.tetrahedronlist[i * 4 + 0];
    tet.v1 = out.tetrahedronlist[i * 4 + 1];
    tet.v2 = out.tetrahedronlist[i * 4 + 2];
    tet.v3 = out.tetrahedronlist[i * 4 + 3];
    out_tetrahedra.push_back(tet);
  }
}

// Writes mesh to VTU file
void WriteTetrahedralMeshVTU(const std::vector<Point3D> &points,
                             const std::vector<Tetrahedron> &tets,
                             const std::string &filename, MPI_Comm comm) {
  DM dm;
  PetscInt dim = 3, numCells = tets.size(), numVertices = points.size();
  std::vector<PetscInt> cells(numCells * 4);
  std::vector<PetscReal> coords(numVertices * 3);

  for (size_t i = 0; i < tets.size(); ++i) {
    cells[i * 4 + 0] = tets[i].v0;
    cells[i * 4 + 1] = tets[i].v1;
    cells[i * 4 + 2] = tets[i].v2;
    cells[i * 4 + 3] = tets[i].v3;
  }
  for (size_t i = 0; i < points.size(); ++i) {
    coords[i * 3 + 0] = points[i].x;
    coords[i * 3 + 1] = points[i].y;
    coords[i * 3 + 2] = points[i].z;
  }

  DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, 4, PETSC_TRUE,
                                cells.data(), dim, coords.data(), &dm);

  PetscViewer viewer;
  PetscViewerVTKOpen(comm, filename.c_str(), FILE_MODE_WRITE, &viewer);
  DMView(dm, viewer);
  PetscViewerDestroy(&viewer);
  DMDestroy(&dm);
}
