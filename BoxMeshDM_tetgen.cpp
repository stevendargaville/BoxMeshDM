// =========================================================
// TetGen backend for the 3D BoxMeshDM generator
// =========================================================
//
// The only translation unit that includes tetgen.h, see BoxMeshDM_tetgen.h. Uses TetGen 1.6
// as built by PETSc's --download-tetgen.
// =========================================================
#include "BoxMeshDM_tetgen.h"
#include <petscsys.h>
#include <iostream>
#include <cstdlib>

#if defined(PETSC_HAVE_TETGEN)

// Same preamble as PETSc's src/dm/impls/plex/generators/tetgen/tetgenerate.cxx
#if PetscDefined(HAVE_TETGEN_TETLIBRARY_NEEDED)
  #define TETLIBRARY
#endif
#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-parameter"
  #pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#elif defined(__GNUC__) || defined(__GNUG__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <tetgen.h>
#if defined(__clang__)
  #pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
  #pragma GCC diagnostic pop
#endif

// ~~~~~~~~~~~~~~~~~

int BoxMeshDM_Delaunay3D(MPI_Comm comm, int num_points, const double *xyz, std::vector<int>& tets) {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    tetgenio in, out;
    in.firstnumber = 0;
    in.numberofpoints = num_points;
    // Freed by the tetgenio destructor
    in.pointlist = new REAL[3 * (size_t)num_points];
    for (size_t i = 0; i < 3 * (size_t)num_points; ++i) in.pointlist[i] = xyz[i];

    // Q: quiet, J: keep every input point in the output so output point i is input point i
    // (without J TetGen drops unused points and renumbers), z: zero-based indexing
    tetgenbehavior behavior;
    char args[] = "QJz";
    behavior.parse_commandline(args);

    // With TETLIBRARY defined TetGen reports a failure by throwing an int. Uncaught, that would
    // kill this rank without stopping the others
    try {
        tetrahedralize(&behavior, &in, &out);
    } catch (int err) {
        std::cerr << "ERROR: [Rank " << comm_rank << "] TetGen failed with error code " << err
                  << " tetrahedralising " << num_points << " points.\n";
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (out.numberofpoints != num_points || out.numberofcorners != 4) {
        std::cerr << "ERROR: [Rank " << comm_rank << "] TetGen returned " << out.numberofpoints << " points (expected "
                  << num_points << ") and " << out.numberofcorners << " corners per tetrahedron (expected 4).\n";
        MPI_Abort(comm, EXIT_FAILURE);
    }

    int num_tets = out.numberoftetrahedra;
    tets.assign(out.tetrahedronlist, out.tetrahedronlist + 4 * (size_t)num_tets);

    // Every input point has to be a vertex of some tetrahedron. With J an exact (or
    // TetGen-tolerance) duplicate is kept in the point list but used by no tetrahedron. Two ranks
    // generating the same point must give it the same position, and a duplicate means the point
    // cloud is broken, so stop rather than hand out a dangling vertex
    std::vector<char> used(num_points, 0);
    for (size_t k = 0; k < tets.size(); ++k) {
        int v = tets[k];
        if (v < 0 || v >= num_points) {
            std::cerr << "ERROR: [Rank " << comm_rank << "] TetGen returned vertex index " << v
                      << " outside [0, " << num_points << ").\n";
            MPI_Abort(comm, EXIT_FAILURE);
        }
        used[v] = 1;
    }
    for (int i = 0; i < num_points; ++i) {
        if (!used[i]) {
            std::cerr << "ERROR: [Rank " << comm_rank << "] Point " << i << " ("
                      << xyz[3 * i] << ", " << xyz[3 * i + 1] << ", " << xyz[3 * i + 2]
                      << ") is not a vertex of any tetrahedron, it duplicates (or nearly duplicates) another point.\n";
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    // Degeneracy and orientation, with TetGen's exact orient3d predicate. Call it now, right
    // after tetrahedralize, whose exactinit has set the static error bound for this point cloud.
    // Never call exactinit ourselves (exactinit(0,...) zeroes the filter and makes it inexact).
    // TetGen's orient3d(a,b,c,d) is the determinant of [a-d; b-d; c-d], which is NEGATIVE for
    // TetGen's output tetrahedra: these are positively oriented in the right-hand sense,
    // (b-a).((c-a)x(d-a)) > 0. An exact zero is a flat tetrahedron. Neither case can be dropped
    // (that would leave a hole), so stop
    for (int t = 0; t < num_tets; ++t) {
        REAL *p[4];
        for (int c = 0; c < 4; ++c) p[c] = &out.pointlist[3 * (size_t)tets[4 * (size_t)t + c]];
        REAL o = orient3d(p[0], p[1], p[2], p[3]);
        if (o == 0.0) {
            std::cerr << "ERROR: [Rank " << comm_rank << "] TetGen returned a degenerate (flat) tetrahedron with vertices "
                      << tets[4 * t] << ", " << tets[4 * t + 1] << ", " << tets[4 * t + 2] << ", " << tets[4 * t + 3] << ".\n";
            MPI_Abort(comm, EXIT_FAILURE);
        }
        if (o > 0.0) {
            std::cerr << "ERROR: [Rank " << comm_rank << "] TetGen returned a tetrahedron with unexpected orientation "
                      << "(orient3d > 0), vertices " << tets[4 * t] << ", " << tets[4 * t + 1] << ", "
                      << tets[4 * t + 2] << ", " << tets[4 * t + 3] << ".\n";
            MPI_Abort(comm, EXIT_FAILURE);
        }
    }

    return num_tets;
}

#else

// ~~~~~~~~~~~~~~~~~

int BoxMeshDM_Delaunay3D(MPI_Comm comm, int num_points, const double *xyz, std::vector<int>& tets) {
    (void)num_points; (void)xyz;
    tets.clear();
    std::cerr << "BoxMeshDM was built without TetGen (reconfigure PETSc with --download-tetgen)\n";
    MPI_Abort(comm, EXIT_FAILURE);
    return 0;
}

#endif
