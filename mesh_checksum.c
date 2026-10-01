// =========================================================
// mesh_checksum - print per-rank hashes of a generated mesh
//
// Generates a mesh with GenerateBoxMeshDMAgglom or GenerateBoxMeshDM3DAgglom (and, by default, refines it once) and prints
// FNV-1a hashes of the local topology, coordinates, cell-to-global-vertex connectivity, point
// SF and boundary labels on every rank. Running the same options against two builds of the
// library and diffing the output shows whether they generate bit-identical meshes.
//
// Options: -dim, -target_edge_length, -domain_width, -domain_height, -domain_depth,
//          -final_smooth_its, -agglomeration_factor (defaults as in BoxMeshDM_main.cpp),
//          -refine (default true)
// =========================================================
#include <petscdmplex.h>
#include <petscsf.h>
#include <stdint.h>
#include "BoxMeshDM.h"

#define FNV_OFFSET 14695981039346656037ULL
#define FNV_PRIME  1099511628211ULL

static void fnv_bytes(uint64_t *h, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0; i < n; ++i) {
        *h ^= (uint64_t)p[i];
        *h *= FNV_PRIME;
    }
}

// Integers are hashed as 64 bit so the hashes don't depend on the PetscInt size
static void fnv_int(uint64_t *h, PetscInt v)
{
    int64_t w = (int64_t)v;
    fnv_bytes(h, &w, sizeof(w));
}

static PetscErrorCode HashVec(Vec v, uint64_t *h)
{
    const PetscScalar *a;
    PetscInt n;

    PetscFunctionBeginUser;
    fnv_int(h, -12345);
    if (!v) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(VecGetLocalSize(v, &n));
    fnv_int(h, n);
    PetscCall(VecGetArrayRead(v, &a));
    fnv_bytes(h, a, (size_t)n * sizeof(PetscScalar));
    PetscCall(VecRestoreArrayRead(v, &a));
    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HashMesh(DM dm, const char *name)
{
    MPI_Comm comm;
    PetscMPIInt rank, size;
    PetscInt dim, d, pStart, pEnd, p, cStart, cEnd, vStart, vEnd, c, i;
    PetscInt nroots, nleaves, ncells, nverts_owned = 0;
    const PetscInt *ilocal;
    const PetscSFNode *iremote;
    PetscSF sf;
    PetscBool *leaf_mask;
    uint64_t hashes[5], combined, *all_hashes = NULL;
    uint64_t *h_stratum = &hashes[0], *h_coords = &hashes[1], *h_cells = &hashes[2], *h_sf = &hashes[3], *h_labels = &hashes[4];
    PetscInt local_owned[4] = {0, 0, 0, 0}, global_owned[4] = {0, 0, 0, 0};
    Vec coords;
    IS vnum_is;
    const PetscInt *vnum;
    const char *label_names[2] = {"Face Sets", "markers"};

    PetscFunctionBeginUser;
    PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCheck(dim <= 3, comm, PETSC_ERR_SUP, "Dimension %" PetscInt_FMT " not supported", dim);
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    for (i = 0; i < 5; ++i) hashes[i] = FNV_OFFSET;

    // Mark which points are leaves of the point SF (ghosts)
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));
    if (nleaves < 0) nleaves = 0;
    PetscCall(PetscCalloc1(pEnd - pStart + 1, &leaf_mask));
    for (i = 0; i < nleaves; ++i) {
        PetscInt lp = ilocal ? ilocal[i] : i;
        PetscCheck(lp >= pStart && lp < pEnd, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SF leaf %" PetscInt_FMT " outside chart", lp);
        leaf_mask[lp - pStart] = PETSC_TRUE;
    }

    // (1) Stratum sizes, local SF leaves per stratum and global (owned) sizes
    for (d = 0; d <= dim; ++d) {
        PetscInt s, e, nleaf_d = 0;
        PetscCall(DMPlexGetDepthStratum(dm, d, &s, &e));
        for (p = s; p < e; ++p)
            if (leaf_mask[p - pStart]) nleaf_d++;
        local_owned[d] = (e - s) - nleaf_d;
        fnv_int(h_stratum, d);
        fnv_int(h_stratum, e - s);
        fnv_int(h_stratum, nleaf_d);
    }
    PetscCallMPI(MPI_Allreduce(local_owned, global_owned, 4, MPIU_INT, MPI_SUM, comm));
    for (d = 0; d <= dim; ++d) fnv_int(h_stratum, global_owned[d]);

    // (2) Local then global coordinates, raw bytes
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    PetscCall(HashVec(coords, h_coords));
    PetscCall(DMGetCoordinates(dm, &coords));
    PetscCall(HashVec(coords, h_coords));

    // (3) Each cell's closure vertices as raw global vertex numbers, in closure order
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(DMPlexGetVertexNumbering(dm, &vnum_is));
    PetscCall(ISGetIndices(vnum_is, &vnum));
    ncells = cEnd - cStart;
    fnv_int(h_cells, ncells);
    for (c = cStart; c < cEnd; ++c) {
        PetscInt *closure = NULL, csize, k, nv = 0;
        PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &csize, &closure));
        for (k = 0; k < csize; ++k) {
            PetscInt q = closure[2 * k];
            if (q >= vStart && q < vEnd) {
                fnv_int(h_cells, vnum[q - vStart]);
                nv++;
            }
        }
        fnv_int(h_cells, nv);
        PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &csize, &closure));
    }
    PetscCall(ISRestoreIndices(vnum_is, &vnum));
    nverts_owned = local_owned[0];

    // (4) Point SF graph
    fnv_int(h_sf, nroots);
    fnv_int(h_sf, nleaves);
    for (i = 0; i < nleaves; ++i) {
        fnv_int(h_sf, ilocal ? ilocal[i] : i);
        fnv_int(h_sf, (PetscInt)iremote[i].rank);
        fnv_int(h_sf, (PetscInt)iremote[i].index);
    }
    PetscCall(PetscFree(leaf_mask));

    // (5) Boundary label values on every point in the chart (-1 if unset or the label is missing)
    for (i = 0; i < 2; ++i) {
        DMLabel label;
        PetscCall(DMGetLabel(dm, label_names[i], &label));
        fnv_int(h_labels, label ? 1 : 0);
        for (p = pStart; p < pEnd; ++p) {
            PetscInt val = -1;
            if (label) PetscCall(DMLabelGetValue(label, p, &val));
            fnv_int(h_labels, val);
        }
    }

    // Combined hash over every rank's hashes in rank order
    if (rank == 0) PetscCall(PetscMalloc1(5 * (size_t)size, &all_hashes));
    PetscCallMPI(MPI_Gather(hashes, 5, MPI_UINT64_T, all_hashes, 5, MPI_UINT64_T, 0, comm));
    combined = FNV_OFFSET;
    if (rank == 0) fnv_bytes(&combined, all_hashes, 5 * (size_t)size * sizeof(uint64_t));
    PetscCall(PetscFree(all_hashes));

    PetscCall(PetscPrintf(comm, "=== %s ===\n", name));
    PetscCall(PetscPrintf(comm, "global: nverts=%" PetscInt_FMT " ncells=%" PetscInt_FMT, global_owned[0], global_owned[dim]));
    for (d = 1; d < dim; ++d) PetscCall(PetscPrintf(comm, " depth%" PetscInt_FMT "=%" PetscInt_FMT, d, global_owned[d]));
    PetscCall(PetscPrintf(comm, "\n"));
    PetscCall(PetscSynchronizedPrintf(comm, "rank %d: hash_stratum=%016llx hash_coords=%016llx hash_cells=%016llx hash_sf=%016llx hash_labels=%016llx ncells=%" PetscInt_FMT " nverts_owned=%" PetscInt_FMT "\n",
                                      (int)rank, (unsigned long long)*h_stratum, (unsigned long long)*h_coords, (unsigned long long)*h_cells, (unsigned long long)*h_sf, (unsigned long long)*h_labels, ncells, nverts_owned));
    PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));
    PetscCall(PetscPrintf(comm, "combined: %016llx\n", (unsigned long long)combined));
    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
    PetscReal target_len = 0.0025, domain_width = 1.0, domain_height = 1.0, domain_depth = 1.0;
    PetscInt final_smooth_its = 4, agglomeration_factor = 1, dim = 2;
    PetscBool refine = PETSC_TRUE;
    DM dm, dm_refined;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-target_edge_length", &target_len, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_width", &domain_width, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_height", &domain_height, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-final_smooth_its", &final_smooth_its, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-agglomeration_factor", &agglomeration_factor, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-refine", &refine, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-domain_depth", &domain_depth, NULL));
    PetscCheck(dim == 2 || dim == 3, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-dim %" PetscInt_FMT " must be 2 or 3", dim);

    if (dim == 2) dm = GenerateBoxMeshDMAgglom(PETSC_COMM_WORLD, target_len, domain_width, domain_height, (int)final_smooth_its, PETSC_FALSE, PETSC_FALSE, (int)agglomeration_factor);
    else dm = GenerateBoxMeshDM3DAgglom(PETSC_COMM_WORLD, target_len, domain_width, domain_height, domain_depth, (int)final_smooth_its, PETSC_FALSE, PETSC_FALSE, (int)agglomeration_factor);
    PetscCheck(dm, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Mesh generation failed");

    PetscCall(HashMesh(dm, "mesh"));
    if (refine) {
        PetscCall(DMRefine(dm, PETSC_COMM_WORLD, &dm_refined));
        PetscCall(HashMesh(dm_refined, "refined mesh"));
        PetscCall(DMDestroy(&dm_refined));
    }
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    return 0;
}
