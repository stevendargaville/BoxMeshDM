# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

BoxMeshDM generates **fully unstructured meshes of a box, in parallel with MPI, and returns a
distributed PETSc `DMPlex`**: triangles on `[0,width] x [0,height]` (2D) or tetrahedra on
`[0,width] x [0,height] x [0,depth]` (3D). The design point: a *load-balanced, fully
unstructured* mesh with **no file I/O and no mesh partitioner**. Each rank generates its own
piece directly in the right place. The price is the restrictions in [README.md](README.md)
(box domain, uniform resolution, a minimum size per rank, result depends on rank count).

The 2D generator is validated at scale and is the thing to protect. The 3D generator is new and
has not been run at scale.

Files:
- [BoxMeshDM.cpp](BoxMeshDM.cpp): the whole generator, templated on the dimension.
- [BoxMeshDM_tetgen.cpp](BoxMeshDM_tetgen.cpp) / [.h](BoxMeshDM_tetgen.h): the TetGen backend,
  in its own translation unit so `tetgen.h` never meets the 2D code.
- [BoxMeshDM.h](BoxMeshDM.h): the C-linkage API, `GenerateBoxMeshDM`/`GenerateBoxMeshDMAgglom`
  and `GenerateBoxMeshDM3D`/`GenerateBoxMeshDM3DAgglom`. Deliberately no C++ overloads; the 2D
  signatures must not change.
- [BoxMeshDM_main.cpp](BoxMeshDM_main.cpp): the executable's `main`, so the library never
  contains one.
- [test_lib.c](test_lib.c): the library test. [mesh_checksum.c](mesh_checksum.c): per-rank
  hashes of a generated mesh, for bit-identity checks.

## Build and test

Requires PETSc **>= 3.24** with Triangle (`--download-triangle`). TetGen (`--download-tetgen`)
is optional: without it the 3D tests are skipped and a 3D request stops with an error.
`PETSC_DIR`/`PETSC_ARCH` must be set for *every* make target, including `clean` (without them
the Makefile's `awk` config check waits on stdin forever). The Makefile uses only PETSc's
standard rules; keep it that way rather than adding custom compile rules or per-target defines.

```bash
make clean && make          # executable ./BoxMeshDM
```

```bash
make clean && make lib      # libboxmeshdm.so / .dylib / .a
```

```bash
make clean && make tests    # the gate: executable runs (2D, then 3D if TetGen), then lib + test_lib
```

CI ([ci_build.yml](.github/workflows/ci_build.yml)) runs `make tests` in the images in
[dockerfiles/](dockerfiles) (debug, opt, 64-bit, PFLARE) plus macOS, all with TetGen. Debug CI
uses `PETSC_OPTIONS="-on_error_abort -fp_trap on"`, so new floating-point operations must not
generate NaN/Inf even transiently. 3D is slow in a debug build, so the 3D test sizes are small.

## Changing the mesh is high risk

The 2D generator has been validated at scale (see the README weak-scaling table) and that can't
easily be repeated. Any change that can reach the 2D path must leave the 2D mesh
**bit-identical**, shown with `mesh_checksum`, old build vs new build, on the same machine:

```bash
make clean && make lib && make mesh_checksum     # in a checkout of the old code, and of the new
mpiexec -n 3 ./mesh_checksum -target_edge_length 0.01 > new_n3.txt   # same options for both
```

Cover several rank counts, agglomeration factors, smoothing counts (including 0), a non-square
domain and very large and very small scales. Do it in three builds: the opt arch, the opt arch
with `make CXXFLAGS=-march=native` (so the compiler contracts FMAs, as production builds do),
and the debug arch. Identical hashes in all three are the gate.

3D changes may alter the 3D mesh, but say so explicitly. Use `mesh_checksum -dim 3` to show
when a 3D change is meant to be neutral. Changes to checks, errors and validation are much
safer than changes to geometry.

## The dimension template

Everything is written once against `Point<DIM>` and `Simplex<DIM>` and instantiated through
`GenerateBoxMeshDMImpl<2>` and `<3>`. Globals are axis-indexed (`DOMAIN_SIZE[3]`, ...).

- **Dimension logic lives only in explicit specialisations.** Never write `if (DIM == 3)` in a
  template body. A 3D retune of anything shared becomes a `<3>` specialisation, not a branch.
- **Keep the 2D floating-point expressions exactly as they are**: parenthesisation, divide vs
  multiply by inverse, literal constants, strict vs non-strict comparisons. "Equivalent"
  rewrites change the last bit and so the mesh.
- **Keep the 2D loop orders**: any loop whose order feeds a floating-point sum or the output
  numbering. 2D points are never sorted.
- **Keep the FMA guards**: `StaticFor<N>::run`, `BOXMESHDM_NOINLINE` on
  `apply_boundary_constraint`, and separate per-axis accumulator arrays. With FMA contraction
  the result depends on the compiler's unrolling and inlining, not just the expression text, so
  re-run the full checksum comparison after **any** change to generic code.

## Algorithm

`GenerateBoxMeshDMImpl<DIM>` validates inputs, sets file-scope globals, factors the ranks into a
tile grid (**exactly one tile per rank**), then runs `process_tile` → optional
`CheckMeshIntegrity` / `ComputeAndPrintStats` → `CreateDM` → optional `CheckDMIntegrity` →
`LabelBoundaries`.

`process_tile`, per rank:
- **Halo.** Points are generated for the tile plus a halo of
  `pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8)`, since smoothing distortion
  travels about one edge per iteration. It hard-errors if `pad` exceeds half the tile size
  along any axis split between ranks (that would need neighbour-of-neighbour data), which is
  why few elements per rank fails.
- **Points.** Explicit corners, then evenly spaced points on the walls (and edges and faces in
  3D), then one jittered interior point per grid cell, rejected near the walls. In 3D the
  points are sorted by `unique_hash_id` so TetGen breaks ties the same way on every rank.
- **Smoothing.** `ANNEAL_ITERS` rounds of jitter → triangulate → Lloyd → spring →
  `ResolveBoundaryOwnership`, then `final_smooth_its` rounds without jitter, then a final
  triangulation and `ResolveBoundaryOwnership`. Only points inside tile ± `sync_margin` move;
  the outer halo rim is frozen so it doesn't collapse inward.
- **Improvement (3D only).** `RepairSlivers<3>` (a no-op in 2D) keeps the final
  tetrahedralisation and improves it, because Delaunay leaves slivers in 3D that smoothing
  does not remove. It moves the vertices of the worst tets, then flips faces and edges
  (`FlipTetrahedra3D`), then smooths what remains. A change is only accepted if it strictly
  improves the worst tet involved, so nothing inverts. The result is valid but no longer
  exactly Delaunay.
- **Filtering.** A simplex belongs to the rank owning its vertex with the **smallest
  `unique_hash_id`**. Points owned spatially but in no owned simplex are kept as **orphans** so
  the rank can still hand out their global ids.

`CreateDM` numbers owned points with `MPI_Exscan` and fetches ghost ids from their owners, then
calls `DMPlexCreateFromCellListParallelPetsc` with only owned coordinates. In 3D every
tetrahedron has vertices 0 and 1 swapped, as PETSc's own TetGen interface does, because PETSc's
reference tetrahedron has the opposite orientation. `DMPlexDistributeSetDefault(dm, PETSC_FALSE)`
is deliberate: the mesh is already balanced.

`LabelBoundaries` writes `"Face Sets"` and `"markers"`, with the values in the README (3D
follows PETSc's `DMPlexCreateBoxMesh`). A refine hook relabels refined meshes using the domain
size attached to each DM (not the globals, which the next generator call overwrites).

## Determinism is the core invariant

There is no global consensus step for geometry: ranks that generate the same point must produce
*bit-identical* coordinates for it. This relies on:

- **`unique_hash_id`**, built from *global* grid indices, not coordinates. The far walls use a
  reserved index, so every other index must stay below it (checked on input).
- **Stateless RNG**: `splitmix64` seeded from the grid indices for placement and from
  `unique_hash_id ^ iteration` for jitter. No shared state, no rank dependence.
- **`ResolveBoundaryOwnership`**: after each smoothing round, neighbouring ranks exchange claims
  and all adopt the coordinates of the **lowest-numbered claiming rank**.
- **Boundary constraints**: boundary points only slide within their wall, edge or (fixed)
  corner and are snapped exactly onto it.
- **The 3D improvement step**: only the owner of a vertex moves it and sends the new position
  to its neighbours. A flip is decided only by the owner of a vertex that is in all its old
  tets, and sent to every rank that holds one of them. Each rank keeps only the tets near its
  tile; the rest of its halo is dropped before anything reads it. Quantities are evaluated in
  an order fixed by the hash ids, so ranks that store a tet differently agree.

**When editing anything in the geometry path, preserve determinism.** Anything that makes a
position depend on rank ordering, unordered-container iteration order, local point counts or
floating-point accumulation order produces cracks or duplicated/lost vertices at tile
interfaces, usually seen as a failure in `CheckMeshIntegrity` or a hang/error in
`DMPlexCreateFromCellListParallelPetsc`.

The integrity check also compares every ghost vertex with its owner's copy bitwise. In 3D it
fails on any badly shaped tet (`MIN_ETA3_3D`, `MIN_DIHEDRAL_DEG_3D`). Re-measure the mesh
quality before changing those thresholds or the improvement step.

## Agglomeration

`agglomeration_factor` (k) makes the fine tile grid a refinement of the grid a plain
`comm_size/k` run would use, and numbers ranks so coarse group j is exactly ranks
`[j*k, (j+1)*k)`. Tile↔rank conversion must go through `tile_to_rank` / `rank_to_tile`: never
open-code `ty * TILE_DIM[0] + tx`. Ownership tie-breaks are deterministic under any rank
permutation, so this doesn't affect determinism.

## Conventions when editing

- **Style**: follow what's there: 4-space indent, `static` for everything not in the header,
  `snake_case` geometry helpers, `CamelCase` larger/PETSc-facing routines, and
  `// ~~~~~~~~~~~~~~~~~` section separators.
- **C++11 target**: `Point` has explicit constructors for C++11. Don't use newer features.
  `test_lib.c` and `mesh_checksum.c` are C.
- Free large containers with the `std::vector<T>().swap(v)` idiom, as the code does, to return
  memory at scale.
- PETSc error handling is loose by design in places (`(void)ierr;`, `PetscCallVoid`). Match the
  surrounding function rather than changing its signature.
- Variable names carry meaning: `points_with_halos` (everything generated),
  `points_on_owned_triangles_and_orphans` (what survives filtering, includes ghosts),
  `triangles_owned` (simplices, tetrahedra in 3D). Don't blur them.
- Commented-out code (`remove_duplicates`, hash-collision checking) documents things that were
  tried; leave it unless asked.
- `ANNEAL_ITERS` and `final_smooth_its` both set the halo width, so raising them raises memory
  and the minimum viable elements per rank (in 3D, cubically).

## Verifying a change

Single-rank runs exercise none of the interface logic, so always run multi-rank cases with the
integrity check on (it is on by default in the executable), then the full gate:

```bash
make clean && make && mpiexec -n 2 ./BoxMeshDM -target_edge_length 0.01 -integrity_check 1
```

```bash
mpiexec -n 8 ./BoxMeshDM -dim 3 -target_edge_length 0.022 -final_smooth_its 0
```

```bash
make clean && make tests
```

For changes that can reach 2D, add the `mesh_checksum` comparison above. In 3D pick sizes with
at least `2 * (11 + final_smooth_its)` edge lengths per tile along every split axis.

To inspect a mesh, run with `-write_mesh true` (needs HDF5), then
`${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5` and open the `.xmf` in Paraview.

Contributions follow [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).
