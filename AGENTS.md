# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

BoxMeshDM generates **fully unstructured 2D triangular meshes on a rectangular domain
`[0,width] x [0,height]`, in parallel with MPI, and returns a distributed PETSc `DMPlex`.**
The design point: a *load-balanced, fully unstructured* mesh with **no file I/O and no mesh
partitioner**. Each rank generates its own piece directly in the right place. The price is the
restrictions in [README.md](README.md) (rectangular domain, uniform resolution, not robust below
~100k elements/rank, result depends on rank count).

The mesh generator is one translation unit, [BoxMeshDM.cpp](BoxMeshDM.cpp), with the public API
in [BoxMeshDM.h](BoxMeshDM.h) (`GenerateBoxMeshDM`, and `GenerateBoxMeshDMAgglom` which holds the
implementation; both C linkage, deliberately no C++ overloads). The executable's `main` lives in
[BoxMeshDM_main.cpp](BoxMeshDM_main.cpp), so the library is built from `BoxMeshDM.o` alone and
never contains a `main`.

## Build and test

Requires PETSc **>= 3.24** with Triangle (`--download-triangle`), and `PETSC_DIR`/`PETSC_ARCH`
set for *every* make target, including `clean` (without them the Makefile's `awk` config check
waits on stdin forever). The Makefile uses only PETSc's standard rules; keep it that way rather
than adding custom compile rules or per-target defines.

```bash
make clean && make          # executable ./BoxMeshDM
```

```bash
make clean && make lib      # libboxmeshdm.so / .dylib / .a
```

```bash
make clean && make tests    # the gate: executable runs, then builds lib and runs test_lib
```

CI ([ci_build.yml](.github/workflows/ci_build.yml)) runs `make tests` in the images in
[dockerfiles/](dockerfiles) (debug, opt, 64-bit, PFLARE) plus macOS. Debug CI uses
`PETSC_OPTIONS="-on_error_abort -fp_trap on"`, so new floating-point operations must not
generate NaN/Inf even transiently.

## Changing the mesh is high risk

The generator has been validated at scale (see the README weak-scaling table) and that can't
easily be repeated. Prefer changes that leave the generated mesh **bit-identical**, and show it
by comparing a checksum of the final DM (coordinates, cell count, labels) against the old code
on several rank counts, agglomeration factors and smoothing counts. If a change must alter the
mesh, say so explicitly and keep it as narrow as possible. Changes to checks, errors and
validation are much safer than changes to geometry.

## Algorithm

`GenerateBoxMeshDMAgglom` validates inputs, sets file-scope globals, factors the ranks into a
tile grid (**exactly one tile per rank**), then runs `process_tile` → optional
`CheckMeshIntegrity` / `ComputeAndPrintStats` → `CreateDM` → `LabelBoundaries`.

`process_tile`, per rank:
- **Halo.** Points are generated for the tile plus a halo of
  `pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8)`, since smoothing distortion
  travels about one edge per iteration. It hard-errors if `pad > min(tile_size)/2` (that would
  need neighbour-of-neighbour data), which is why few elements per rank fails.
- **Points.** Explicit corners; wall points evenly spaced at
  `length / round(length / TARGET_EDGE_LENGTH)` (stepping by the edge length could strand a
  point next to a corner as a sliver); one jittered interior point per grid cell, rejected near
  the walls.
- **Smoothing.** `ANNEAL_ITERS` rounds of jitter → triangulate → Lloyd → spring →
  `ResolveBoundaryOwnership`, then `final_smooth_its` rounds without jitter, then a final
  triangulation and `ResolveBoundaryOwnership`. Only points inside tile ± `sync_margin` move;
  the outer halo rim is frozen so it doesn't collapse inward.
- **Filtering.** A triangle belongs to the rank owning its vertex with the **smallest
  `unique_hash_id`**. Points owned spatially but in no owned triangle are kept as **orphans** so
  the rank can still hand out their global ids.

`CreateDM` numbers owned points with `MPI_Exscan` and fetches ghost ids from their owners
(tags 100/101), then calls `DMPlexCreateFromCellListParallelPetsc` with only owned coordinates.
`DMPlexDistributeSetDefault(dm, PETSC_FALSE)` is deliberate: the mesh is already balanced.

`LabelBoundaries` writes `"Face Sets"` and `"markers"` (1=Bottom, 2=Right, 3=Top, 4=Left). A
refine hook relabels refined meshes using the domain size attached to each DM (not the
globals, which the next `GenerateBoxMeshDM` call overwrites).

## Determinism is the core invariant

There is no global consensus step for geometry: ranks that generate the same point must produce
*bit-identical* coordinates for it. This relies on:

- **`unique_hash_id`**: packs `[type:2][ix:31][iy:31]` from *global* grid indices, not
  coordinates. The right/top walls use index `MAX_GRID_IDX`, so every other index must stay
  below it (checked on input).
- **Stateless RNG**: `splitmix64` seeded from `(ix, iy)` for placement and from
  `unique_hash_id ^ iteration` for jitter. No shared state, no rank dependence.
- **`ResolveBoundaryOwnership`**: after each smoothing round, neighbouring ranks exchange claims
  (tag 999) and all adopt the coordinates of the **lowest-numbered claiming rank**.
- **Boundary constraints**: wall points only slide tangentially and are snapped exactly onto
  the wall; interior points are kept clear of the `EPSILON` capture zone.

**When editing anything in the geometry path, preserve determinism.** Anything that makes a
position depend on rank ordering, unordered-container iteration order, local point counts or
floating-point accumulation order produces cracks or duplicated/lost vertices at tile
interfaces, usually seen as an Euler or perimeter failure in `CheckMeshIntegrity` or a
hang/error in `DMPlexCreateFromCellListParallelPetsc`.

## Agglomeration

`agglomeration_factor` (k) makes the fine tile grid a refinement of the grid a plain
`comm_size/k` run would use, and numbers ranks so coarse group j is exactly ranks
`[j*k, (j+1)*k)`. Tile↔rank conversion must go through `tile_to_rank` / `rank_to_tile` (plain
row-major at k=1): never open-code `ty * TILE_DIM_X + tx`. Ownership tie-breaks are
deterministic under any rank permutation, so this doesn't affect determinism.

## Conventions when editing

- **Style**: follow what's there: 4-space indent, `static` for everything not in the header,
  `snake_case` geometry helpers, `CamelCase` larger/PETSc-facing routines, and
  `// ~~~~~~~~~~~~~~~~~` section separators.
- **C++11 target**: `Point` has explicit constructors for C++11. Don't use newer features.
- Free large containers with the `std::vector<T>().swap(v)` idiom, as the code does, to return
  memory at scale.
- PETSc error handling is loose by design in places (`(void)ierr;`, `PetscCallVoid`). Match the
  surrounding function rather than changing its signature.
- Variable names carry meaning: `points_with_halos` (everything generated),
  `points_on_owned_triangles_and_orphans` (what survives filtering, includes ghosts),
  `triangles_owned`. Don't blur them.
- Commented-out code (`remove_duplicates`, hash-collision checking) documents things that were
  tried; leave it unless asked.
- `ANNEAL_ITERS` and `final_smooth_its` both set the halo width, so raising them raises memory
  and the minimum viable elements per rank.

## Verifying a change

Single-rank runs exercise none of the interface logic, so always run a multi-rank case with the
integrity check on, then the full gate:

```bash
make clean && make && mpiexec -n 2 ./BoxMeshDM -target_edge_length 0.01 -integrity_check 1
```

```bash
make clean && make tests
```

To inspect a mesh, run with `-write_mesh true` (needs HDF5), then
`${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5` and open the `.xmf` in Paraview.

Contributions follow [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).
