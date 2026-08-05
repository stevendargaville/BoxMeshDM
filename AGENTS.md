# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

BoxMeshDM generates **fully unstructured 2D triangular meshes on a rectangular domain
`[0,width] x [0,height]`, in parallel with MPI, and returns a distributed PETSc `DMPlex`.**

The whole thing is one translation unit: [BoxMeshDM.cpp](BoxMeshDM.cpp) (~2000 lines) plus a
one-function public header [BoxMeshDM.h](BoxMeshDM.h). It can be built either as a standalone
executable (`main` guarded by `STANDALONE_MESH_GEN`) or as a library (`libboxmeshdm`).

The design point that drives everything: generate a *load-balanced, fully unstructured* mesh
with **no file I/O and no mesh partitioner** (no ParMETIS). Each MPI rank generates its own
piece of the mesh directly in the right place, so no redistribution step is needed. The price
is the restrictions listed in [README.md](README.md) (rectangular domain only, uniform
resolution, not robust below ~100k elements/rank, result depends on rank count).

## Dependencies and build

Requires PETSc **>= 3.24** configured with Triangle (`--download-triangle`). `PETSC_DIR` and
`PETSC_ARCH` must be set; the Makefile pulls compilers/flags from PETSc's config and errors out
if the version is too old or Triangle is missing. For meshes beyond ~2B global points PETSc
must also be configured `--with-64-bit-indices`.

```bash
make clean && make          # executable ./BoxMeshDM (defines -DSTANDALONE_MESH_GEN)
```

```bash
make clean && make lib      # libboxmeshdm.so / .dylib / .a (no STANDALONE define)
```

```bash
make clean && make tests    # executable tests, then builds lib and runs tests_lib
```

`make tests` is the gate — it runs the executable across several edge lengths, smoothing
counts, flag combinations, non-square domains, agglomeration factors, and 1 and 2 MPI ranks,
then builds the library and runs [test_lib.c](test_lib.c) on 1 and 2 ranks. CI
([.github/workflows/ci_build.yml](.github/workflows/ci_build.yml)) runs exactly this inside the
Docker images in [dockerfiles/](dockerfiles) (debug, opt, 64-bit, PFLARE) plus a macOS build.
Debug CI runs with `PETSC_OPTIONS="-on_error_abort -fp_trap on"`, so new floating-point
operations must not generate NaN/Inf even transiently.

Build products (`BoxMeshDM`, `BoxMeshDM.o`, `libboxmeshdm.so`, `test_lib`) live in the repo
root and are untracked; there is no `.gitignore`.

## Public API

```c
DM GenerateBoxMeshDM(MPI_Comm comm, double target_edge_length,
                     double width, double height, int final_smooth_its,
                     PetscBool integrity_check, PetscBool print_stats);

DM GenerateBoxMeshDMAgglom(MPI_Comm comm, double target_edge_length,
                           double width, double height, int final_smooth_its,
                           PetscBool integrity_check, PetscBool print_stats,
                           int agglomeration_factor);
```

Returns a distributed `DMPlex` named `"Mesh"`, or `NULL` if `integrity_check` is on and the
mesh fails validation. Collective on `comm`. Caller destroys with `DMDestroy`.

`GenerateBoxMeshDMAgglom` holds the implementation; `GenerateBoxMeshDM` is a forwarder passing
`agglomeration_factor = 1`. Both have C linkage, so C and C++ callers use the same two names —
deliberately no C++-only overload, since C linkage allows only one function per name and having
the spelling differ by language would be a trap.

Executable options (parsed in `main`): `-target_edge_length`, `-final_smooth_its`,
`-domain_width`, `-domain_height`, `-write_mesh`, `-integrity_check`, `-print_stats`,
`-agglomeration_factor`.

## Algorithm / control flow

`GenerateBoxMeshDM` (BoxMeshDM.cpp:1841) is the driver:

1. **Set globals** — `TARGET_EDGE_LENGTH`, `DOMAIN_WIDTH`, `DOMAIN_HEIGHT`, and the tolerances
   `TOL_LEN`, `TOL_LEN_SQ`, `TOL_VOLUME` (all derived from the target edge length).
2. **Factor `comm_size` into `TILE_DIM_X x TILE_DIM_Y`** minimising the total interface cut
   length for the domain aspect ratio (`factorize_min_cut`). **Exactly one tile per rank**, so
   tiles and ranks are in bijection — see [Agglomeration](#agglomeration) for the mapping.
3. **`process_tile`** (BoxMeshDM.cpp:818) — each rank builds its own subdomain, independently.
4. **`ComputeValenceAndEdges`**, then optionally **`CheckMeshIntegrity`** and
   **`ComputeAndPrintStats`**.
5. **`CreateDM`** (BoxMeshDM.cpp:1119) — assign global vertex numbers and build the `DMPlex`.
6. **`LabelBoundaries`** (BoxMeshDM.cpp:1335) plus a `DMRefineHookAdd` so labels survive
   `-dm_refine`.

### `process_tile` — point generation and smoothing

- **Padding/halo.** `pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8)`. Each
  rank generates points not just for its tile but for a halo of width `pad` around it, because
  smoothing distortion propagates roughly one edge per iteration. The code hard-errors if
  `pad > min(tile_size)/2` — that would require neighbour-of-neighbour data. This is why the
  code is not robust with few elements per rank.
- **Point creation.** Corners explicitly; then boundary points on a regular 1D lattice of
  spacing `TARGET_EDGE_LENGTH` along each of the four walls; then interior points, one
  pseudo-randomly placed inside each cell of a `TARGET_EDGE_LENGTH` grid, with rejection
  sampling near the walls (`exclusion = TARGET_EDGE_LENGTH * (START_JITTER + 0.25)`) so
  interior points can never collide with the boundary lattice.
- **Anneal loop.** `ANNEAL_ITERS` (=3) rounds of `apply_jitter` → `triangulation` →
  `relax_points_lloyd` → `relax_points_spring` → `ResolveBoundaryOwnership`, followed by
  `final_smooth_its` rounds of the same without jitter, then one final triangulation.
- **Freezing.** Smoothing only moves points inside a "safe box" (tile ± `sync_margin`, where
  `sync_margin = pad - 1.5 * TARGET_EDGE_LENGTH`). The outer rim of each halo is frozen so the
  halo doesn't collapse inward. Every point that is *allowed* to move is inside the region that
  gets synchronised across ranks.
- **Filtering.** A triangle is owned by the rank owning the triangle vertex with the **smallest
  `unique_hash_id`** — a rule every rank computes identically. Halo triangles are discarded.
  Points spatially owned by this rank but not in any owned triangle are kept as **orphans**, so
  the rank can still hand out their global IDs.

### Determinism is the core invariant

There is no global consensus step for geometry. Instead, ranks that generate the same point
must generate *bit-identical* coordinates for it. This is enforced by:

- **`unique_hash_id`** (`create_point_with_unique_hash_id`): packs `[type:2][ix:31][iy:31]`
  into a `uint64_t` from the *global* grid indices, not from coordinates. Two ranks generating
  the same lattice site produce the same ID. `type` is 0=interior, 1=boundary/corner.
- **Stateless RNG.** `splitmix64` seeded from the packed `(ix, iy)` for placement, and from
  `unique_hash_id ^ iteration` for jitter. No shared RNG state, no rank dependence.
- **`ResolveBoundaryOwnership`** (BoxMeshDM.cpp:146): after every smoothing round, ranks
  exchange `{id, geo_rank, x, y}` claims with their 8 grid neighbours, and every rank adopts
  the coordinates supplied by the **lowest-numbered claiming rank**. This kills float drift
  before it can cause ranks to disagree about geometry.
- **Boundary constraints.** `apply_boundary_constraint` zeroes the normal component of any
  displacement for points on a wall (they slide tangentially only) and snaps them exactly onto
  the wall; `keep_interior_point_inside` reflects interior points back and keeps them clear of
  the `EPSILON` capture zone so they never get misclassified as boundary points.

**When editing anything in the geometry path, preserve determinism.** Any change that makes a
point's position depend on rank ordering, iteration order over unordered containers, local
point counts, or non-associative floating-point accumulation ordering will produce cracks
between subdomains or duplicated/lost vertices at tile interfaces — usually showing up as an
Euler-characteristic or perimeter failure in `CheckMeshIntegrity`, or as a hang/error inside
`DMPlexCreateFromCellListParallelPetsc`.

### Agglomeration

`agglomeration_factor` (k) makes the fine rank grid a refinement of the grid a plain
`comm_size/k`-rank run would use, and reorders ranks so the k fine ranks composing coarse group
j are exactly ranks `[j*k, (j+1)*k)`. `factorize_min_cut` is called twice: once on `comm_size/k`
against the domain aspect ratio to get `COARSE_DIM_X x COARSE_DIM_Y`, once on k against the
*coarse tile* aspect ratio to get the compact sub-block `SUB_DIM_X x SUB_DIM_Y`;
`TILE_DIM_X/Y` is their product. Tile↔rank conversion goes exclusively through `tile_to_rank`
and `rank_to_tile`, which reduce to plain row-major at k=1 — never open-code
`ty * TILE_DIM_X + tx` again. Since global numbering is contiguous in rank order, each coarse
group's rows form one contiguous range, so a consumer can merge them without a repartitioner.

Ownership tie-breaks (hash ids, "lowest claimant rank") are deterministic under any rank
permutation, so the reordering does not affect the determinism invariants above.

### `CreateDM` — global numbering

Owned points get contiguous global IDs via `MPI_Exscan`. Ghost points (vertices of owned
triangles owned by a neighbour) are resolved in a two-phase point-to-point exchange: send the
`unique_hash_id` to the owner (tag 100), receive the global ID back (tag 101). Cells are then
handed to `DMPlexCreateFromCellListParallelPetsc` with only the *owned* vertex coordinates.
`DMPlexDistributeSetDefault(dm, PETSC_FALSE)` is set deliberately — the mesh is already
balanced and calling a partitioner would be expensive. Users who want ParMETIS can call
`DMPlexDistribute` themselves.

### Labels

`LabelBoundaries` labels vertices and facets geometrically into both `"Face Sets"` and
`"markers"`, with values **1=Bottom, 2=Right, 3=Top, 4=Left**. Corners resolve by that priority
order. `RefineHook_LabelBoundaries` re-labels and re-registers itself after each refinement.

### Validation

- `CheckMeshIntegrity` (BoxMeshDM.cpp:1494) checks, globally: total area == `width*height`,
  boundary perimeter == `2*(width+height)`, Euler characteristic `V - E + F == 1`, and that no
  edge exceeds `3x` the target length. Returns false on rank 0's verdict, broadcast to all;
  `GenerateBoxMeshDM` then returns `NULL`. Note the two hardcoded failure messages say
  "Expected 1.0" / "Expected 4.0" — those strings are stale for non-unit domains, the
  comparison itself is correct.
- `ComputeAndPrintStats` prints point counts and load imbalance, valence histogram, triangle
  count, min/max volume and ratio, min/max angle, average edge length, and an edge-orientation
  histogram in 10-degree bins. Rank 0 only.

Both cost extra time and memory; production runs should disable them.

## Tunables (file-scope in BoxMeshDM.cpp)

| Name | Value | Meaning |
| --- | --- | --- |
| `EPSILON` | `1e-13` | Boundary-membership tolerance. Used for snapping and labelling. |
| `START_JITTER` | `0.30` | Jitter amplitude as a fraction of target edge length. |
| `ANNEAL_ITERS` | `3` | Jitter+smooth rounds before the jitter-free final smooths. |
| `TILE_DIM_X/Y` | computed | Rank grid; one tile per rank. |
| `AGG_FACTOR`, `COARSE_DIM_X/Y`, `SUB_DIM_X/Y` | computed | Agglomeration; all 1/trivial by default. |
| `TOL_LEN`, `TOL_LEN_SQ`, `TOL_VOLUME` | derived | Length/degenerate-triangle tolerances. |

`ANNEAL_ITERS` and `final_smooth_its` both feed the halo width, so raising them raises memory
and the minimum viable elements-per-rank.

## Conventions when editing

- **Style**: follow what's already there — 4-space indent, `static` for everything not in the
  header, `snake_case` for local helpers and geometry functions, `CamelCase` for the
  larger-scale/PETSc-facing routines, `// ~~~~~~~~~~~~~~~~~` as the section separator.
- **C++11 target** — `Point` has explicit constructors specifically for C++11 compatibility.
  Don't reach for newer language features.
- The code deliberately frees large containers with the `std::vector<T>().swap(v)` idiom to
  return memory to the allocator at scale. Keep that pattern for anything large.
- PETSc error handling in this file is loose by design in places (`(void)ierr;`,
  `PetscCallVoid` in void functions). Match the surrounding function rather than converting a
  function's signature.
- Variable names carry meaning: `points_with_halos` (everything a rank generated),
  `points_on_owned_triangles_and_orphans` (what survives filtering, includes ghosts),
  `triangles_owned`. Don't blur these.
- There is commented-out code (`remove_duplicates`, hash-collision checking) kept as
  documentation of things that were tried; leave it unless asked.
- MPI tags in use: 100/101 (`CreateDM` global-ID exchange), 999 (`ResolveBoundaryOwnership`).

## Verifying a change

Always run at least one multi-rank case with the integrity check on — single-rank runs exercise
none of the interface logic:

```bash
make clean && make && mpiexec -n 2 ./BoxMeshDM -target_edge_length 0.01 -integrity_check 1
```

Full gate before proposing a change:

```bash
make clean && make tests
```

To inspect a mesh visually, run with `-write_mesh true` (needs PETSc with HDF5), convert with
`${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5`, and open the `.xmf` in Paraview.

## Contributing rules

[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) applies: open an issue first, branch off and
PR into `main`, rebase rather than merging `main` in, descriptive commits, all tests passing.
MIT licensed.
