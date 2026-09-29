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
not run at scale; after the sliver repair its minimum dihedral angle is about 10 degrees at the
default smoothing (about 5.6 with no final smoothing), see [Known limitations](#known-limitations-and-ideas).

Files:
- [BoxMeshDM.cpp](BoxMeshDM.cpp): the whole generator, templated on the dimension (below).
- [BoxMeshDM_tetgen.cpp](BoxMeshDM_tetgen.cpp) / [.h](BoxMeshDM_tetgen.h): the TetGen Delaunay
  backend (`BoxMeshDM_Delaunay3D`), in its own translation unit so `tetgen.h` (its `REAL` macro,
  lowercase macros, global `orient3d`, C++ exceptions) never meets the 2D code. The TetGen code
  sits behind `#if defined(PETSC_HAVE_TETGEN)`; without it the 3D entry points stop up front with
  one message on rank 0 (and the backend still aborts if it is ever reached).
- [BoxMeshDM.h](BoxMeshDM.h): the C-linkage API, `GenerateBoxMeshDM`/`GenerateBoxMeshDMAgglom`
  and `GenerateBoxMeshDM3D`/`GenerateBoxMeshDM3DAgglom`. Deliberately no C++ overloads; the 2D
  signatures must not change.
- [BoxMeshDM_main.cpp](BoxMeshDM_main.cpp): the executable's `main` (`-dim 2|3`,
  `-domain_depth`), so the library (`BoxMeshDM.o BoxMeshDM_tetgen.o`) never contains a `main`.
- [test_lib.c](test_lib.c): the library test, run on 1 and 2 ranks.
- [mesh_checksum.c](mesh_checksum.c): per-rank hashes of a generated mesh, for bit-identity checks.

## Build and test

Requires PETSc **>= 3.24** with Triangle (`--download-triangle`). TetGen (`--download-tetgen`)
is optional: without it the Makefile warns, `make tests` and `test_lib` skip the 3D cases, and a
3D request aborts at runtime. `PETSC_DIR`/`PETSC_ARCH` must be set for *every* make target,
including `clean` (without them the Makefile's `awk` config check waits on stdin forever). The
Makefile uses only PETSc's standard rules; keep it that way rather than adding custom compile
rules or per-target defines. `ifeq` lines inside a recipe must not be tab-indented.

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
[dockerfiles/](dockerfiles) (debug, opt, 64-bit, PFLARE) plus macOS. Only the macOS job builds
PETSc with TetGen, so only it runs the 3D tests until the Docker base images are rebuilt with
`--download-tetgen`. Debug CI uses `PETSC_OPTIONS="-on_error_abort -fp_trap on"`, so new
floating-point operations must not generate NaN/Inf even transiently. TetGen is compiled at
`-O0` in debug PETSc builds, which is why the 3D test sizes are small and mostly use no final
smoothing.

## Changing the mesh is high risk

The 2D generator has been validated at scale (see the README weak-scaling table) and that can't
easily be repeated. Any change that can reach the 2D path must leave the 2D mesh
**bit-identical**, shown with `mesh_checksum`, old build vs new build, **on the same machine**:

```bash
make clean && make lib && make mesh_checksum     # in a checkout of the old code, and of the new
mpiexec -n 3 ./mesh_checksum -target_edge_length 0.01 > old_n3.txt   # same options for both
diff old_n3.txt new_n3.txt
```

Cover several rank counts (1, 2, 3, 4, 8), agglomeration factors, smoothing counts (including
0), a non-square domain and the scale extremes (`-target_edge_length 1e5 -domain_width 1e7
-domain_height 1e7`, `4e-10` on `1e-7`), and do it in three builds: the opt arch, the opt arch
with `make CXXFLAGS=-march=native` (so the compiler contracts FMAs, as production builds on
e.g. ARCHER2 do, and plain builds on x86 don't), and the debug arch. Identical hashes in all
three are the gate; "close enough" is not. Before trusting a comparison, check it is sensitive:
a deliberate one-ulp change (e.g. `/3.0` to `*(1.0/3.0)` in Lloyd) must change the hashes.

3D changes may alter the 3D mesh, but say so explicitly, and still use `mesh_checksum -dim 3`
to show when a 3D change is meant to be neutral. Changes to checks, errors and validation are
much safer than changes to geometry.

## The dimension template

Everything is written once against `Point<DIM>` (`double c[DIM]` plus the `unique_hash_id`)
and `Simplex<DIM>` (`int v[DIM + 1]`), and instantiated through `GenerateBoxMeshDMImpl<2>` and
`<3>`. Globals are axis-indexed (`DOMAIN_SIZE[3]`, `TILE_DIM[3]`, ...).

- **Dimension logic lives only in explicit specialisations.** Never write `if (DIM == 3)` in a
  template body (C++11 has no `if constexpr`; the dead branch still compiles and trips
  `-Warray-bounds`). The per-dimension pieces are: `create_point_with_unique_hash_id`,
  `far_wall_grid_index`, `interior_seed`, `triangulation` (Triangle `"zQ"` / TetGen `"QJz"`),
  `order_points_for_delaunay`, `simplex_volume`, `simplex_centroid`, `dist_sq`,
  `star_simplex_badness` (2D max cosine, 3D `-eta^3`), `get_simplex_edges`,
  `get_boundary_features`, `dm_cell_vertex_order`, `num_walls`, `classify_point_wall`,
  `facet_wall_value`, `integrity_accumulate_simplex`, `integrity_accumulate_owned_point`,
  `RepairSlivers`, `evaluate_integrity`, `CheckDMIntegrity`,
  `stats_accumulate_simplex`, `edge_orientation_bin`, `print_simplex_stats`,
  `factorize_min_cut`, `validate_inputs`, `volume_tolerance`, `print_domain_header`. A 3D retune
  of anything shared becomes a `<3>` specialisation, not a branch.
- **Token-for-token rule for the 2D path.** Every floating-point expression the 2D build
  executes keeps the old code's exact form: parenthesisation, association, divide vs multiply
  by inverse, operand types, literal constants, strict vs non-strict comparisons. "Equivalent"
  rewrites change the last bit and so the mesh.
- **Order rule.** Every loop whose order feeds floating-point accumulation or output numbering
  keeps the 2D order: Delaunay output order, vertex order within a simplex, spring edge order,
  point generation order (corners, then walls/edges/faces from `get_boundary_features`, then
  interior with `iy` outer and `ix` inner), Lloyd's Gauss-Seidel order, and owned-point numbering
  (triangle order, then vertex order, then orphans). 2D points are never sorted.
- **The FMA lesson.** Under FMA contraction, bit-identity depends on the compiler's
  unrolling and inlining, not just the expression text. A rolled `for (d < DIM)` loop let GCC
  hoist products out of loops and vectorise interleaved accumulators, so some multiply-adds
  stopped contracting and others started. The code guards this with `StaticFor<N>::run`
  (compile-time unrolled per-axis/per-vertex loops), `BOXMESHDM_NOINLINE` on
  `apply_boundary_constraint`, and separate per-axis accumulator arrays rather than one
  interleaved `[n*DIM]` array. Keep these, and re-run the full checksum comparison, including
  the `-march=native` build, after **any** change to generic code. Other compilers (clang, Cray,
  Intel) are not verified.
- **Optimisation level.** The protocol is at PETSc's -O2. The 2D mesh equals main's at
  -O1/-O2/-Os with FMA and in every non-FMA build. At -O3 with FMA it equals main compiled with
  apply_boundary_constraint out of line, not main as compiled (main's own FMA output differs
  between -O2 and -O3).

## Algorithm

`GenerateBoxMeshDMImpl<DIM>` validates inputs, sets file-scope globals, factors the ranks into a
tile grid (**exactly one tile per rank**, `factorize_min_cut` minimising the cut length/area),
then runs `process_tile` → optional `CheckMeshIntegrity` / `ComputeAndPrintStats` → `CreateDM`
→ optional `CheckDMIntegrity` → `LabelBoundaries`.

`process_tile`, per rank:
- **Halo.** Points are generated for the tile plus a halo of
  `pad = TARGET_EDGE_LENGTH * (ANNEAL_ITERS + final_smooth_its + 8)` (`ANNEAL_ITERS` = 3), since
  smoothing distortion travels about one edge per iteration. It hard-errors if `pad` exceeds
  half the tile size along any axis split between ranks (that would need neighbour-of-neighbour
  data), which is why few elements per rank fails. An axis with a single tile has no
  neighbours, so it isn't checked. So every split axis needs a tile of at least
  `2 * (11 + final_smooth_its)` edge lengths: 30 at the default smoothing, which in 3D is about
  27k points and 170k tets per rank (about 6.2 tets per point), with up to 8x as many halo
  points as owned points. In 3D every side of the box must also be at least 3 target edge
  lengths (`validate_inputs<3>`), split or not, or the wall exclusion zone rejects every
  interior point.
- **Points.** Boundary points from the feature table (2D: 4 corners then walls L,R,B,T; 3D: 8
  corners, 12 edges, 6 faces), evenly spaced at `length / round(length / TARGET_EDGE_LENGTH)`
  (stepping by the edge length could strand a point next to a corner as a sliver); one jittered
  interior point per grid cell, rejected near the walls. In 3D the points are then sorted by
  `unique_hash_id` so any cospherical tie in TetGen is broken the same way on every rank, and
  face points are jittered in-plane from the first iteration so the boundary lattice is not
  left cospherical.
- **Smoothing.** `ANNEAL_ITERS` rounds of jitter → triangulate → Lloyd → spring →
  `ResolveBoundaryOwnership`, then `final_smooth_its` rounds without jitter, then a final
  triangulation and `ResolveBoundaryOwnership`. Only points inside tile ± `sync_margin` move;
  the outer halo rim is frozen so it doesn't collapse inward.
- **Sliver repair (3D only).** `RepairSlivers<3>` (a no-op `<2>`) keeps the final
  tetrahedralisation and, for up to 24 rounds, moves vertices of tets with `eta^3 < 0.05`
  (`SLIVER_REPAIR_ETA3_3D`) along the normal of the opposite face and the `eta^3` gradient of the
  worst tet in their star, accepting a move only if the star's worst `eta^3` strictly improves.
  Each round only an independent set moves (a point goes only if its star is worse than that of
  every other movable point in it, ties by a hash priority), so no two vertices of a tet move
  together and nothing inverts. It exists because Lloyd removes slivers from the connectivity
  it is given but every re-triangulation brings them back. Nothing is re-triangulated, so the
  output is no longer exactly Delaunay (every cell stays positively oriented) and the halo does
  not grow (a point moves at most 24 x 0.4 edge lengths). Only the **owner** of a point moves it
  (its tetrahedralisation is exact at least `pad` inside its cloud) and sends the new position to
  the neighbours using it (`SendMovedPoints3D`, point to point over the fixed list of <= 26
  neighbouring ranks: counts on tag 106, then data on tags 104/105). It is deliberately not
  `ResolveBoundaryOwnership`: there the lowest claiming rank wins, which need not be the owner,
  and that threw away most repairs within `sync_margin` of an interface. `eta^3` is evaluated in
  a vertex order sorted by hash id (`canonical_tet_order`) so ranks storing a tet differently
  agree. Cost: about 4% of `process_tile` at the default smoothing, more at 0.
- **Filtering.** A simplex belongs to the rank owning its vertex with the **smallest
  `unique_hash_id`**. Points owned spatially but in no owned simplex are kept as **orphans** so
  the rank can still hand out their global ids.

`CreateDM` numbers owned points with `MPI_Exscan` and fetches ghost ids from their owners
(tags 100/101), then calls `DMPlexCreateFromCellListParallelPetsc` with only owned coordinates.
In 3D every tetrahedron has vertices 0 and 1 swapped (`dm_cell_vertex_order<3>`), as PETSc's own
TetGen interface does, because PETSc's reference tetrahedron has the opposite orientation.
`DMPlexDistributeSetDefault(dm, PETSC_FALSE)` is deliberate: the mesh is already balanced.

`LabelBoundaries` writes `"Face Sets"` and `"markers"`:

| Value | 2D | 3D (PETSc `DMPlexCreateBoxMesh` convention) |
| --- | --- | --- |
| 1 | Bottom, y=0 | Bottom, z=0 |
| 2 | Right, x=width | Top, z=depth |
| 3 | Top, y=height | Front, y=0 |
| 4 | Left, x=0 | Back, y=height |
| 5 | | Right, x=width |
| 6 | | Left, x=0 |

In 3D a face is labelled when every vertex in its closure is on the wall (vertices are snapped
exactly, so no centroid is involved). A refine hook relabels refined meshes using the domain
size attached to each DM (not the globals, which the next generator call overwrites).

## Determinism is the core invariant

There is no global consensus step for geometry: ranks that generate the same point must produce
*bit-identical* coordinates for it. This relies on:

- **`unique_hash_id`**, built from *global* grid indices, not coordinates. 2D packs
  `[type:2][ix:31][iy:31]` with the right/top walls at index `MAX_GRID_IDX`; 3D packs
  `[type:1][ix:21][iy:21][iz:21]` with the far walls at `FAR_GRID_IDX_3D = 2^21-1`. Every other
  index must stay below the far-wall index (checked in `validate_inputs`).
- **Stateless RNG**: `splitmix64` seeded from the grid indices (`interior_seed`) for placement
  and from `unique_hash_id ^ iteration` for jitter. No shared state, no rank dependence.
- **`ResolveBoundaryOwnership`**: after each smoothing round, neighbouring ranks (8 in 2D, 26 in
  3D) exchange claims (tag 999) and all adopt the coordinates of the **lowest-numbered claiming
  rank**. The 3D sliver repair is the exception: owner wins (tags 104-106), see above. So the 3D
  mesh is not bitwise invariant to `pad` (a different pad changes what the lowest rank sees
  within `sync_margin`); its quality statistics agree across rank counts to about 0.02%.
- **Boundary constraints**: boundary points only slide within their wall, edge or (fixed)
  corner and are snapped exactly onto it; interior points are kept clear of the `EPSILON`
  capture zone.

**When editing anything in the geometry path, preserve determinism.** Anything that makes a
position depend on rank ordering, unordered-container iteration order, local point counts or
floating-point accumulation order produces cracks or duplicated/lost vertices at tile
interfaces, usually seen as an Euler, perimeter or surface-area failure in
`CheckMeshIntegrity` or a hang/error in `DMPlexCreateFromCellListParallelPetsc`.
`CheckMeshIntegrity` also asks the owner of every ghost vertex for its coordinates (tags
102/103) and fails on any bitwise mismatch, since simplices are built from the local copy but
only the owner's copy reaches the DM. In 3D the Euler characteristic is checked on the created
DM (`CheckDMIntegrity<3>`: `V - E + F - C == 1` from owned points per depth), the pre-DM
check adds total signed volume, boundary surface area, non-positive tets, bad edges and the
boundary surface being a sphere (`Vb - Fb/2 == 2`), and element quality: it fails on any tet
with `eta^3 < MIN_ETA3_3D` (5e-4) or a dihedral angle below `MIN_DIHEDRAL_DEG_3D` (1 degree),
printing the first few (`LOW QUALITY TETRAHEDRON`). `eta` is the mean ratio (1 for a regular
tet), so `eta^3` goes to 0 for slivers and also catches needles whose angles look fine. The
thresholds come from measured meshes: about 12x (`eta^3`) and 4x (angle) below the worst
repaired mesh (6.3e-3 / 3.78 degrees: 6 ranks, agglomerated, no final smoothing) and 16x / 3x
above the best unrepaired one, so the pre-repair code fails them. If you change the smoothing or
the repair, re-measure before touching them. The 3D stats print three histograms: `eta^3`,
the smallest dihedral angle per tet (with the count below 5 degrees) and the largest.

## Known limitations and ideas

- **Bulk quality** (3D): the repair only fixes the worst tets. About 2.3% of tets have a
  dihedral angle below 20 degrees at the default smoothing (7.4% at 0; the repair raises this a
  little as slivers become 10-20 degree tets); mean `eta^3` is about 0.61. More final smoothing
  helps but widens the halo.
- **Not Delaunay** (3D): the repaired mesh is valid and positively oriented, not Delaunay.
- **Not run at scale** (3D), including the repair.
- Ideas not yet pursued: sliver exudation via a weighted Delaunay with hash-derived weights
  (TetGen supports weights); a repair threshold of 0.1 instead of 0.05 (about 6x the cost);
  quality-guarded moves for the 10-20 degree band. Already measured and rejected (see the
  commit "Repair slivers on the final 3D tetrahedralisation"): an ODT target, sliver-normal
  candidates inside Lloyd, less or no spring.

## Agglomeration

`agglomeration_factor` (k) makes the fine tile grid a refinement of the grid a plain
`comm_size/k` run would use, and numbers ranks so coarse group j is exactly ranks
`[j*k, (j+1)*k)`. Tile↔rank conversion must go through `tile_to_rank` / `rank_to_tile` (plain
x-fastest at k=1): never open-code `ty * TILE_DIM[0] + tx`. Ownership tie-breaks are
deterministic under any rank permutation, so this doesn't affect determinism.

## Conventions when editing

- **Style**: follow what's there: 4-space indent, `static` for everything not in the header,
  `snake_case` geometry helpers, `CamelCase` larger/PETSc-facing routines, and
  `// ~~~~~~~~~~~~~~~~~` section separators.
- **C++11 target**: `Point` has explicit constructors for C++11. Don't use newer features.
- **C in `test_lib.c` and `mesh_checksum.c`**: they are built with PETSc's C compiler.
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
mpiexec -n 2 ./BoxMeshDM -dim 3 -target_edge_length 0.02 -final_smooth_its 0
mpiexec -n 8 ./BoxMeshDM -dim 3 -target_edge_length 0.022 -final_smooth_its 0   # 2x2x2: edge and corner neighbours
```

```bash
make clean && make tests
```

PETSc compiles with `-std=gnu++20`, so nothing in the build enforces the C++11 target. Check it
by hand:

```bash
mpicxx -std=c++11 -Wall -Wextra -fsyntax-only -I$PETSC_DIR/include -I$PETSC_DIR/$PETSC_ARCH/include BoxMeshDM.cpp BoxMeshDM_tetgen.cpp BoxMeshDM_main.cpp
```

For 2D changes, add the `mesh_checksum` comparison above. 3D runs are slow in a debug build
(minutes for a few hundred thousand tets); pick sizes that pass the halo rule on every split axis.

To inspect a mesh, run with `-write_mesh true` (needs HDF5), then
`${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py box_mesh.h5` and open the `.xmf` in Paraview.

Contributions follow [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).
