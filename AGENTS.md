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
not run at scale; after the sliver repair, the flips and the guarded smoothing its minimum
dihedral angle is about 21 degrees at the default smoothing (about 19 with no final smoothing,
17.4 in the worst case measured), see
[Known limitations](#known-limitations-and-ideas).

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
[dockerfiles/](dockerfiles) (debug, opt, 64-bit, PFLARE) plus macOS. The Docker base images
and the macOS job all build PETSc with TetGen, so every job runs the 3D tests (a base image
without TetGen would silently test 2D only, as the Makefile skips them). Debug CI uses `PETSC_OPTIONS="-on_error_abort -fp_trap on"`, so new
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
  `factorize_min_cut`, `validate_inputs`, `volume_tolerance`, `print_domain_header`. The flips
  (`FlipTetrahedra3D`) are plain 3D functions called only from `RepairSlivers<3>`. A 3D retune
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
  (stepping by the edge length could strand a point next to a corner as a sliver), and kept by
  index along each free axis (1..n-1 for n = round(length / TARGET_EDGE_LENGTH)), not by
  coordinate: from a side of 1024 up `SIZE - EPSILON` rounds to `SIZE`, so a coordinate test kept
  index n when it landed 1 ulp inside the far edge, next to the explicit point there; one jittered
  interior point per grid cell, rejected near the walls. In 3D the points are then sorted by
  `unique_hash_id` so any cospherical tie in TetGen is broken the same way on every rank, and
  face points are jittered in-plane from the first iteration so the boundary lattice is not
  left cospherical.
- **Smoothing.** `ANNEAL_ITERS` rounds of jitter → triangulate → Lloyd → spring →
  `ResolveBoundaryOwnership`, then `final_smooth_its` rounds without jitter, then a final
  triangulation and `ResolveBoundaryOwnership`. Only points inside tile ± `sync_margin` move;
  the outer halo rim is frozen so it doesn't collapse inward.
- **Sliver repair, flips and guarded smoothing (3D only).** `RepairSlivers<3>` (a no-op `<2>`)
  keeps the final tetrahedralisation and improves it in three steps: a vertex pass, flips that
  change the connectivity (next item), and a second vertex pass on the new connectivity. It takes
  the tet list by (non-const) reference and returns only the rank's *maintained* tets (next item),
  a superset of the ones it owns. First, for 24 rounds,
  vertices of tets with `eta^3 < 0.05` (`SLIVER_REPAIR_ETA3_3D`) try steps along the normal of the
  opposite face and the `eta^3` gradient of the worst tet in their star, keeping the one that
  most improves the star's worst `eta^3`. Then the flips. Then, for 16 rounds
  (`SMOOTH_ROUNDS_3D`), vertices of tets with `eta^3 < 0.2` (`SMOOTH_ETA3_3D`) step down the
  gradient of `F = sum 1/eta^3` over their star (0.3, 0.15, 0.075 edge lengths, the first that lowers `F` and strictly raises the star's
  worst `eta^3`). In both, a move is accepted only if the star's worst `eta^3` strictly improves,
  and each round only an independent set moves (a point goes only if its star is worse than that
  of every other movable point in it, ties by a hash priority), so no two vertices of a tet move
  together and nothing inverts. Whether a point has a move is only computed when an owned point's
  test needs it (the same result as computing it for all). It exists because Lloyd removes
  slivers from the connectivity it is given but every re-triangulation brings them back. Nothing
  is re-triangulated, so the output is no longer exactly Delaunay (every cell stays positively
  oriented), and no move may take a point more than 1 edge length
  (`REPAIR_MAX_DISPLACEMENT_3D`) from its position in the final tetrahedralisation, so the halo
  does not grow. Only the **owner** of a point moves it
  (its tetrahedralisation is exact at least `pad` inside its cloud) and sends the new position to
  the neighbours using it (`SendMovedPoints3D`, point to point over the fixed list of <= 26
  neighbouring ranks: counts on tag 106, then data on tags 104/105). It is deliberately not
  `ResolveBoundaryOwnership`: there the lowest claiming rank wins, which need not be the owner,
  and that threw away most repairs within `sync_margin` of an interface. `eta^3` and its gradient
  are evaluated in a vertex order sorted by hash id (`canonical_tet_order`), and `F` and its
  gradient are summed over the star sorted by the tets' ids (`tet_before`), so ranks storing a tet
  or a star differently agree. Every star a vertex pass reads (an owned point's and its
  neighbours') must be of a maintained point; `refresh` aborts otherwise. Cost of the whole of
  `RepairSlivers` (TEL 0.02, 1 rank, medians of 3 on a loaded machine): 2.6 of 21.7 s of
  `process_tile` at the default smoothing (before the flips: 3.8 of 23.0 s, the smoothing pass
  after the flips has much less to do), 9.6 of 18.7 s at 0 (before: 7.4 of 16.5 s).
- **Flips (3D only).** `FlipTetrahedra3D`, between the two vertex passes: the 2-3 flip of an
  interior face and the removal of an interior edge with a closed ring of up to 7 tets
  (`FLIP_MAX_RING_3D`; 3-2 and 4-4 are rings of 3 and 4), replaced by the max-min triangulation of
  the ring (dynamic programming). Candidates are tets whose smallest dihedral-angle sine
  (`tet_min_sine`, `6V|e|/(|n_k||n_l|)`, no `acos`) is below 0.5 (`FLIP_SINE_3D`, 30 degrees); a
  flip is made only if the smallest sine of its new tets strictly beats that of its old ones
  (which also keeps every new tet positive with a margin, and the boundary surface is never
  touched). Up to 8 rounds (`FLIP_ROUNDS_3D`, stops early when no rank flips); each round
  re-evaluates only candidates with a vertex whose star changed or whose proposal lost (exactly
  the same result as re-evaluating all). **Consistency by construction:** every flip has a *key*
  vertex in all its old tets (the smaller id of the removed edge, the smallest id of the removed
  face); only the key's owner proposes it (it holds all the old tets). Proposals go to the
  neighbours that may maintain one of its vertices (tags 107/108) and a proposal wins if it beats
  (larger new smallest sine, then a hash of the ids) every proposal sharing an old tet with it:
  both owners of two such proposals have both, so at most one wins. Winners go to the same ranks
  (tags 109/110) and every rank applies all of them in hash order. A rank *maintains* exactly the
  tets with a vertex that started within `FLIP_MAINTAINED_MARGIN_3D` = 4.5 edge lengths of its
  tile (fixed set); all other tets are dropped at the start of `RepairSlivers<3>`, and applying a
  flip removes the old tets it holds and adds the new tets with a maintained vertex, so the
  maintained set stays exact round after round (checked: a missing old tet or a missing vertex
  aborts). Measured on 2 ranks at fs 0 (pad 11 edge lengths): the maintained set is exact up to a
  margin of 9 edge lengths (10 fails), and a margin of 1.5 is too small for the vertex passes (2.5
  works). The result does not depend on the rank count beyond what the vertex passes already
  did: minimum tiles (TEL 0.0166, fs 4) give a minimum dihedral angle of 20.26 degrees on 1, 2, 4
  and 8 ranks. Flipping only away from the interfaces (no communication) left 14.3 degrees and
  2.6-3.3% below 30 degrees on 4-8 ranks.
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
  rank**. The 3D sliver repair is the exception: owner wins (tags 104-106), and the flips are
  decided by the owner of their key vertex (tags 107-110), see above. So the 3D
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
thresholds come from measured meshes: they were set about 12x (`eta^3`) and 4x (angle) below
the worst mesh with the sliver repair alone (6.3e-3 / 3.78 degrees: 6 ranks, agglomerated, no
final smoothing) and 16x / 3x above the best unrepaired one, so the pre-repair code fails them.
With the guarded smoothing the same case gives 1.96e-2 / 6.44 degrees, and with the flips
0.106 / 18.2 degrees (the smallest angle measured anywhere is 17.4 degrees, 3 ranks on a 3x1x1
box at fs 0). The thresholds were left alone: they could now be raised several-fold, but would
then no longer describe a mesh with the repair alone, the case they were set for. If you change
the smoothing, the flips or the repair, re-measure before touching them. The 3D stats print three
histograms: `eta^3`, the smallest dihedral angle per tet (with the count below 5 degrees) and the
largest.

## Known limitations and ideas

- **Bulk quality** (3D): at TEL 0.02 on the unit cube no tet has a dihedral angle below 20
  degrees and about 1.2% are below 30 at the default smoothing (5.6% at fs 0, 0.6% at fs 8);
  the largest dihedral angle is about 143 degrees (146 at fs 0) and mean `eta^3` about 0.65
  (0.55 at fs 0, 0.68 at fs 8). The share below 30 at fs 0 is what the flips cannot reach with
  30 degree candidates in 8 rounds; more final smoothing helps but widens the halo. The flips
  remove about 5% of the tets (3-2 flips dominate). Boundary faces are never flipped.
- **Cost at fs 0** (3D): the flips cost about +13% of `process_tile` at fs 0 (many candidates,
  about 180k in the first round at TEL 0.02) while at the default smoothing the whole of
  `RepairSlivers` got cheaper.
- **Not Delaunay** (3D): the repaired mesh is valid and positively oriented, not Delaunay.
- **Not run at scale** (3D), including the repair and the flips.
- Ideas not yet pursued: sliver exudation via a weighted Delaunay with hash-derived weights
  (TetGen supports weights); multi-face removal; flips that change the boundary surface
  triangulation within a wall. Already measured and rejected (see the commits "Repair slivers on
  the final 3D tetrahedralisation", "Add a quality-guarded smoothing pass after the 3D sliver
  repair" and "Flip tetrahedra between the 3D vertex passes"): an ODT target, sliver-normal
  candidates inside Lloyd, less or no spring, a max-min (active-set) smoothing step, which
  equalises a star and lowers its good tets, and `F = sum 1/(eta^3)^2`. For the flips: `eta^3` as
  the flip measure (3.7% below 30 degrees at fs 4 with candidates below 0.2; 1.5% with candidates
  below 0.4, but a repair time of 9.6 s against 5.3 s for the sine measure before it was made
  cheaper), flipping after the smoothing pass instead of before it (minimum 14.6 vs 20.8 degrees
  at fs 4; smoothing again afterwards recovers it at a higher cost), a sine threshold of 0.4 or
  0.6 instead of 0.5 (4.1% below 30 degrees, or no gain at a much higher cost), only 2-3/3-2
  flips (fs 0 minimum 10.3 degrees), rings of at most 4 (fs 0 minimum `eta^3` lower), an extra
  flip pass after the smoothing (a little better, costlier), flips before the sliver repair (a
  little better at fs 4, much costlier at fs 0), and flipping only away from the interfaces (see
  above).

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
