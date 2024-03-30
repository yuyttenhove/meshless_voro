# `meshless_voronoi`

<!-- cargo-rdme start -->

**An implementation of the
[Meshless Voronoi algorithm](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf)
in Rust.**

The algorithm is primarily aimed at generating 3D
[Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram), but can
also be used to compute 1D and 2D Voronoi diagrams.

Like [`Voro++`](https://math.lbl.gov/voro++/), this algorithm is *meshless*
implying that no global geometry is constructed. Instead a cell-based
approach is used and we only compute integrals (cell/face volumes and
centroids) and connectivity information (it is possible to determine a
cell's neighbours).

The algorithm can generate Voronoi tessellations with a rectangular boundary
or periodic boundary conditions and also supports computing a subset of the
Voronoi tessellation.

If necessary, arbitrary precision arithmetic is used to treat degeneracies
and to ensure globally consistent local geometry. See the appendix of [this
reference](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) for more
info:

> <cite>Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, Bruno Lévy. Meshless
> Voronoi on the GPU. ACM Transactions on Graphics, 2018, 37 (6), pp.1-12.
> 10.1145/3272127.3275092. hal-01927559</cite>

## Features

- Construction of 1D, 2D and 3D Voronoi grids.

- Partial construction of grids.

- Parallel construction of the Voronoi grid.

- Saving Voronoi grids to [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5).

- Evaluation of *custom integrals* for cells (e.g. weighted centroid) and
  faces (e.g. solid angles).

## Cargo Features

<!-- cargo-rdme end -->
- `rayon` (enabled by default) — Enable parallel construction of the Voronoi
  grid.

- `malachite` (enabled by default) — Use the `malachite` crate for arbitrary
  precision integer arithmethic. This is slower than rug but builds considerably
  faster and allows the crate itself to remain under the MIT/Apache licenses.

- `rug` — Use the rug crate for arbitrary precision intger arithmethic. This is
  faster than `malachite` but depends on GNU GMP via the `gmp-mpfr-sys` crate
  which makes this crate fall under the GPL license.

- `hdf5` — Allow saving Voronoi grids to HDF5 format.
