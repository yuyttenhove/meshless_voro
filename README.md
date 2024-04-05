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

## Integer Arithmetic Backend

The default backend for arbitrary precision integer arithemtic is 
[`num_bigint`](https://crates.io/crates/num-bigint) (MIT/Apache 2.0).
For most use cases there is no significant performance difference between the different 
backends and `num_bigint` is the fastest to build.

However, for highly degenerate seed configurations, it is recommended to use the alternative 
[`rug`](https://crates.io/crates/rug) (LGPL-3.0+) backend, which can increase performance 
up to 2x in these cases.
It should be noted that this backend requires a C compiler to build and hence has the slowest build time.

*Using the `rug` backend also changes the license of the crate to LGPL-3.0+.*

## Cargo Features

<!-- cargo-rdme end -->
- `rayon` (enabled by default) – Enable parallel construction of the Voronoi
  grid.

- `rug` – Use the `rug` crate as the arbitrary precision integer arithmethic backend.

- `hdf5` – Allow saving Voronoi grids to HDF5 format.

## License

Apache-2.0 OR MIT at your option when using the default (`num_bigint`) backend OR LGPL-3.0+ (`rug` backend).
