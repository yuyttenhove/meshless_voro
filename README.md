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

You can select from five backends for arbitrary precision integer arithmetic.
These all provide identical functionality and vary only in performance and licensing.

For most practical applications, the choice of backend does not significantly alter
performance. However, for highly degenerate seed configurations - i.e. with many groups of more
than 4 (almost) co-spherical seed points - many arbitrary precision arithmetic tests must be
performed leading to some performance differences in such cases.

- [`dashu`](https://crates.io/crates/dashu) (MIT/Apache 2.0): This is the default backend.
  Can be up to 40% slower than the `rug` backend for highly degenerate seed configurations.

- [`ibig`](https://crates.io/crates/ibig) (MIT/Apache 2.0): Similar performance to the `dashu`
  backend.

- [`num_bigint`](https://crates.io/crates/num-bigint) (MIT/Apache 2.0): Worst performance for
  degenerate seed configurations (measured up to 109% slower than `rug`)

- [`malachite`](https://crates.io/crates/malachite) (LGPL-3.0-only): Slightly faster than the
  `dashu` backend (up to 30% slower than `rug`).

- [`rug`](https://crates.io/crates/rug) (LGPL-3.0+): The fastest backend, but depends on GNU GMP
  via the `gmp-mpfr-sys` crate which requires a C compiler to build and hence has the slowest
  build time.

## Cargo Features

<!-- cargo-rdme end -->
- `rayon` (enabled by default) – Enable parallel construction of the Voronoi
  grid.

- `hdf5` – Allow saving Voronoi grids to [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5).

- `dashu` (enabled by default) — Use the `dashu` crate as the arbitrary precision integer arithmetic backend.

- `ibig` — Use the `ibig` crate as the arbitrary precision integer arithmetic backend.

- `malachite` — Use the `malachite` crate as the arbitrary precision integer arithmetic backend. 
   
  *Disclaimer*: this changes the license to the more restrictive LGPL-3.0-only license.

- `num_bigint` — Use the `num_bigint` crate as the arbitrary precision integer arithmetic backend.

- `rug` – Use the `rug` crate as the arbitrary precision integer arithmetic backend. 
  This can increase performance significantly for highly degenerate seed configurations where lots of arbitrary 
  precision arithmetic is needed. 
  
  *Disclaimer:* this changes the license to the more restrictive LGPL-3.0+ license.

## License

Licensed under:
 - [Apache-2.0](www.apache.org/licenses/LICENSE-2.0) OR [MIT](https://opensource.org/license/MIT) at your option when 
   using the `ibig`, `dashu` or `num_bigint` arbitrary precision arithmetic backends.
 - [LGPL-3.0-only](https://www.gnu.org/licenses/lgpl-3.0.html) when using the `malachite` backend
 - [LGPL-3.0+](https://www.gnu.org/licenses/lgpl-3.0.html) when using the `rug` backend.
