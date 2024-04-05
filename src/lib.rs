//! **An implementation of the
//! [Meshless Voronoi algorithm](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf)
//! in Rust.**
//!
//! The algorithm is primarily aimed at generating 3D
//! [Voronoi diagrams](https://en.wikipedia.org/wiki/Voronoi_diagram), but can
//! also be used to compute 1D and 2D Voronoi diagrams.
//!
//! Like [`Voro++`](https://math.lbl.gov/voro++/), this algorithm is *meshless*
//! implying that no global geometry is constructed. Instead a cell-based
//! approach is used and we only compute integrals (cell/face volumes and
//! centroids) and connectivity information (it is possible to determine a
//! cell's neighbours).
//!
//! The algorithm can generate Voronoi tessellations with a rectangular boundary
//! or periodic boundary conditions and also supports computing a subset of the
//! Voronoi tessellation.
//!
//! If necessary, arbitrary precision arithmetic is used to treat degeneracies
//! and to ensure globally consistent local geometry. See the appendix of [this
//! reference](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) for more
//! info:
//!
//! > <cite>Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, Bruno Lévy. Meshless
//! > Voronoi on the GPU. ACM Transactions on Graphics, 2018, 37 (6), pp.1-12.
//! > 10.1145/3272127.3275092. hal-01927559</cite>
//!
//! # Features
//!
//! - Construction of 1D, 2D and 3D Voronoi grids.
//!
//! - Partial construction of grids.
//!
//! - Parallel construction of the Voronoi grid.
//!
//! - Saving Voronoi grids to [HDF5 format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format#HDF5).
//!
//! - Evaluation of *custom integrals* for cells (e.g. weighted centroid) and
//!   faces (e.g. solid angles).
//!
//! # Integer Arithmetic Backend
//!
//! You can select from three backends for arbitrary precision integer
//! arithmetic.
//!
//! - [`dashu`](https://crates.io/crates/dashu) (MIT/Apache 2.0) runs the test
//!   suit at the same speed as `rug`. This is the default backend.
//!
//! - [`malachite`](https://crates.io/crates/malachite) (LGPL-3.0-only) runs the
//!   test suite about 6% slower than `rug` but builds considerably faster.
//!
//! - [`rug`](https://crates.io/crates/rug) (LGPL-3.0+) runs the test suite
//!   faster than `malachite` but depends on GNU GMP via the `gmp-mpfr-sys`
//!   crate which requires a C compiler to build and has the slowest build time
//!   thus.
//!
//! # Cargo Features
#![doc = document_features::document_features!()]

#[cfg(any(
    all(feature = "rug", feature = "malachite"),
    all(feature = "rug", feature = "malachite-base"),
    all(feature = "rug", feature = "malachite-nz"),
    all(feature = "dashu", feature = "malachite"),
    all(feature = "dashu", feature = "malachite-base"),
    all(feature = "dashu", feature = "malachite-nz"),
    all(feature = "rug", feature = "dashu"),
))]
compile_error!("Illegal feature combination selected");

mod bounding_sphere;
pub mod geometry;
mod part;
mod rtree_nn;
mod simple_cycle;
// Space is no longer used, I left it in as a reference for the gpu
// implementation
#[allow(dead_code)]
mod space;
mod util;
mod voronoi;

pub use voronoi::{integrals, ConvexCell, Voronoi, VoronoiCell, VoronoiFace, VoronoiIntegrator};
