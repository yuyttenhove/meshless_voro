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
//! You can select from five backends for arbitrary precision integer
//! arithmetic. These all provide identical functionality and vary only in
//! performance and licensing.
//!
//! For most practical applications, the choice of backend does not
//! significantly alter performance. However, for highly degenerate seed
//! configurations -- i.e. with many groups of more than four (almost)
//! co-spherical seed points -- many arbitrary precision arithmetic tests must
//! be performed leading to some performance differences in such cases.
//!
//! Benchmarks for construction of a Voronoi grid with 35³ seeds (single
//! threaded):
//!
//! |              | Perfect grid      | Perturbed grid     |
//! | ------------ | ----------------- | ------------------ |
//! | `rug`        | 1.129 s ± 0.011 s | 705.9 ms ± 9.7 ms  |
//! | `malachite`  | 1.477 s ± 0.070 s | 702.8 ms ± 9.0 ms  |
//! | `dashu`      | 1.731 s ± 0.037 s | 735.6 ms ± 10.3 ms |
//! | `num-bigint` | 2.249 s ± 0.125 s | 695.2 ms ± 6.8 ms  |
//!
//! See the next section for details.
//!
//! # Cargo Features
#![doc = document_features::document_features!()]

#[cfg(any(
    all(feature = "malachite", feature = "rug"),
    all(feature = "malachite", feature = "dashu"),
    all(feature = "malachite", feature = "num_bigint"),
    all(feature = "malachite", feature = "ibig"),
    all(feature = "malachite-base", feature = "rug"),
    all(feature = "malachite-base", feature = "dashu"),
    all(feature = "malachite-base", feature = "num_bigint"),
    all(feature = "malachite-base", feature = "ibig"),
    all(feature = "malachite-nz", feature = "rug"),
    all(feature = "malachite-nz", feature = "dashu"),
    all(feature = "malachite-nz", feature = "num_bigint"),
    all(feature = "malachite-nz", feature = "ibig"),
    all(feature = "rug", feature = "dashu"),
    all(feature = "rug", feature = "num_bigint"),
    all(feature = "rug", feature = "ibig"),
    all(feature = "dashu", feature = "num_bigint"),
    all(feature = "dashu", feature = "ibig"),
    all(feature = "num_bigint", feature = "ibig"),
))]
compile_error!("Multiple arbitrary precision arithmetic backends enabled!");

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
