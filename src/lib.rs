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
//! significantly alter performance (see results for a perturbed grid below).
//! However, for highly degenerate seed configurations -- i.e. with many groups
//! of more than four (almost) co-spherical seed points -- many arbitrary precision
//! arithmetic tests must be performed leading to some performance differences
//! in such cases (see results for a perfect grid below).
//!
//! Benchmarks for construction of a 3D Voronoi grid with 64³ seeds:
//!
//! |              | Perfect grid      | Perturbed grid    |
//! | ------------ | ----------------- | ----------------- |
//! | `rug`        | 2.062 s ± 0.005 s | 1.308 s ± 0.008 s |
//! | `malachite`  | 2.846 s ± 0.016 s | 1.293 s ± 0.005 s |
//! | `ibig`       | 3.105 s ± 0.048 s | 1.320 s ± 0.022 s |
//! | `dashu`      | 3.249 s ± 0.091 s | 1.313 s ± 0.009 s |
//! | `num-bigint` | 4.852 s ± 0.078 s | 1.301 s ± 0.004 s |
//!
//! See the next section for details.
//!
//! # Cargo Features
//!
//! **Note**: the features for choosing a backend are all *mutually exclusive*.
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

#[allow(dead_code)]
mod bounding_sphere;
pub mod geometry;
mod part;
mod rtree_nn;
mod simple_cycle;
// Space is no longer used, I left it in as a reference for the gpu
// implementation
#[allow(dead_code)]
mod space;
#[allow(dead_code)]
mod util;
mod voronoi;

pub use voronoi::{
    integrals, ConvexCell, Dimensionality, Voronoi, VoronoiCell, VoronoiFace, VoronoiIntegrator,
};
