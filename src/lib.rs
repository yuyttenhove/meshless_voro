//! **An implementation of the [Meshless Voronoi algorithm](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) in rust.**
//!
//! The algorithm is primarily aimed at generating 3D Voronoi diagrams, but can also be used to compute 1D and 2D Voronoi diagrams.
//! Like Voro++, this algorithm is _meshless_ implying that no global geometry is constructed. Instead a cell based approach is used and we only compute integrals (cell/face volumes and centroids) and connectivity information (it is possible to determine a cell's neighbours).
//!
//! The algorithm can generate Voronoi tesselations with a rectangular boundary or periodic boundary conditions and also supports computing a subset of the Voronoi tesselation.
//!
//! If necessary, arbitrary precision arithmetic is used to treat degeneracies and to ensure globaly consistent local geometry, see the appendix of [this reference](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) for more info:
//! > <cite>Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, Bruno Lévy. Meshless Voronoi on the GPU. ACM
//! > Transactions on Graphics, 2018, 37 (6), pp.1-12.  10.1145/3272127.3275092 .  hal-01927559<cite>
//!
//! **Features**:
//! - Construction of 1D, 2D and 3D Voronoi grids.
//! - Partial construction of grids.
//! - Parallel construction of the voronoi grid (requires `rayon` feature)
//! - Save Voronoi grids to `.hdf5` format (requires `hdf5` feature)
//! - Evaluation of custom _integrals_ for cells (e.g. weighted centroid) and faces (e.g. solid angles).

mod bounding_sphere;
pub mod geometry;
mod part;
mod rtree_nn;
mod simple_cycle;
// Space is no longer used, I left it in as a reference for the gpu implementation
#[allow(dead_code)]
mod space;
mod util;
mod voronoi;

pub use voronoi::{ConvexCell, Voronoi, VoronoiCell, VoronoiFace, VoronoiIntegrator};
pub use voronoi::integrals;

// pub use voronoi::integrals::{AreaIntegral, CellIntegral, FaceIntegral, VolumeIntegral};
