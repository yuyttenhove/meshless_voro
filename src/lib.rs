//! An implementation of the [Meshless Voronoi algorithm](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) in rust.
//!
//! The algorithm is primarily aimed at generating 3D Voronoi diagrams, but can also be used to compute 1D and 2D Voronoi diagrams.
//! Like Voro++, this algorithm is _meshless_ implying that no global geometry is constructed. Instead a cell based approach is used and we only compute integrals (cell/face volumes and centroids) and connectivity information (it is possible to determine a cell's neighbours). 

mod part;
mod rtree_nn;
mod simple_cycle;
#[allow(dead_code)]  // Space is no longer used, I left it in as a reference for the gpu implementation
mod space;
mod util;
mod voronoi;

pub use voronoi::{Voronoi, VoronoiCell, VoronoiFace};
