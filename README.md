# meshless_voronoi
An implementation of the [Meshless Voronoi algorithm](https://hal.inria.fr/hal-01927559/file/voroGPU.pdf) in rust.

The algorithm is primarily aimed at generating 3D Voronoi diagrams, but can also be used to compute 1D and 2D Voronoi diagrams.
Like Voro++, this algorithm is _meshless_ implying that no globally consistent geometry is constructed. Instead a cell based approach is used and we only compute integrals (cell/face volumes and centroids) and connectivity information (it is possible to determine a cell's neighbours). 

The algorithm can generate Voronoi tesselations with a rectangular boundary or periodic boundary conditions and also supports computing a subset of the Voronoi tesselation.