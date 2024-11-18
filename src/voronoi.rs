use self::integrals::{CellIntegral, CellIntegralWithData, FaceIntegral, FaceIntegralWithData};
use crate::rtree_nn::{build_rtree, nn_iter, wrapping_nn_iter};

use convex_cell::{ConvexCellMarker, WithFaces, WithoutFaces};
use glam::DVec3;
use integrals::FaceIntegrator;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "hdf5")]
use std::error::Error;
#[cfg(feature = "hdf5")]
use std::path::Path;

use boundary::SimulationBoundary;
pub use convex_cell::ConvexCell;
pub use generator::Generator;
pub use voronoi_cell::VoronoiCell;
pub use voronoi_face::VoronoiFace;

pub(crate) mod boundary;
pub mod convex_cell;
#[allow(unused)]
mod convex_cell_alternative;
mod generator;
pub mod half_space;
pub mod integrals;
mod voronoi_cell;
mod voronoi_face;

/// The dimensionality of the Voronoi tessellation.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, num_enum::IntoPrimitive, num_enum::TryFromPrimitive,
)]
#[repr(usize)]
pub enum Dimensionality {
    OneD = 1,
    TwoD = 2,
    #[default]
    ThreeD = 3,
}

impl Dimensionality {
    pub fn vector_is_valid(&self, v: DVec3) -> bool {
        match self {
            Self::OneD => v.y == 0. && v.z == 0.,
            Self::TwoD => v.z == 0.,
            Self::ThreeD => true,
        }
    }
}

/// Unravel a contiguous `array` of consecutive chunks of `n` elements.
#[allow(unused)]
macro_rules! unravel {
    ($array:expr, $n:expr) => {
        (0..$n)
            .map(|i| $array.iter().take(i).step_by($n).map(|v| *v).collect())
            .collect()
    };
}

macro_rules! flatten {
    ($array:expr) => {
        $array.into_iter().flatten().collect::<Vec<_>>()
    };
}

#[allow(unused_macros)]
macro_rules! cells_map {
    ($cells:expr, $mappable:expr) => {
        $cells
            .iter()
            .filter_map(|maybe_cell| maybe_cell.as_ref().map($mappable))
            .collect()
    };
}

#[allow(unused_macros)]
macro_rules! cells_map_par {
    ($cells:expr, $mappable:expr) => {
        $cells
            .par_iter()
            .filter_map(|maybe_cell| maybe_cell.as_ref().map($mappable))
            .collect()
    };
}

#[allow(unused_macros)]
macro_rules! cells_map_flatten {
    ($cells:expr, $mappable:expr) => {
        $cells
            .iter()
            .filter_map(|maybe_cell| maybe_cell.as_ref().map($mappable))
            .flatten()
            .collect()
    };
}

#[allow(unused_macros)]
macro_rules! cells_map_flatten_par {
    ($cells:expr, $mappable:expr) => {
        $cells
            .par_iter()
            .filter_map(|maybe_cell| maybe_cell.as_ref().map($mappable))
            .flatten()
            .collect()
    };
}

/// The main Voronoi struct.
///
/// This representation aims to be efficient and compact.
/// It has pre-calculated faces (storing only area, normal and centroid) and
/// cells (storing only volume, centroid and face-links).
///
/// If more flexibility is needed, the [`VoronoiIntegrator`] should be used instead.
#[derive(Clone)]
pub struct Voronoi {
    anchor: DVec3,
    width: DVec3,
    voronoi_cells: Vec<VoronoiCell>,
    faces: Vec<VoronoiFace>,
    cell_face_connections: Vec<usize>,
    dimensionality: Dimensionality,
    periodic: bool,
}

impl Voronoi {
    /// Construct the Voronoi tessellation. This method runs in parallel if the
    /// `rayon` feature is enabled.
    ///
    /// Iteratively construct each Voronoi cell independently of each other by
    /// repeatedly clipping it by the nearest generators until a safety
    /// criterion is reached. For non-periodic Voronoi tessellations, all
    /// Voronoi cells are clipped by the simulation volume with given `anchor`
    /// and `width` if necessary.
    ///
    /// * `generators` -- The seed points of the Voronoi cells.
    ///
    /// * `anchor` -- The lower left corner of the simulation volume.
    ///
    /// * `width` -- The width of the simulation volume. Also determines the
    ///   period of periodic Voronoi tessellations.
    ///
    /// * `dimensionality` -- The dimensionality of the Voronoi tessellation.
    ///   The algorithm is mainly aimed at constructing 3D Voronoi
    ///   tessellations, but can be used for 1D or 2D as well.
    ///
    /// * `periodic` -- Whether to apply periodic boundary conditions to the
    ///   Voronoi tessellation.
    pub fn build(
        generators: &[DVec3],
        anchor: DVec3,
        width: DVec3,
        dimensionality: Dimensionality,
        periodic: bool,
    ) -> Self {
        Self::build_internal(generators, None, anchor, width, dimensionality, periodic)
    }

    /// Same as `build`, but now, only a subset of the Voronoi cells is fully
    /// constructed. The other Voronoi cells will have zero volume and
    /// centroid, but still might have some faces linked to them
    /// (between them and other Voronoi cells that _are_ fully constructed).
    ///
    /// * `generators` -- The seed points of the Voronoi cells.
    ///
    /// * `mask` -- `true` for Voronoi cells which have to be fully constructed.
    ///
    /// * `anchor` -- The lower left corner of the simulation volume.
    ///
    /// * `width` -- The width of the simulation volume. Also determines the
    ///   period of periodic Voronoi tessellations.
    ///
    /// * `dimensionality` -- The dimensionality of the Voronoi tessellation.
    ///   The algorithm is mainly aimed at constructing 3D Voronoi
    ///   tessellations, but can be used for 1D or 2D as well.
    ///
    /// * `periodic` 0- Whether to apply periodic boundary conditions to the
    ///   Voronoi tessellation.
    pub fn build_partial(
        generators: &[DVec3],
        mask: &[bool],
        anchor: DVec3,
        width: DVec3,
        dimensionality: Dimensionality,
        periodic: bool,
    ) -> Self {
        Self::build_internal(generators, Some(mask), anchor, width, dimensionality, periodic)
    }

    fn build_internal(
        generators: &[DVec3],
        mask: Option<&[bool]>,
        mut anchor: DVec3,
        mut width: DVec3,
        dimensionality: Dimensionality,
        periodic: bool,
    ) -> Self {
        // Normalize the unused components of the simulation volume, so that the lower
        // dimensional volumes will be correct.
        if let Dimensionality::OneD = dimensionality {
            anchor.y = -0.5;
            width.y = 1.;
        }

        if let Dimensionality::OneD | Dimensionality::TwoD = dimensionality {
            anchor.z = -0.5;
            width.z = 1.;
        }

        // build cells
        let n_cells = generators.len();
        let mut faces = vec![vec![]; n_cells];
        let voronoi_cells = Self::build_voronoi_cells(
            generators,
            &mut faces,
            mask,
            anchor,
            width,
            dimensionality,
            periodic,
        );

        // flatten faces
        let faces = flatten!(faces);

        Voronoi {
            anchor,
            width,
            voronoi_cells,
            faces,
            cell_face_connections: vec![],
            dimensionality,
            periodic,
        }
        .finalize()
    }

    /// Link the Voronoi faces to their respective cells.
    fn finalize(mut self) -> Self {
        let mut cell_face_connections: Vec<Vec<usize>> =
            (0..self.voronoi_cells.len()).map(|_| vec![]).collect();

        for (i, face) in self.faces.iter().enumerate() {
            cell_face_connections[face.left()].push(i);
            // link faces to their right generator if necessary (no boundary faces).
            if let (Some(right_idx), None) = (face.right(), face.shift()) {
                cell_face_connections[right_idx].push(i);
            }
        }

        let mut face_connections_offset = 0;
        for (i, cell) in self.voronoi_cells.iter_mut().enumerate() {
            let face_count = cell_face_connections[i].len();
            cell.finalize(face_connections_offset, face_count);
            face_connections_offset += face_count;
        }

        self.cell_face_connections = flatten!(cell_face_connections);

        self
    }

    fn build_voronoi_cells(
        generators: &[DVec3],
        faces: &mut [Vec<VoronoiFace>],
        mask: Option<&[bool]>,
        anchor: DVec3,
        width: DVec3,
        dimensionality: Dimensionality,
        periodic: bool,
    ) -> Vec<VoronoiCell> {
        // Some general properties
        let generators: Vec<Generator> = generators
            .iter()
            .enumerate()
            .map(|(id, &loc)| Generator::new(id, loc, dimensionality))
            .collect();

        let rtree = build_rtree(&generators);
        let simulation_volume = SimulationBoundary::cuboid(anchor, width, periodic, dimensionality);

        // Helper function to build a single cell
        let build = |(idx, faces)| {
            if mask.map_or(true, |mask| mask[idx]) {
                let generator: &Generator = &generators[idx];
                let loc = generator.loc();
                debug_assert_eq!(generator.id(), idx);
                let nearest_neighbours = if periodic {
                    wrapping_nn_iter(&rtree, loc, width, dimensionality)
                } else {
                    nn_iter(&rtree, loc)
                };
                let convex_cell = ConvexCell::build(
                    loc,
                    idx,
                    &generators,
                    nearest_neighbours,
                    &simulation_volume,
                );
                VoronoiCell::from_convex_cell(&convex_cell, faces, mask)
            } else {
                VoronoiCell::default()
            }
        };

        #[cfg(feature = "rayon")]
        let voronoi_cells = faces.par_iter_mut().enumerate().map(build).collect::<Vec<_>>();

        #[cfg(not(feature = "rayon"))]
        let voronoi_cells = faces.iter_mut().enumerate().map(build).collect::<Vec<_>>();

        voronoi_cells
    }

    /// The anchor of the simulation volume.
    ///
    /// All generators are assumed to be contained in this simulation volume.
    pub fn anchor(&self) -> DVec3 {
        self.anchor
    }

    /// The width of the simulation volume.
    ///
    /// All generators are assumed to be contained in this simulation volume.
    pub fn width(&self) -> DVec3 {
        self.width
    }

    /// Get the Voronoi cells.
    pub fn cells(&self) -> &[VoronoiCell] {
        self.voronoi_cells.as_ref()
    }

    /// Get the Voronoi faces.
    pub fn faces(&self) -> &[VoronoiFace] {
        self.faces.as_ref()
    }

    /// Get a vector of the Voronoi faces by consuming the Voronoi struct.
    pub fn into_faces(self) -> Vec<VoronoiFace> {
        self.faces
    }

    /// Get the links between the cells and their faces.
    pub fn cell_face_connections(&self) -> &[usize] {
        self.cell_face_connections.as_ref()
    }

    /// Get the dimensionality used to compute this Voronoi tessellation.
    pub fn dimensionality(&self) -> usize {
        self.dimensionality.into()
    }

    /// Whether this Voronoi tessellation is periodic or not.
    pub fn periodic(&self) -> bool {
        self.periodic
    }

    /// For all cells, check that the area of the faces is larger than the area
    /// of a sphere with the same volume
    #[cfg(test)]
    fn consistency_check(&self) {
        use float_cmp::assert_approx_eq;

        let mut total_volume = 0.;
        let mut all_active = true;
        for cell in self.cells() {
            if cell.volume() == 0. {
                all_active = false;
                continue;
            }
            total_volume += cell.volume();
            let faces = cell.faces(self);
            let mut area = 0.;
            for f in faces {
                assert!(self.dimensionality.vector_is_valid(f.normal()));
                area += f.area();
            }
            match self.dimensionality {
                Dimensionality::ThreeD => {
                    let radius =
                        (0.25 * 3. * std::f64::consts::FRAC_1_PI * cell.volume()).powf(1. / 3.);
                    let sphere_area = 4. * std::f64::consts::PI * radius * radius;
                    assert!(area > sphere_area);
                },
                Dimensionality::TwoD =>  {
                    let radius = (std::f64::consts::FRAC_1_PI * cell.volume()).sqrt();
                    let circumference = 2. * std::f64::consts::PI * radius;
                    assert!(area > circumference);
                }
                Dimensionality::OneD => {
                    assert_approx_eq!(f64, area, 2.);
                }
            }
        }
        if all_active && self.dimensionality == Dimensionality::ThreeD {
            let box_volume = self.width.x * self.width.y * self.width.z;
            assert_approx_eq!(
                f64,
                total_volume,
                box_volume,
                epsilon = box_volume * 1e-11,
                ulps = 4
            );
        }
    }

    /// Save the Voronoi tessellation to a HDF5 file.
    /// Requires the `hdf5` feature to be enabled!
    /// 
    /// * `filename` - Filename to write to. Contents will be overwritten!
    /// 
    /// This creates two groups in the `.hdf5` file called `Cells` and `Faces`, 
    /// in addition to one top level dataset `CellFaceConnections`.
    /// 
    /// `Cells` stores the following datasets:
    /// * `Volume` - The volumes of the cells
    /// * `Centroid` - The centroid of the cells
    /// * `FaceCount` - The number of faces of each cell
    /// * `FaceConnectionsOffset` - The offset of a given cell's face indices in the 
    ///                             `CellFaceConnections` dataset.
    /// * `Generator` - The position of the generator of each cell.
    /// 
    /// `Faces` stores the following datasets:
    /// * `Area` - The area of the faces
    /// * `Centroid` - The centroid of the faces
    /// * `Normal` - The normal vectors of the faces.
    /// 
    /// `CellFaceConnections` contains the concatenated indices of the faces of each cell.
    #[cfg(feature = "hdf5")]
    pub fn write_to_hdf5<P: AsRef<Path>>(&self, filename: P) -> Result<(), Box<dyn Error>> {
        // Create the file to write the data to
        let file = hdf5::File::create(filename)?;

        // Write cell info
        let group = file.create_group("Cells")?;
        let data = self.voronoi_cells.iter().map(|c| c.volume()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Volume")?;
        let data = self
            .voronoi_cells
            .iter()
            .map(|c| c.face_connections_offset())
            .collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("FaceConnectionsOffset")?;
        let data = self.voronoi_cells.iter().map(|c| c.face_count()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("FaceCount")?;
        let data = self.voronoi_cells.iter().map(|c| c.centroid().to_array()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Centroid")?;
        let data = self.voronoi_cells.iter().map(|c| c.loc().to_array()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Generator")?;

        // Write face info
        let group = file.create_group("Faces")?;
        let data = self.faces.iter().map(|f| f.area()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Area")?;
        let data = self.faces.iter().map(|f| f.centroid().to_array()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Centroid")?;
        let data = self.faces.iter().map(|f| f.normal().to_array()).collect::<Vec<_>>();
        group.new_dataset_builder().with_data(&data).create("Normal")?;
        if let Dimensionality::TwoD = self.dimensionality {
            // Also write face start and end points
            let face_directions = self
                .faces
                .iter()
                .map(|f| f.area() * f.normal().cross(DVec3::Z))
                .collect::<Vec<_>>();
            let face_start = self
                .faces
                .iter()
                .zip(face_directions.iter())
                .map(|(f, &d)| (f.centroid() - 0.5 * d).to_array())
                .collect::<Vec<_>>();
            let face_end = self
                .faces
                .iter()
                .zip(face_directions.iter())
                .map(|(f, &d)| (f.centroid() + 0.5 * d).to_array())
                .collect::<Vec<_>>();
            group.new_dataset_builder().with_data(&face_start).create("Start")?;
            group.new_dataset_builder().with_data(&face_end).create("End")?;
        }

        // Write cell face connections
        file.new_dataset_builder()
            .with_data(self.cell_face_connections())
            .create("CellFaceConnections")?;

        Ok(())
    }
}

impl<M: ConvexCellMarker + 'static> From<&VoronoiIntegrator<M>> for Voronoi {
    fn from(voronoi_integrator: &VoronoiIntegrator<M>) -> Self {
        let mut faces = vec![vec![]; voronoi_integrator.cells.len()];
        let voronoi_cells = voronoi_integrator.build_voronoi_cells(&mut faces);

        // flatten faces
        let faces = flatten!(faces);

        Voronoi {
            anchor: voronoi_integrator.anchor,
            width: voronoi_integrator.width,
            voronoi_cells,
            faces,
            cell_face_connections: vec![],
            dimensionality: voronoi_integrator.dimensionality,
            periodic: voronoi_integrator.periodic,
        }
        .finalize()
    }
}

/// An meshless representation of the Voronoi mesh as a collection of independent [`ConvexCell`]s.
///
/// This representation is less efficient than the main [`Voronoi`] struct, but is more flexible.
/// It can be used to:
/// * Compute (custom) integrals over the faces (e.g. centroid, area, solid angle...)
/// * Compute (custom) integrals over the cells (e.g. centroid, volume...)
/// * Retrieve the vertices of the faces
///
/// It can also be converted into a [`Voronoi`] struct by using the `From<&VoronoiIntegrator>` trait
/// implemented on [`Voronoi`], but note that direct construction of the [`Voronoi`] mesh will be
/// more efficient.
#[derive(Clone)]
pub struct VoronoiIntegrator<Marker: ConvexCellMarker> {
    cells: Vec<Option<ConvexCell<Marker>>>,
    cell_is_active: Vec<bool>,
    anchor: DVec3,
    width: DVec3,
    dimensionality: Dimensionality,
    periodic: bool,
}

impl VoronoiIntegrator<WithoutFaces> {
    /// Build the [`ConvexCell`]s for the given generators.
    ///
    /// * `generators` - The seed points of the Voronoi cells.
    /// * `mask` - (Optional) `True` for generators whose corresponding cells have to be
    ///   constructed.
    /// * `anchor` - The lower left corner of the simulation volume.
    /// * `width` - The width of the simulation volume. Also determines the
    ///   periodicity distance of periodic Voronoi tessellations.
    /// * `dimensionality` - The dimensionality of the Voronoi tessellation. The
    ///   algorithm is mainly aimed at constructing 3D Voronoi tessellations,
    ///   but can be used for 1 or 2D as well.
    /// * `periodic` - Whether to apply periodic boundary conditions.
    pub fn build(
        generators: &[DVec3],
        mask: Option<&[bool]>,
        mut anchor: DVec3,
        mut width: DVec3,
        dimensionality: Dimensionality,
        periodic: bool,
    ) -> Self {
        // Normalize the unused components of the simulation volume, so that the lower
        // dimensional volumes will be correct.
        if let Dimensionality::OneD = dimensionality {
            anchor.y = -0.5;
            width.y = 1.;
        };

        if let Dimensionality::OneD | Dimensionality::TwoD = dimensionality {
            anchor.z = -0.5;
            width.z = 1.;
        }

        let cell_is_active = mask.map_or(vec![true; generators.len()], |mask| mask.to_vec());

        // Construct generators
        let generators: Vec<Generator> = generators
            .iter()
            .enumerate()
            .map(|(id, &loc)| Generator::new(id, loc, dimensionality))
            .collect();

        let rtree = build_rtree(&generators);
        let simulation_volume = SimulationBoundary::cuboid(anchor, width, periodic, dimensionality);

        // Helper function
        let build = |(idx, generator): (usize, &Generator)| {
            if cell_is_active[idx] {
                let loc = generator.loc();
                debug_assert_eq!(generator.id(), idx);
                let nearest_neighbours = if periodic {
                    wrapping_nn_iter(&rtree, loc, width, dimensionality)
                } else {
                    nn_iter(&rtree, loc)
                };
                let convex_cell = ConvexCell::build(
                    loc,
                    idx,
                    &generators,
                    nearest_neighbours,
                    &simulation_volume,
                );
                Some(convex_cell)
            } else {
                None
            }
        };

        #[cfg(feature = "rayon")]
        let cells = generators.par_iter().enumerate().map(build).collect::<Vec<_>>();

        #[cfg(not(feature = "rayon"))]
        let cells = generators.iter().enumerate().map(build).collect::<Vec<_>>();

        Self {
            cells,
            cell_is_active,
            anchor,
            width,
            dimensionality,
            periodic,
        }
    }

    /// Convert this [`VoronoiIntegrator`]'s [`ConvexCell`]s into ones with face information stored.
    ///
    /// NOTE: this is only supported for 3D Voronoi meshes, since the underlying
    /// representation is always 3D, this method would lead to additional, nonsensical vertices
    /// along the extra dimensions in 1D or 2D. The function panics when invoked in 1D or 2D.
    pub fn with_faces(self) -> VoronoiIntegrator<WithFaces> {
        #[cfg(feature = "rayon")]
        let cells_with_faces = self
            .cells
            .into_par_iter()
            .map(|cell| cell.map(|cell| cell.with_faces()))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let cells_with_faces =
            self.cells.into_iter().map(|cell| cell.map(|cell| cell.with_faces())).collect();
        VoronoiIntegrator::<WithFaces> {
            cells: cells_with_faces,
            cell_is_active: self.cell_is_active,
            anchor: self.anchor,
            width: self.width,
            dimensionality: self.dimensionality,
            periodic: self.periodic,
        }
    }
}

impl<M: ConvexCellMarker + 'static> VoronoiIntegrator<M> {
    /// Get the [`ConvexCell`] at the specified index, if any.
    ///
    /// When using partial construction, no [`ConvexCell`] is constructed for some generators,
    /// in which case this function will return None when retrieving unconstructed cells.
    ///
    /// * `index` - The index of the generator whose corresponding [`ConvexCell`] to retrieve.
    pub fn get_cell_at(&self, index: usize) -> Option<&ConvexCell<M>> {
        self.cells[index].as_ref()
    }

    /// Get an iterator over all the [`ConvexCell`]s.
    ///
    /// This iterator is already filtered in the case of partial construction.
    /// The index of the generator corresponding with a [`ConvexCell`] from the iterator
    /// must be retrieved from its `idx` field.
    pub fn cells_iter(&self) -> impl Iterator<Item = &ConvexCell<M>> {
        self.cells.iter().filter_map(|cell| cell.as_ref())
    }

    /// Compute a custom cell integral for the active cells in this
    /// representation.
    pub fn compute_cell_integrals<I: CellIntegral>(&self) -> Vec<I> {
        #[cfg(feature = "rayon")]
        return cells_map_par!(self.cells, |cell| cell.compute_cell_integral::<(), I>(()));
        #[cfg(not(feature = "rayon"))]
        cells_map!(self.cells, |cell| cell.compute_cell_integral(()))
    }

    /// Compute a custom cell integral for the active cells in this
    /// representation, for integrals that need some external data.
    ///
    /// * `extra_data` - The extra data used by the cell integral, for each cell.
    pub fn compute_cell_integrals_with_data<D: Copy + Sync, I: CellIntegralWithData<Data = D>>(
        &self,
        extra_data: &[D],
    ) -> Vec<I> {
        #[cfg(feature = "rayon")]
        return self
            .cells
            .par_iter()
            .zip(extra_data.par_iter())
            .filter_map(|(cell, data)| cell.as_ref().map(|cell| cell.compute_cell_integral(*data)))
            .collect();
        #[cfg(not(feature = "rayon"))]
        self.cells
            .iter()
            .zip(extra_data.iter())
            .filter_map(|(cell, data)| cell.as_ref().map(|cell| cell.compute_cell_integral(*data)))
            .collect()
    }

    /// Compute a custom face integral for all the faces of each active cell in
    /// this representation.
    ///
    /// Assumes non-symmetric integral (i.e. The integral of a face evaluated
    /// from the left cell might be different than the integral evaluated
    /// from the right cell). Therefore all faces are treated twice.
    /// For symmetric integrals, `compute_face_integrals_sym` will be more
    /// efficient.
    ///
    /// Returns a flat vector with of [`FaceIntegrator`]s storing the necessary info to link the faces
    /// to their respective cells.
    pub fn compute_face_integrals<I: FaceIntegral>(&self) -> Vec<FaceIntegrator<I>> {
        #[cfg(feature = "rayon")]
        return cells_map_flatten_par!(self.cells, |cell| cell.compute_face_integrals(()));
        #[cfg(not(feature = "rayon"))]
        cells_map_flatten!(self.cells, |cell| cell.compute_face_integrals(()))
    }

    /// Compute a custom face integral for all the faces of each active cell in
    /// this representation.
    ///
    /// Assumes symmetric integral (i.e. The integral of a face evaluated
    /// from the left cell will be the same as the integral evaluated
    /// from the right cell).
    ///
    /// Returns a flat vector with of [`FaceIntegrator`]s storing the necessary info to link the faces
    /// to their respective cells.
    pub fn compute_face_integrals_sym<I: FaceIntegral>(&self) -> Vec<FaceIntegrator<I>> {
        #[cfg(feature = "rayon")]
        return cells_map_flatten_par!(self.cells, |cell| cell
            .compute_face_integrals_sym((), &self.cell_is_active));
        #[cfg(not(feature = "rayon"))]
        cells_map_flatten!(self.cells, |cell| cell
            .compute_face_integrals_sym((), &self.cell_is_active))
    }

    /// Compute a custom face integral, with some extra associated data,
    /// for all the faces of each active cell in this representation.
    /// The extra data is cloned between faces.
    ///
    /// Assumes non-symmetric integral (i.e. The integral of a face evaluated
    /// from the left cell might be different than the integral evaluated
    /// from the right cell). Therefore all faces are treated twice.
    /// For symmetric integrals, `compute_face_integrals_sym` will be more
    /// efficient.
    ///
    /// * `extra_data` - The extra data used by the cell integral, for each cell.
    ///
    /// Returns a flat vector with of [`FaceIntegrator`]s storing the necessary info to link the faces
    /// to their respective cells.
    pub fn compute_face_integrals_with_data<D: Copy + Sync, I: FaceIntegralWithData<Data = D>>(
        &self,
        extra_data: &[D],
    ) -> Vec<FaceIntegrator<I>> {
        #[cfg(feature = "rayon")]
        return self
            .cells
            .par_iter()
            .zip(extra_data.par_iter())
            .filter_map(|(cell, data)| cell.as_ref().map(|cell| cell.compute_face_integrals(*data)))
            .flatten()
            .collect();
        #[cfg(not(feature = "rayon"))]
        self.cells
            .iter()
            .zip(extra_data.iter())
            .filter_map(|(cell, data)| cell.as_ref().map(|cell| cell.compute_face_integrals(*data)))
            .flatten()
            .collect()
    }

    /// Compute a custom face integral, with some extra associated data,
    /// for all the faces of each active cell in this representation.
    /// The extra data is cloned between faces.
    ///
    /// Assumes symmetric integral (i.e. The integral of a face evaluated
    /// from the left cell will be the same as the integral evaluated
    /// from the right cell).
    ///
    /// * `extra_data` - The extra data used by the cell integral, for each cell.
    ///
    /// Returns a flat vector with of [`FaceIntegrator`]s storing the necessary info to link the faces
    /// to their respective cells.
    pub fn compute_face_integrals_sym_with_data<
        D: Copy + Sync,
        I: FaceIntegralWithData<Data = D>,
    >(
        &self,
        extra_data: &[D],
    ) -> Vec<FaceIntegrator<I>> {
        #[cfg(feature = "rayon")]
        return self
            .cells
            .par_iter()
            .zip(extra_data.par_iter())
            .filter_map(|(cell, data)| {
                cell.as_ref()
                    .map(|cell| cell.compute_face_integrals_sym(*data, &self.cell_is_active))
            })
            .flatten()
            .collect();
        #[cfg(not(feature = "rayon"))]
        self.cells
            .iter()
            .zip(extra_data.iter())
            .filter_map(|(cell, data)| {
                cell.as_ref()
                    .map(|cell| cell.compute_face_integrals_sym(*data, &self.cell_is_active))
            })
            .flatten()
            .collect()
    }

    /// Compute the more compact [`VoronoiCell`]s from the [`ConvexCell`]s of this [`VoronoiIntegrator`].
    /// This also computes the volumes of the cells and the areas of the faces.
    ///
    /// * `faces` - Mutable slice of vectors to store the faces of each cell (one vector per cell).
    pub fn build_voronoi_cells(&self, faces: &mut [Vec<VoronoiFace>]) -> Vec<VoronoiCell> {
        let build = |(convex_cell, faces): (&Option<ConvexCell<_>>, _)| match convex_cell {
            Some(convex_cell) => {
                VoronoiCell::from_convex_cell(convex_cell, faces, Some(&self.cell_is_active))
            }
            None => VoronoiCell::default(),
        };
        #[cfg(feature = "rayon")]
        let voronoi_cells = self.cells.par_iter().zip(faces.par_iter_mut()).map(build).collect();
        #[cfg(not(feature = "rayon"))]
        let voronoi_cells = self.cells.iter().zip(faces.iter_mut()).map(build).collect();

        voronoi_cells
    }
}

#[cfg(test)]
mod tests {
    use super::{
        integrals::{AreaCentroidIntegral, VolumeCentroidIntegral},
        *,
    };
    use float_cmp::assert_approx_eq;
    use rand::{distributions::Uniform, prelude::*};

    fn perturbed_grid(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
        let mut generators = vec![];
        let mut rng = thread_rng();
        let distr = Uniform::new(-0.5, 0.5);
        for n in 0..count.pow(3) {
            let i = n / count.pow(2);
            let j = (n % count.pow(2)) / count;
            let k = n % count;
            let pos = DVec3 {
                x: i as f64 + 0.5 + pert * rng.sample(distr),
                y: j as f64 + 0.5 + pert * rng.sample(distr),
                z: k as f64 + 0.5 + pert * rng.sample(distr),
            } * width
                / count as f64
                + anchor;
            generators.push(pos.clamp(anchor, anchor + width));
        }

        generators
    }

    fn perturbed_grid_vector_pert(anchor: DVec3, width: DVec3, count: usize, pert: DVec3) -> Vec<DVec3> {
        let mut generators = vec![];
        let mut rng = thread_rng();
        let distr = Uniform::new(-0.5, 0.5);
        for n in 0..count.pow(3) {
            let i = n / count.pow(2);
            let j = (n % count.pow(2)) / count;
            let k = n % count;
            let pos = DVec3 {
                x: i as f64 + 0.5 + pert.x * rng.sample(distr),
                y: j as f64 + 0.5 + pert.y * rng.sample(distr),
                z: k as f64 + 0.5 + pert.z * rng.sample(distr),
            } * width
                / count as f64
                + anchor;
            generators.push(pos.clamp(anchor, anchor + width));
        }

        generators
    }

    fn perturbed_plane(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
        let mut generators = vec![];
        let mut rng = thread_rng();
        let distr = Uniform::new(-0.5, 0.5);
        for n in 0..count.pow(2) {
            let i = n / count;
            let j = n % count;
            let pos = DVec3 {
                x: i as f64 + 0.5 + pert * rng.sample(distr),
                y: j as f64 + 0.5 + pert * rng.sample(distr),
                z: 0.5 * count as f64,
            } * width
                / count as f64
                + anchor;
            generators.push(pos.clamp(anchor, anchor + width));
        }

        generators
    }

    #[test]
    fn test_single_cell() {
        let generators = vec![DVec3::splat(0.5)];
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        assert_approx_eq!(f64, voronoi.voronoi_cells[0].volume(), 1.);
    }

    #[test]
    fn test_two_cells() {
        let generators = vec![
            DVec3 {
                x: 0.3,
                y: 0.4,
                z: 0.25,
            },
            DVec3 {
                x: 0.7,
                y: 0.6,
                z: 0.75,
            },
        ];
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        assert_approx_eq!(f64, voronoi.voronoi_cells[0].volume(), 0.5);
        assert_approx_eq!(f64, voronoi.voronoi_cells[1].volume(), 0.5);
    }

    #[test]
    fn test_4_cells() {
        let generators = vec![
            DVec3 {
                x: 0.4,
                y: 0.3,
                z: 0.,
            },
            DVec3 {
                x: 1.6,
                y: 0.2,
                z: 0.,
            },
            DVec3 {
                x: 0.6,
                y: 0.8,
                z: 0.,
            },
            DVec3 {
                x: 1.4,
                y: 0.7,
                z: 0.,
            },
        ];
        let anchor = DVec3::ZERO;
        let width = DVec3 {
            x: 2.,
            y: 1.,
            z: 1.,
        };
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::TwoD, true);
        #[cfg(feature = "hdf5")]
        voronoi.write_to_hdf5("test_4_cells.hdf5").unwrap();
        voronoi.consistency_check();
    }

    #[test]
    fn test_five_cells() {
        let delta = 0.1f64.sqrt();
        let generators = vec![
            DVec3 {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            },
            DVec3 {
                x: 0.5 - delta,
                y: 0.5 - delta,
                z: 0.5,
            },
            DVec3 {
                x: 0.5 - delta,
                y: 0.5 + delta,
                z: 0.5,
            },
            DVec3 {
                x: 0.5 + delta,
                y: 0.5 + delta,
                z: 0.5,
            },
            DVec3 {
                x: 0.5 + delta,
                y: 0.5 - delta,
                z: 0.5,
            },
        ];
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::TwoD, false);
        assert_approx_eq!(f64, voronoi.voronoi_cells[0].volume(), 0.2);
        assert_approx_eq!(f64, voronoi.voronoi_cells[1].volume(), 0.2);
        assert_approx_eq!(f64, voronoi.voronoi_cells[2].volume(), 0.2);
        assert_approx_eq!(f64, voronoi.voronoi_cells[3].volume(), 0.2);
        assert_approx_eq!(f64, voronoi.voronoi_cells[4].volume(), 0.2);
    }

    #[test]
    fn test_eight_cells() {
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let generators = perturbed_grid(anchor, width, 2, 0.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        for cell in &voronoi.voronoi_cells {
            assert_approx_eq!(f64, cell.volume(), 0.125);
        }
    }

    #[test]
    fn test_27_cells() {
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let generators = perturbed_grid(anchor, width, 3, 0.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        for cell in &voronoi.voronoi_cells {
            assert_approx_eq!(f64, cell.volume(), 1. / 27.);
        }
    }

    #[test]
    fn test_64_cells() {
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let generators = perturbed_grid(anchor, width, 4, 0.);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        for cell in &voronoi.voronoi_cells {
            assert_approx_eq!(f64, cell.volume(), 1. / 64.);
        }
    }

    #[test]
    fn test_125_cells() {
        let pert = 0.5;
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let generators = perturbed_grid(anchor, width, 5, pert);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        voronoi.consistency_check();
    }

    #[test]
    fn test_partial() {
        let pert = 0.9;
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(1.);
        let generators = perturbed_grid(anchor, width, 3, pert);
        let voronoi_all = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        for i in 0..27 {
            let mut mask = vec![false; 27];
            mask[i] = true;
            let voronoi_partial = Voronoi::build_partial(
                &generators,
                &mask,
                anchor,
                width,
                Dimensionality::ThreeD,
                false,
            );
            for j in 0..27 {
                if j == i {
                    assert_approx_eq!(
                        f64,
                        voronoi_all.voronoi_cells[j].volume(),
                        voronoi_partial.voronoi_cells[j].volume()
                    );
                    assert_eq!(
                        voronoi_all.voronoi_cells[j].face_count(),
                        voronoi_partial.voronoi_cells[j].face_count()
                    );
                } else {
                    assert_eq!(voronoi_partial.voronoi_cells[j].volume(), 0.);
                }
            }
        }
    }

    #[test]
    fn test_2_d() {
        let pert = 0.95;
        let count = 25;
        let anchor = DVec3::splat(2.);
        let width = DVec3 {
            x: 2.,
            y: 2.,
            z: 1.,
        };
        let generators = perturbed_plane(anchor, width, count, pert);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::TwoD, true);

        #[cfg(feature = "hdf5")]
        voronoi.write_to_hdf5("test_2_d.hdf5").unwrap();

        assert_approx_eq!(
            f64,
            voronoi.voronoi_cells.iter().map(|c| c.volume()).sum(),
            4.,
            epsilon = 1e-10,
            ulps = 8
        );
    }

    #[test]
    fn test_3_d() {
        let pert = 0.95;
        let count = 15;
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(2.);
        let generators = perturbed_grid(anchor, width, count, pert);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        assert_eq!(voronoi.voronoi_cells.len(), generators.len());
        voronoi.consistency_check();
    }

    #[test]
    fn test_density_grad_2_d() {
        let pert = 1.;
        let counts = [10, 40, 20, 80];
        let anchor = DVec3::ZERO;
        let width = DVec3::ONE;
        let anchor_delta = DVec3 {
            x: 0.25,
            y: 0.,
            z: 0.,
        };
        let width_part = DVec3 {
            x: 0.25,
            y: 1.,
            z: 1.,
        };
        let mut plane = vec![];
        for (i, count) in counts.into_iter().enumerate() {
            plane.extend(perturbed_plane(
                anchor + i as f64 * anchor_delta,
                width_part,
                count,
                pert,
            ));
        }
        let voronoi = Voronoi::build(&plane, anchor, width, Dimensionality::TwoD, true);
        #[cfg(feature = "hdf5")]
        voronoi.write_to_hdf5("test_density_grad_2_d.hdf5").unwrap();

        assert_eq!(voronoi.voronoi_cells.len(), plane.len());
        voronoi.consistency_check();
    }

    #[test]
    fn degenerate_test() {
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(2e15);
        let mut generators = perturbed_grid(anchor, width, 10, 0.);
        generators[42] = 1e14 * DVec3::new(1.00007490802, 9.00019014286, 5.00014639879);
        let voronoi = Voronoi::build(&generators, anchor, width, Dimensionality::ThreeD, false);
        voronoi.consistency_check();
    }

    #[test]
    fn degenerate_test_simplified() {
        let anchor = DVec3::new(0., 6e14, 2e14);
        let width = DVec3::splat(6e14);
        let mut generators = perturbed_grid(anchor, width, 3, 0.);
        generators[4] = 1e14 * DVec3::new(1.00007490802, 9.00019014286, 5.00014639879);
        let mut mask = [false; 27];
        mask[3] = true;
        let voronoi = Voronoi::build_partial(
            &generators,
            &mask,
            anchor,
            width,
            Dimensionality::ThreeD,
            false,
        );
        voronoi.consistency_check();
    }

    #[test]
    fn test_integrator() {
        let pert = 0.95;
        let count = 5;
        let anchor = DVec3::ZERO;
        let width = DVec3::splat(2.);
        let generators = perturbed_grid(anchor, width, count, pert);

        let mut mask = vec![false; 125];
        mask[62] = true;
        let voronoi =
            Voronoi::build_partial(&generators, &mask, anchor, width, Dimensionality::ThreeD, true);
        let integrator = VoronoiIntegrator::build(
            &generators,
            Some(&mask),
            anchor,
            width,
            Dimensionality::ThreeD,
            true,
        );

        let area_centroids = integrator.compute_face_integrals::<AreaCentroidIntegral>();
        let volume_centroids = integrator.compute_cell_integrals::<VolumeCentroidIntegral>();

        assert_eq!(area_centroids.len(), voronoi.faces().len());
        for (
            i,
            FaceIntegrator {
                integral: AreaCentroidIntegral { area, centroid },
                ..
            },
        ) in area_centroids.iter().enumerate()
        {
            assert_eq!(*area, voronoi.faces()[i].area());
            assert_eq!(*centroid, voronoi.faces()[i].centroid());
        }
        assert_eq!(volume_centroids.len(), 1);
        assert_eq!(volume_centroids[0].volume, voronoi.cells()[62].volume());
        assert_eq!(volume_centroids[0].centroid, voronoi.cells()[62].centroid());

        let voronoi2: Voronoi = (&integrator).into();
        assert_eq!(voronoi2.anchor, voronoi.anchor);
        assert_eq!(voronoi2.width, voronoi.width);
        assert_eq!(voronoi2.periodic, voronoi.periodic);
        assert_eq!(voronoi2.cells().len(), voronoi.cells().len());
        assert_eq!(voronoi2.cells()[62].centroid(), voronoi.cells()[62].centroid());
        assert_eq!(voronoi2.cells()[62].volume(), voronoi.cells()[62].volume());
        assert_eq!(
            voronoi2.cells()[62].face_connections_offset(),
            voronoi.cells()[62].face_connections_offset()
        );
        assert_eq!(voronoi2.cells()[62].face_count(), voronoi.cells()[62].face_count());
        assert_eq!(voronoi2.faces().len(), voronoi.faces().len());
        for (face0, face1) in voronoi.faces().iter().zip(voronoi2.faces.iter()) {
            assert_eq!(face0.area(), face1.area());
            assert_eq!(face0.centroid(), face1.centroid());
            assert_eq!(face0.right(), face1.right());
            assert_eq!(face0.normal(), face1.normal());
        }
    }

    #[test]
    fn test_non_perturbed_z() {
        let anchor = DVec3::new(-14.17150624593099, 16.202089309692383, 26.540990829467773);
        let width = DVec3::new(32.670213063557945, 4.626067479451496, 0.2923425038655587);
        let points = vec![
            DVec3::new(17.159128189086914, 17.747779846191406, 26.790985743204754),
            DVec3::new(16.933035532633465, 17.705819447835285, 26.666666666666668),
            DVec3::new(17.00386683146159, 17.983611424763996, 26.833333333333332),
            DVec3::new(16.84653155008952, 18.130948384602863, 26.833333333333332),
            DVec3::new(16.59058952331543, 18.026987075805664, 26.666666666666668),
            DVec3::new(16.594066619873047, 18.36242739359538, 26.833333333333332),
            DVec3::new(16.32268778483073, 18.27001889546712, 26.666666666666668),
            DVec3::new(16.346779505411785, 18.498706817626953, 26.790985743204754),
            DVec3::new(16.202088673909504, 18.457736333211262, 26.70674769083659),
            DVec3::new(16.058500289916992, 18.417078018188477, 26.623419443766277),
            DVec3::new(15.915996233622232, 18.376726786295574, 26.540990829467773),
            DVec3::new(17.12963612874349, 17.60032081604004, 26.70674769083659),
            DVec3::new(17.100369135538738, 17.4539852142334, 26.623419443766277),
            DVec3::new(17.071322758992512, 17.308754603068035, 26.540990829467773),
            DVec3::new(-14.153460820515951, 19.766197840372723, 26.540990829467773),
            DVec3::new(-14.159430185953775, 19.91418393452962, 26.623419443766277),
            DVec3::new(-14.165445327758789, 20.063296635945637, 26.70674769083659),
            DVec3::new(-14.17150624593099, 20.21355374654134, 26.790985743204754),
            DVec3::new(-13.954760869344076, 20.136741638183594, 26.666666666666668),
            DVec3::new(-13.981264114379883, 20.42219416300456, 26.833333333333332),
            DVec3::new(-13.655904134114584, 20.340506235758465, 26.666666666666668),
            DVec3::new(-13.697244962056478, 20.613646189371746, 26.833333333333332),
            DVec3::new(-13.51724942525228, 20.73223940531413, 26.833333333333332),
            DVec3::new(-13.263668060302734, 20.598515192667644, 26.666666666666668),
            DVec3::new(-13.251688321431478, 20.82815678914388, 26.790985743204754),
            DVec3::new(-13.115188280741373, 20.765055974324543, 26.70674769083659),
            DVec3::new(-12.979728062947592, 20.702435811360676, 26.623419443766277),
            DVec3::new(-12.845291455586752, 20.640288670857746, 26.540990829467773),
            DVec3::new(18.498706817626953, 16.34678014119466, 26.790985743204754),
            DVec3::new(18.270018259684246, 16.322688420613606, 26.666666666666668),
            DVec3::new(18.3624267578125, 16.594066619873047, 26.833333333333332),
            DVec3::new(18.21713638305664, 16.75329335530599, 26.833333333333332),
            DVec3::new(17.953826268513996, 16.66973304748535, 26.666666666666668),
            DVec3::new(17.983611424763996, 17.00386683146159, 26.833333333333332),
            DVec3::new(17.705819447835285, 16.933035532633465, 26.666666666666668),
            DVec3::new(17.747779846191406, 17.159128189086914, 26.790985743204754),
            DVec3::new(17.60032081604004, 17.12963612874349, 26.70674769083659),
            DVec3::new(17.4539852142334, 17.100369135538738, 26.623419443766277),
            DVec3::new(17.308754603068035, 17.071322758992512, 26.540990829467773),
            DVec3::new(18.457736333211262, 16.202089309692383, 26.70674769083659),
        ];

        let voronoi = Voronoi::build(&points, anchor, width, 3.try_into().unwrap(), false);
        voronoi.consistency_check();
    }
}
