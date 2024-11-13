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
        let anchor = DVec3::new(0.9948180758087021, 1.6889588194242708, 0.91);
        let width = DVec3::new(0.34199451271047493, 0.34004556816642234, 0.20590832420001487);
        let points = vec![
            DVec3::new(1.2239545773396252, 1.9094804159447902, 1.01),
            DVec3::new(1.2238339347857417, 1.9124712565164197, 1.0116659054711301),
            DVec3::new(1.2237123670786065, 1.9154848661375201, 1.0133499927735252),
            DVec3::new(1.2235898742182199, 1.9185216045899114, 1.0150524674967965),
            DVec3::new(1.2279703592205604, 1.9169692102830167, 1.0125399440056135),
            DVec3::new(1.2274347211865646, 1.9227382860693099, 1.0159083242000149),
            DVec3::new(1.2340103378680234, 1.921087350091765, 1.0125399440056135),
            DVec3::new(1.2331748281116657, 1.9266075853520692, 1.0159083242000149),
            DVec3::new(1.236812588519177, 1.9290043875906933, 1.0159083242000149),
            DVec3::new(1.1934934825009318, 1.885806091167891, 1.01),
            DVec3::new(1.1931385513107469, 1.8887782487835776, 1.0116659054711301),
            DVec3::new(1.1927809217569112, 1.8917730341058774, 1.0133499927735252),
            DVec3::new(1.1924205424420222, 1.89479080691661, 1.0150524674967965),
            DVec3::new(1.196909321772207, 1.8935868741518143, 1.0125399440056135),
            DVec3::new(1.1959227036514153, 1.899296136210519, 1.0159083242000149),
            DVec3::new(1.2026075847866977, 1.8981662028565032, 1.0125399440056135),
            DVec3::new(1.201341538260535, 1.9036038681890988, 1.0159083242000149),
            DVec3::new(1.2047800309371248, 1.9062786918291905, 1.0159083242000149),
            DVec3::new(1.21010122975482, 1.9039865347028069, 1.0125399440056135),
            DVec3::new(1.2099784606333928, 1.9086323330987556, 1.0150524674967965),
            DVec3::new(1.2128287164588258, 1.9075774271035773, 1.0133499927735252),
            DVec3::new(1.215657256881547, 1.9065305648019484, 1.0116659054711301),
            DVec3::new(1.2184644288340252, 1.9054916048510102, 1.01),
            DVec3::new(1.164983767158252, 1.8598147971683732, 1.01),
            DVec3::new(1.1643967317213875, 1.8627499486539945, 1.0116659054711301),
            DVec3::new(1.163805237559824, 1.8657074323111624, 1.0133499927735252),
            DVec3::new(1.1632091947281067, 1.8686876207710474, 1.0150524674967965),
            DVec3::new(1.1677785908846912, 1.8678395893224156, 1.0125399440056135),
            DVec3::new(1.1663470704199927, 1.8734538432819172, 1.0159083242000149),
            DVec3::new(1.1731000081413487, 1.8728518768995195, 1.0125399440056135),
            DVec3::new(1.171411230825942, 1.8781734483483856, 1.0159083242000149),
            DVec3::new(1.174629247918251, 1.8811098076729744, 1.0159083242000149),
            DVec3::new(1.180113877627718, 1.879242205943637, 1.0125399440056135),
            DVec3::new(1.1796269771809924, 1.8838640659991994, 1.0150524674967965),
            DVec3::new(1.1825512195678547, 1.8830360409896343, 1.0133499927735252),
            DVec3::new(1.1854531747558983, 1.8822143250112715, 1.0116659054711301),
            DVec3::new(1.1883332153762938, 1.8813988152693049, 1.01),
            DVec3::new(1.1386011847306952, 1.8316667589250049, 1.01),
            DVec3::new(1.1377856749887285, 1.8345468059700756, 1.0116659054711301),
            DVec3::new(1.1369639590103657, 1.8374487675827944, 1.0133499927735252),
            DVec3::new(1.1361359340008006, 1.8403730099696571, 1.0150524674967965),
            DVec3::new(1.1407577812070124, 1.8398861095229315, 1.0125399440056135),
            DVec3::new(1.138890179477675, 1.845370752081749, 1.0159083242000149),
            DVec3::new(1.1456695226147544, 1.8453004789826768, 1.0125399440056135),
            DVec3::new(1.1435684353335351, 1.8504731393060818, 1.0159083242000149),
            DVec3::new(1.1465461567180826, 1.8536529295800073, 1.0159083242000149),
            DVec3::new(1.1521604106775842, 1.8522214091153089, 1.0125399440056135),
            DVec3::new(1.1513123792289526, 1.8567908052718933, 1.0150524674967965),
            DVec3::new(1.1542925676888376, 1.8561947624401758, 1.0133499927735252),
            DVec3::new(1.1572500513460056, 1.8556032682786125, 1.0116659054711301),
            DVec3::new(1.1601852028316268, 1.8550162328417479, 1.01),
            DVec3::new(1.1145083951489898, 1.801535558316624, 1.01),
            DVec3::new(1.1134694351980516, 1.8043427366937779, 1.0116659054711301),
            DVec3::new(1.1124225728964225, 1.807171277116499, 1.0133499927735252),
            DVec3::new(1.1113676669012444, 1.8100215329419318, 1.0150524674967965),
            DVec3::new(1.1160134652971931, 1.8098987573958292, 1.0125399440056135),
            DVec3::new(1.1137213081708095, 1.8152199626381997, 1.0159083242000149),
            DVec3::new(1.12048525778573, 1.815681806851027, 1.0125399440056135),
            DVec3::new(1.1179847998348704, 1.8206736768098413, 1.0159083242000149),
            DVec3::new(1.1207038509401306, 1.8240772899239093, 1.0159083242000149),
            DVec3::new(1.1264131129988348, 1.8230906653784422, 1.0125399440056135),
            DVec3::new(1.1252091930833898, 1.8275794511333023, 1.0150524674967965),
            DVec3::new(1.1282269658941226, 1.8272190718184134, 1.0133499927735252),
            DVec3::new(1.1312217512164224, 1.826861442264578, 1.0116659054711301),
            DVec3::new(1.1341939088321091, 1.8265065046497173, 1.01),
            DVec3::new(1.094818075808702, 1.788958819424271, 1.0159083242000149),
            DVec3::new(1.0972617139306904, 1.7925652723887602, 1.0159083242000149),
            DVec3::new(1.1030307897169833, 1.7920296279300887, 1.0125399440056135),
            DVec3::new(1.1014783954100886, 1.796410119357105, 1.0150524674967965),
            DVec3::new(1.10451513386248, 1.796287626496718, 1.0133499927735252),
            DVec3::new(1.1075287306342296, 1.796166058789583, 1.0116659054711301),
            DVec3::new(1.110519571205859, 1.7960454098110241, 1.01),
        ];

        let voronoi = Voronoi::build(&points, anchor, width, 3.try_into().unwrap(), false);
        voronoi.consistency_check();
    }
}
