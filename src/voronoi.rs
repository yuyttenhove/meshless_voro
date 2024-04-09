use self::integrals::{CellIntegral, CellIntegralWithData, FaceIntegral, FaceIntegralWithData};
use crate::rtree_nn::{build_rtree, nn_iter, wrapping_nn_iter};

use glam::DVec3;
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

mod boundary;
mod convex_cell;
#[allow(unused)]
mod convex_cell_alternative;
mod generator;
mod half_space;
pub mod integrals;
mod voronoi_cell;
mod voronoi_face;

/// The dimensionality of the Voronoi tessellation.
#[derive(Clone, Copy, Debug, PartialEq, num_enum::IntoPrimitive)]
#[repr(usize)]
pub enum Dimensionality {
    OneD = 1,
    TwoD = 2,
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

macro_rules! cells_map {
    ($cells:expr, $mappable:expr) => {
        $cells
            .iter()
            .filter_map(|maybe_cell| maybe_cell.as_ref().map($mappable))
            .collect()
    };
}

/// The main Voronoi struct.
///
/// This representation has pre-calculated faces (area, normal, centroid) and
/// cells (volume, centroid).
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
            let faces = cell.faces(self);
            if self.dimensionality == Dimensionality::ThreeD {
                let area: f64 = faces.map(|f| f.area()).sum();
                let radius =
                    (0.25 * 3. * std::f64::consts::FRAC_1_PI * cell.volume()).powf(1. / 3.);
                let sphere_area = 4. * std::f64::consts::PI * radius * radius;
                assert!(area > sphere_area);
                total_volume += cell.volume();
            } else {
                // Just check that we only have faces with correct dimensionality
                for f in faces {
                    assert!(self.dimensionality.vector_is_valid(f.normal()))
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
    ///
    /// Requires the `hdf5` feature to be enabled.
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

impl From<VoronoiIntegrator> for Voronoi {
    fn from(voronoi_integrator: VoronoiIntegrator) -> Self {
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

/// An intermediate (meshless) representation of the Voronoi mesh.
///
/// Can be used to compute integrals over the faces (e.g. area, solid angle,
/// ...) or cells ((weighted) centroid, ...). Can also be converted to a Voronoi
/// struct, which implements the `From<VoronoiIntegrator>` trait.
pub struct VoronoiIntegrator {
    cells: Vec<Option<ConvexCell>>,
    anchor: DVec3,
    width: DVec3,
    dimensionality: Dimensionality,
    periodic: bool,
}

impl VoronoiIntegrator {
    /// Build the `ConvexCell`s for the given generators.
    ///
    /// * `generators` - The seed points of the Voronoi cells.
    /// * `mask` - (Optional) `True` for Voronoi cells which have to be fully
    ///   constructed.
    /// * `anchor` - The lower left corner of the simulation volume.
    /// * `width` - The width of the simulation volume. Also determines the
    ///   period of periodic Voronoi tessellations.
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
            if mask.map_or(true, |mask| mask[idx]) {
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
            anchor,
            width,
            dimensionality,
            periodic,
        }
    }

    /// Compute a custom cell integral for the active cells in this
    /// representation.
    pub fn compute_cell_integrals<T: CellIntegral>(&self) -> Vec<T> {
        cells_map!(self.cells, |cell| cell.compute_cell_integral(()))
    }

    pub fn compute_cell_integrals_with_data<D: Copy, T: CellIntegralWithData<D>>(
        &self,
        extra_data: D,
    ) -> Vec<T> {
        cells_map!(self.cells, |cell| cell.compute_cell_integral(extra_data))
    }

    /// Compute a custom face integral for all the faces of each active cell in
    /// this representation.
    ///
    /// Assumes non-symmetric integral (i.e. The integral of a face evaluated
    /// from the left cell might be different that the integral evaluated
    /// from the right cell). Therefore all faces are treated twice.
    /// For symmetric integrals, `compute_face_integrals_sym` will be more
    /// efficient.
    ///
    /// Returns a vector with for each active cell a vector of face integrals.
    pub fn compute_face_integrals<T: FaceIntegral>(&self) -> Vec<Vec<T>> {
        cells_map!(self.cells, |cell| cell.compute_face_integrals(()))
    }

    pub fn compute_face_integrals_sym<T: FaceIntegral>(&self) -> Vec<Vec<T>> {
        cells_map!(self.cells, |cell| cell
            .compute_face_integrals_sym((), &self.active_cells_mask()))
    }

    pub fn compute_face_integrals_with_data<D: Copy, T: FaceIntegralWithData<D>>(
        &self,
        extra_data: D,
    ) -> Vec<Vec<T>> {
        cells_map!(self.cells, |cell| cell.compute_face_integrals(extra_data))
    }

    pub fn compute_face_integrals_sym_with_data<D: Copy, T: FaceIntegralWithData<D>>(
        &self,
        extra_data: D,
    ) -> Vec<Vec<T>> {
        cells_map!(self.cells, |cell| cell
            .compute_face_integrals_sym(extra_data, &self.active_cells_mask()))
    }

    fn build_voronoi_cells(&self, faces: &mut [Vec<VoronoiFace>]) -> Vec<VoronoiCell> {
        let build = |(convex_cell, faces): (&Option<ConvexCell>, _)| match convex_cell {
            Some(convex_cell) => {
                VoronoiCell::from_convex_cell(convex_cell, faces, Some(&self.active_cells_mask()))
            }
            None => VoronoiCell::default(),
        };
        #[cfg(feature = "rayon")]
        let voronoi_cells = self.cells.par_iter().zip(faces.par_iter_mut()).map(build).collect();
        #[cfg(not(feature = "rayon"))]
        let voronoi_cells = self.cells.iter().zip(faces.iter_mut()).map(build).collect();

        voronoi_cells
    }

    fn active_cells_mask(&self) -> Vec<bool> {
        self.cells.iter().map(|cell| cell.is_some()).collect()
    }
}

#[cfg(test)]
mod test {
    use super::{
        integrals::{AreaCentroidIntegrator, VolumeCentroidIntegrator},
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

        let area_centroids = integrator.compute_face_integrals::<AreaCentroidIntegrator>();
        let volume_centroids = integrator.compute_cell_integrals::<VolumeCentroidIntegrator>();

        assert_eq!(area_centroids.len(), 1);
        for (i, AreaCentroidIntegrator { area, centroid }) in area_centroids[0].iter().enumerate() {
            assert_eq!(*area, voronoi.faces()[i].area());
            assert_eq!(*centroid, voronoi.faces()[i].centroid());
        }
        assert_eq!(volume_centroids.len(), 1);
        assert_eq!(volume_centroids[0].volume, voronoi.cells()[62].volume());
        assert_eq!(volume_centroids[0].centroid, voronoi.cells()[62].centroid());

        let voronoi2: Voronoi = integrator.into();
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
}
