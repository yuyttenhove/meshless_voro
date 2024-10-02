//! Contains traits used to define custom integrals over cells and faces.

use glam::DVec3;

use crate::geometry::{signed_area_tri, signed_volume_tet};

use super::convex_cell::{ConvexCell, ConvexCellMarker};

/// Trait to implement new integrators for Voronoi cells.
///
/// Integrators are expected to compute quantities of interest for Voronoi cells
/// iteratively. The Voronoi cell is decomposed into a number of oriented
/// tetrahedra, which are fed one by one to the cell-integrators.
///
/// We use the following orientation convention:
///
/// - If the three vertices are ordered counterclockwise as seen from the top,
///   the tetrahedron is assumed to be part of the Voronoi cell and should
///   contribute positively to integrals.
///
/// - If the three vertices are ordered clockwise, the tetrahedron should
///   subtract from the integrals. this is to correct for another tetrahedron
///   that is not fully contained within the Voronoi cell.
pub trait CellIntegral: Sized + Send {
    /// Initialize a [`CellIntegral`] for the given [`ConvexCell`].
    fn init<M: ConvexCellMarker>(cell: &ConvexCell<M>) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with
    /// the cell's generator `gen` as top), which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
}

/// Trait to implement new integrators that use external data in their
/// calculation.
pub trait CellIntegralWithData: CellIntegral {
    type Data: Copy;

    /// Initialize a [`CellIntegral`] with some extra data.
    fn init_with_data<M: ConvexCellMarker>(cell: &ConvexCell<M>, data: Self::Data) -> Self;
}

impl<T: CellIntegral> CellIntegralWithData for T {
    type Data = ();

    fn init_with_data<M: ConvexCellMarker>(cell: &ConvexCell<M>, _data: ()) -> Self {
        T::init(cell)
    }
}

/// Example implementation of a simple cell integrator for computing the volume
/// of a [`ConvexCell`].
/// 
/// Use as follows:
/// ```
/// # use glam::DVec3;
/// # use meshless_voronoi::VoronoiIntegrator;
/// # use meshless_voronoi::integrals::VolumeCentroidIntegral;
/// # let generators = vec![DVec3::splat(1.), DVec3::splat(2.)];
/// let voronoi_integrator = VoronoiIntegrator::build(&generators, None, DVec3::ZERO, DVec3::splat(3.), 3.try_into().unwrap(), false);
/// // Compute volumes of all the voronoi cells.
/// let all_centroid_volumes = voronoi_integrator.compute_cell_integrals::<VolumeCentroidIntegral>();
/// // Compute the volume of a specific ConvexCell:
/// let convex_cell = voronoi_integrator.get_cell_at(0).unwrap();
/// // Here we need to explicitely specify that VolumeIntegral needs no extra data (empty type).
/// let convex_cell_areas = convex_cell.compute_cell_integral::<(), VolumeCentroidIntegral>(());
#[derive(Default)]
pub struct VolumeCentroidIntegral {
    /// The centroid of a [`ConvexCell`]
    pub centroid: DVec3,
    /// The volume of a [`ConvexCell`]
    pub volume: f64,
}

impl VolumeCentroidIntegral {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            volume: 0.,
        }
    }
}

impl CellIntegral for VolumeCentroidIntegral {
    fn init<M: ConvexCellMarker>(_cell: &ConvexCell<M>) -> Self {
        Self::default()
    }

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        let volume = signed_volume_tet(v0, v1, v2, gen);
        self.volume += volume;
        self.centroid += volume * (v0 + v1 + v2 + gen);
    }

    fn finalize(mut self) -> Self {
        let normalisation = if self.volume > 0. {
            0.25 / self.volume
        } else {
            0.
        };
        self.centroid *= normalisation;
        self
    }
}

/// Example implementation of a simple cell integrator for computing the volume
/// of a [`ConvexCell`].
///
///  Use as follows:
/// ```
/// # use glam::DVec3;
/// # use meshless_voronoi::VoronoiIntegrator;
/// # use meshless_voronoi::integrals::VolumeIntegral;
/// # let generators = vec![DVec3::splat(1.), DVec3::splat(2.)];
/// let voronoi_integrator = VoronoiIntegrator::build(&generators, None, DVec3::ZERO, DVec3::splat(3.), 3.try_into().unwrap(), false);
/// // Compute volumes of all the voronoi cells.
/// let all_centroid_volumes = voronoi_integrator.compute_cell_integrals::<VolumeIntegral>();
/// // Compute the volume and centroid of a specific ConvexCell:
/// let convex_cell = voronoi_integrator.get_cell_at(0).unwrap();
/// // Here we need to explicitely specify that VolumeIntegral needs no extra data (empty type).
/// let convex_cell_areas = convex_cell.compute_cell_integral::<(), VolumeIntegral>(());
#[derive(Default)]
pub struct VolumeIntegral {
    /// The volume of a [`ConvexCell`]
    pub volume: f64,
}

impl CellIntegral for VolumeIntegral {
    fn init<M: ConvexCellMarker>(_cell: &ConvexCell<M>) -> Self {
        Self::default()
    }

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        self.volume += signed_volume_tet(v0, v1, v2, gen);
    }

    fn finalize(self) -> Self {
        self
    }
}

/// Trait to implement new integrators for Voronoi faces.
///
/// Integrators are expected to compute quantities of interest for Voronoi faces
/// iteratively. The Voronoi cell is decomposed into a number of oriented
/// tetrahedra. Tetrahedra contributing to the same face are fed one by one to
/// the face-integrators.
///
/// We use the following orientation convention:
///
/// - If the three vertices are ordered counterclockwise as seen from the top,
///   the tetrahedron is assumed to be part of the Voronoi cell and should
///   contribute positively to integrals.
///
/// - If the three vertices are ordered clockwise, the tetrahedron should
///   subtract from the integrals. this is to correct for another tetrahedron
///   that is not fully contained within the Voronoi cell.
pub trait FaceIntegral: Clone + Send {
    /// Initialize a [`FaceIntegral`] for the given [`ConvexCell`] and
    /// clipping_plane_index.
    fn init<M: ConvexCellMarker>(cell: &ConvexCell<M>, clipping_plane_idx: usize) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with
    /// the cell's generator `gen` as top), which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
}

/// Trait to implement new integrators that use external data in their
/// calculation.
/// The data is the cloned for all faces of a [`ConvexCell`].
pub trait FaceIntegralWithData: FaceIntegral {
    type Data: Copy;

    /// Initialize a [`CellIntegral`] with some extra data.
    fn init_with_data<M: ConvexCellMarker>(cell: &ConvexCell<M>, clipping_plane_idx: usize, data: Self::Data) -> Self;
}

impl<T: FaceIntegral> FaceIntegralWithData for T {
    type Data = ();

    fn init_with_data<M: ConvexCellMarker>(cell: &ConvexCell<M>, clipping_plane_idx: usize, _data: ()) -> Self {
        T::init(cell, clipping_plane_idx)
    }
}

/// Struct wrapping a FaceIntegral with some extra info needed for bookkeeping.
/// 
/// When computing FaceIntegrals, they will always be wrapped in a FaceIntegrator,
/// making it possible to retrieve some info about the face for which the integral was 
/// evaluated.
#[derive(Clone)]
pub struct FaceIntegrator<I: FaceIntegralWithData> {
    pub(super) left: usize,

    pub(super) right: Option<usize>,

    pub(super) integral: I,

    pub(super) shift: Option<DVec3>,
}

impl<D: Copy, I: FaceIntegralWithData<Data = D>> FaceIntegrator<I> {
    pub(crate) fn init<M: ConvexCellMarker>(cell: &ConvexCell<M>, clipping_plane_idx: usize, data: D) -> Self {
        Self {
            left: cell.idx,
            right: cell.clipping_planes[clipping_plane_idx].right_idx,
            integral: I::init_with_data(cell, clipping_plane_idx, data),
            shift: cell.clipping_planes[clipping_plane_idx].shift,
        }
    }

    pub(crate) fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        self.integral.collect(v0, v1, v2, gen);
    }

    pub(crate) fn finalize(mut self) -> Self {
        self.integral = self.integral.finalize();
        self
    }
    
    /// Get the index of the generator on the left of this face.
    pub fn left(&self) -> usize {
        self.left
    }

    /// Get the index of the generator on the right of this face (if any).
    /// Boundary faces have no right generator.
    pub fn right(&self) -> Option<usize> {
        self.right
    }
    
    /// The shift to apply to the right generator's position (for periodic boundary 
    /// conditions), if any.
    pub fn shift(&self) -> Option<DVec3> {
        self.shift
    }
    
    /// The integral evaluated for this face.
    pub fn integral(&self) -> &I {
        &self.integral
    }
}

/// Example implementation of a simple FaceIntegral for computing the area *and* centroid 
/// of the faces of a [`ConvexCell`].
/// 
/// Use as follows:
/// ```
/// # use glam::DVec3;
/// # use meshless_voronoi::VoronoiIntegrator;
/// # use meshless_voronoi::integrals::AreaCentroidIntegral;
/// # let generators = vec![DVec3::splat(1.), DVec3::splat(2.)];
/// let voronoi_integrator = VoronoiIntegrator::build(&generators, None, DVec3::ZERO, DVec3::splat(3.), 3.try_into().unwrap(), false);
/// // Compute areas and centroids of all the voronoi faces (treating all the faces of every ConvexCell, 
/// // i.e. twice for each face shared between two ConvexCells, assuming they might differ 
/// // from both sides of the face).
/// // This returns a Vec<FaceIntegrator<AreaIntegral>>.
/// let all_face_areas = voronoi_integrator.compute_face_integrals::<AreaCentroidIntegral>();
/// // Compute areas and centroids of all the voronoi faces (once for each face, 
/// // assuming they are the same from both sides of the face)
/// let all_face_areas_sym = voronoi_integrator.compute_face_integrals_sym::<AreaCentroidIntegral>();
/// // Compute the areas and centroids of the voronoi faces of a specific ConvexCell:
/// let convex_cell = voronoi_integrator.get_cell_at(0).unwrap();
/// // Here we need to explicitely specify that AreaCentroidIntegral needs no extra data (empty type).
/// let convex_cell_areas = convex_cell.compute_face_integrals::<(), AreaCentroidIntegral>(());
#[derive(Default, Clone)]
pub struct AreaCentroidIntegral {
    /// The centroid of a face
    pub centroid: DVec3,
    /// The area of a face
    pub area: f64,
}

impl AreaCentroidIntegral {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            area: 0.,
        }
    }
}

impl FaceIntegral for AreaCentroidIntegral {
    fn init<M: ConvexCellMarker>(_cell: &ConvexCell<M>, _clipping_plane_idx: usize) -> Self {
        Self::default()
    }

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        let area = signed_area_tri(v0, v1, v2, gen);
        self.area += area;
        self.centroid += area * (v0 + v1 + v2);
    }

    fn finalize(mut self) -> Self {
        let normalisation = if self.area > 0. {
            1. / (3. * self.area)
        } else {
            0.
        };
        self.centroid *= normalisation;
        self
    }
}

/// Example implementation of a simple face integral for computing the area of
/// the faces of a [`ConvexCell`].
/// 
/// Use as follows:
/// ```
/// use glam::DVec3;
/// use meshless_voronoi::VoronoiIntegrator;
/// use meshless_voronoi::integrals::AreaIntegral;
/// # let generators = vec![DVec3::splat(1.), DVec3::splat(2.)];
/// let voronoi_integrator = VoronoiIntegrator::build(&generators, None, DVec3::ZERO, DVec3::splat(3.), 3.try_into().unwrap(), false);
/// // Compute areas of all the voronoi faces (twice for each face, 
/// // assuming they might differ from both sides of the face)
/// let all_face_areas = voronoi_integrator.compute_face_integrals::<AreaIntegral>();
/// // Compute areas of all the voronoi faces (once for each face, 
/// // assuming they are the same from both sides of the face)
/// let all_face_areas_sym = voronoi_integrator.compute_face_integrals_sym::<AreaIntegral>();
/// // Compute the areas of the voronoi faces of a specific ConvexCell:
/// let convex_cell = voronoi_integrator.get_cell_at(0).unwrap();
/// // Here we need to explicitely specify that AreaIntegral needs no extra data (empty type).
/// let convex_cell_areas = convex_cell.compute_face_integrals::<(), AreaIntegral>(());
/// ```
#[derive(Default, Clone)]
pub struct AreaIntegral {
    /// The area of a face
    pub area: f64,
}

impl FaceIntegral for AreaIntegral {
    fn init<M: ConvexCellMarker>(_cell: &ConvexCell<M>, _clipping_plane_idx: usize) -> Self {
        Self { area: 0. }
    }

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        self.area += signed_area_tri(v0, v1, v2, gen);
    }

    fn finalize(self) -> Self {
        self
    }
}
