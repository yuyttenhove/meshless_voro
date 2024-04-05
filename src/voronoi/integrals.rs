//! Contains traits used to define custom integrals over cells and faces.

use glam::DVec3;

use crate::geometry::{signed_area_tri, signed_volume_tet};

use super::convex_cell::ConvexCell;

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
pub trait CellIntegral: Sized {
    /// Initialize a [`CellIntegral`] for the given [`ConvexCell`].
    fn init(cell: &ConvexCell) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with
    /// the cell's generator `gen` as top), which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
}

/// Trait to implement new integrators that use external data in their
/// calculation.
pub trait CellIntegralWithData<D: Copy>: CellIntegral {
    /// Initialize a [`CellIntegral`] with some extra data.
    fn init_with_data(cell: &ConvexCell, data: D) -> Self;
}

impl<T: CellIntegral> CellIntegralWithData<()> for T {
    fn init_with_data(cell: &ConvexCell, _data: ()) -> Self {
        T::init(cell)
    }
}

#[derive(Default)]
pub(super) struct VolumeCentroidIntegrator {
    pub centroid: DVec3,
    pub volume: f64,
}

impl VolumeCentroidIntegrator {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            volume: 0.,
        }
    }
}

impl CellIntegral for VolumeCentroidIntegrator {
    fn init(_cell: &ConvexCell) -> Self {
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
#[derive(Default)]
pub struct VolumeIntegral {
    pub volume: f64,
}

impl CellIntegral for VolumeIntegral {
    fn init(_cell: &ConvexCell) -> Self {
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
pub trait FaceIntegral: Clone {
    /// Initialize a [`FaceIntegral`] for the given [`ConvexCell`] and
    /// clipping_plane_index.
    fn init(cell: &ConvexCell, clipping_plane_idx: usize) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with
    /// the cell's generator `gen` as top), which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
}

/// Trait to implement new integrators that use external data in their
/// calculation.
pub trait FaceIntegralWithData<D: Copy>: FaceIntegral {
    /// Initialize a [`CellIntegral`] with some extra data.
    fn init_with_data(cell: &ConvexCell, clipping_plane_idx: usize, data: D) -> Self;
}

impl<T: FaceIntegral> FaceIntegralWithData<()> for T {
    fn init_with_data(cell: &ConvexCell, clipping_plane_idx: usize, _data: ()) -> Self {
        T::init(cell, clipping_plane_idx)
    }
}

#[derive(Default, Clone)]
pub(super) struct AreaCentroidIntegrator {
    pub centroid: DVec3,
    pub area: f64,
}

impl AreaCentroidIntegrator {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            area: 0.,
        }
    }
}

impl FaceIntegral for AreaCentroidIntegrator {
    fn init(_cell: &ConvexCell, _clipping_plane_idx: usize) -> Self {
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

/// Example implementation of a simple face integrator for computing the area of
/// the faces of a [`ConvexCell`].
pub struct AreaIntegral {
    pub area: f64,
}

impl CellIntegral for AreaIntegral {
    fn init(_cell: &ConvexCell) -> Self {
        Self { area: 0. }
    }

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        self.area += signed_area_tri(v0, v1, v2, gen);
    }

    fn finalize(self) -> Self {
        self
    }
}
