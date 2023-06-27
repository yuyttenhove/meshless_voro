use glam::DVec3;

use crate::util::{signed_area_tri, signed_volume_tet};

use super::convex_cell::ConvexCell;

/// Trait to implement new integrators for Voronoi cells
pub trait CellIntegral {
    /// Initialize a CellIntegral for the given ConvexCell.
    fn init(cell: &ConvexCell) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with the cell's generator `gen` as top),
    /// which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
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

/// Trait to implement new integrators for Voronoi faces
pub trait FaceIntegral: Clone {
    /// Initialize a FaceIntegral for the given ConvexCell and clipping_plane_index.
    fn init(cell: &ConvexCell, clipping_plane_idx: usize) -> Self;

    /// Update the state of the integrator using one oriented tetrahedron (with the cell's generator `gen` as top),
    /// which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(self) -> Self;
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
