use glam::DVec3;

use crate::util::{signed_area_tri, signed_volume_tet};

/// Trait to implement new integrators for Voronoi cells/faces
pub trait VoronoiIntegrator {
    type Output;

    /// Update the state of the integrator using one oriented tetrahedron (with the cell's generator `gen` as top),
    /// which is part of a cell.
    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3);

    /// Finalize the calculation and return the result
    fn finalize(&self) -> Self::Output;
}

pub trait VectorVoronoiIntegrator: VoronoiIntegrator<Output = DVec3> {}
impl<T: VoronoiIntegrator<Output = DVec3>> VectorVoronoiIntegrator for T {}

pub trait ScalarVoronoiIntegrator: VoronoiIntegrator<Output = f64> {}
impl<T: VoronoiIntegrator<Output = f64>> ScalarVoronoiIntegrator for T {}

#[derive(Default)]
pub struct VolumeCentroidIntegrator {
    centroid: DVec3,
    volume: f64,
}

impl VolumeCentroidIntegrator {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            volume: 0.,
        }
    }
}

impl VoronoiIntegrator for VolumeCentroidIntegrator {
    type Output = (f64, DVec3);

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        let volume = signed_volume_tet(v0, v1, v2, gen);
        self.volume += volume;
        self.centroid += volume * (v0 + v1 + v2 + gen);
    }

    fn finalize(&self) -> Self::Output {
        let normalisation = if self.volume > 0. {
            0.25 / self.volume
        } else {
            0.
        };
        (self.volume, normalisation * self.centroid)
    }
}

#[derive(Default)]
pub struct AreaCentroidIntegrator {
    centroid: DVec3,
    area: f64,
}

impl AreaCentroidIntegrator {
    pub fn init() -> Self {
        Self {
            centroid: DVec3::ZERO,
            area: 0.,
        }
    }
}

impl VoronoiIntegrator for AreaCentroidIntegrator {
    type Output = (f64, DVec3);

    fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        let area = signed_area_tri(v0, v1, v2, gen);
        self.area += area;
        self.centroid += area * (v0 + v1 + v2);
    }

    fn finalize(&self) -> Self::Output {
        let normalisation = if self.area > 0. {
            1. / (3. * self.area)
        } else {
            0.
        };
        (self.area, normalisation * self.centroid)
    }
}
