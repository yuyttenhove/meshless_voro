use glam::DVec3;

use super::integrators::{AreaCentroidIntegrator, VoronoiIntegrator};

pub(super) struct VoronoiFaceBuilder {
    left_idx: usize,
    left_loc: DVec3,
    right_idx: Option<usize>,
    shift: Option<DVec3>,
    normal: DVec3,
    area_centroid: AreaCentroidIntegrator,
}

impl VoronoiFaceBuilder {
    pub(super) fn new(
        left_idx: usize,
        left_loc: DVec3,
        right_idx: Option<usize>,
        shift: Option<DVec3>,
        normal: DVec3,
    ) -> Self {
        Self {
            left_idx,
            left_loc,
            right_idx,
            shift,
            normal,
            area_centroid: AreaCentroidIntegrator::init(),
        }
    }

    pub fn extend(&mut self, v0: DVec3, v1: DVec3, v2: DVec3) {
        self.area_centroid.collect(v0, v1, v2, self.left_loc);
    }

    pub fn build(&self) -> VoronoiFace {
        let (area, centroid) = self.area_centroid.finalize();
        VoronoiFace::new(
            self.left_idx,
            self.right_idx,
            area,
            centroid,
            -self.normal,
            self.shift,
        )
    }
}

/// A Voronoi face between two neighbouring generators.
#[derive(Debug, Clone)]
pub struct VoronoiFace {
    left: usize,
    right: Option<usize>,
    area: f64,
    centroid: DVec3,
    normal: DVec3,
    shift: Option<DVec3>,
}

impl VoronoiFace {
    pub(super) fn new(
        left: usize,
        right: Option<usize>,
        area: f64,
        centroid: DVec3,
        normal: DVec3,
        shift: Option<DVec3>,
    ) -> Self {
        VoronoiFace {
            left,
            right,
            area,
            centroid,
            normal,
            shift,
        }
    }

    /// Get the index of the generator on the _left_ of this face.
    pub fn left(&self) -> usize {
        self.left
    }

    /// Get the index of the generator on the _right_ of this face.
    /// Returns `None` if if this is a boundary face (i.e. obtained by clipping a Voronoi cell with the _simulation volume_).
    pub fn right(&self) -> Option<usize> {
        self.right
    }

    /// Get the area of this face.
    pub fn area(&self) -> f64 {
        self.area
    }

    /// Get the position of the centroid of this face.
    pub fn centroid(&self) -> DVec3 {
        self.centroid
    }

    /// Get a normal vector to this face, pointing away from the _left_ generator.
    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    /// Get the shift vector (if any) to apply to the generator to the right of this face to bring it to the reference frame of this face.
    /// Can only be `Some` for periodic Voronoi tesselations.
    pub fn shift(&self) -> Option<DVec3> {
        self.shift
    }
}
