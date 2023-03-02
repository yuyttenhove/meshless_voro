use glam::DVec3;

use crate::util::signed_area_tri;

use super::Dimensionality;

/// A Voronoi face between two neighbouring generators.
pub struct VoronoiFace {
    left: usize,
    right: Option<usize>,
    area: f64,
    centroid: DVec3,
    normal: DVec3,
    shift: Option<DVec3>,
}

impl VoronoiFace {
    pub(super) fn init(
        left: usize,
        right: Option<usize>,
        normal: DVec3,
        shift: Option<DVec3>,
    ) -> Self {
        VoronoiFace {
            left,
            right,
            area: 0.,
            centroid: DVec3::ZERO,
            normal,
            shift,
        }
    }

    pub(super) fn extend(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, t: DVec3) {
        let area = signed_area_tri(v0, v1, v2, t);
        self.area += area;
        self.centroid += area * (v0 + v1 + v2);
    }

    pub(super) fn finalize(&mut self) {
        let area_inv = if self.area != 0. { 1. / self.area } else { 0. };
        self.centroid *= area_inv / 3.;
    }

    pub(super) fn has_valid_dimensionality(&self, dimensionality: Dimensionality) -> bool {
        match dimensionality {
            Dimensionality::Dimensionality1D => self.normal.y == 0. && self.normal.z == 0.,
            Dimensionality::Dimensionality2D => self.normal.z == 0.,
            Dimensionality::Dimensionality3D => true,
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
