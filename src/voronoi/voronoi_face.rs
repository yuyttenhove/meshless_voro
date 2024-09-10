use glam::DVec3;

use super::{
    half_space::HalfSpace,
    integrals::{AreaCentroidIntegrator, FaceIntegral},
};

pub(super) struct VoronoiFaceBuilder<'a> {
    left_idx: usize,
    left_loc: DVec3,
    half_space: &'a HalfSpace,
    area_centroid: AreaCentroidIntegrator,
}

impl<'a> VoronoiFaceBuilder<'a> {
    pub(super) fn new(left_idx: usize, left_loc: DVec3, half_space: &'a HalfSpace) -> Self {
        Self {
            left_idx,
            left_loc,
            half_space,
            area_centroid: AreaCentroidIntegrator::init(),
        }
    }

    pub fn extend(&mut self, v0: DVec3, v1: DVec3, v2: DVec3) {
        self.area_centroid.collect(v0, v1, v2, self.left_loc);
    }

    pub fn build(self) -> VoronoiFace {
        let AreaCentroidIntegrator { area, centroid } = self.area_centroid.finalize();
        VoronoiFace::new(
            self.left_idx,
            self.half_space.right_idx,
            area,
            centroid,
            -self.half_space.normal(),
            self.half_space.shift,
        )
    }
}

/// A Voronoi face between two neighbouring generators.
#[derive(Debug, Clone)]
pub struct VoronoiFace {
    /// Index of generator on the left of this face
    left: usize,
    /// Index of the generator on the right of this face. May be `None` for
    /// boundary faces when reflective boundary conditions are used.
    right: Option<usize>,
    area: f64,
    centroid: DVec3,
    /// We follow the convention that the normals of faces point from the left
    /// generator to the right generator.
    normal: DVec3,
    /// Shift to apply to the right generator to bring it in the reference frame
    /// of the left (if any), when periodic boundary conditions are used.
    shift: Option<DVec3>,
}

impl VoronoiFace {
    fn new(
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
    /// Returns `None` if if this is a boundary face (i.e. obtained by clipping
    /// a Voronoi cell with the _simulation volume_).
    pub fn right(&self) -> Option<usize> {
        self.right
    }

    /// Update the index of the generator on the _left_ of this face.
    pub fn set_left(&mut self, left: usize) {
        self.left = left;
    }

    /// Update the index of the generator on the _right_ of this face.
    pub fn set_right(&mut self, right: usize) {
        self.right = Some(right);
    }

    /// Get the area of this face.
    pub fn area(&self) -> f64 {
        self.area
    }

    /// Get the position of the centroid of this face.
    pub fn centroid(&self) -> DVec3 {
        self.centroid
    }

    /// Get a normal vector to this face, pointing away from the _left_
    /// generator.
    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    /// Get the shift vector (if any) to apply to the generator to the right of
    /// this face to bring it to the reference frame of this face.
    /// Can only be `Some` for periodic Voronoi tessellations.
    pub fn shift(&self) -> Option<DVec3> {
        self.shift
    }

    /// Update the shift vector associated with this face
    pub fn set_shift(&mut self, shift: DVec3) {
        self.shift = Some(shift)
    }

    /// Whether this is a face between a particle and a periodic boundary neighbour
    pub fn is_periodic(&self) -> bool {
        // Periodically wrapping faces must have shift set
        self.shift.is_some()
    }

    /// Whether this is a boundary face
    pub fn is_boundary(&self) -> bool {
        self.right.is_none()
    }
}
