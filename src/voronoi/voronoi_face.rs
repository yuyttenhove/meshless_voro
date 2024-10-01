use glam::DVec3;

use crate::geometry::signed_area_tri;

use super::{convex_cell::ConvexCellMarker, integrals::{FaceIntegral, FaceIntegrator}, ConvexCell};


#[derive(Clone)]
struct VoronoiFaceIntegral {
    area: f64, 
    centroid: DVec3,
    normal: DVec3,
}

impl FaceIntegral for VoronoiFaceIntegral {
    fn init<M: super::convex_cell::ConvexCellMarker>(cell: &super::ConvexCell<M>, clipping_plane_idx: usize) -> Self {
        Self {
            area: 0., 
            centroid: DVec3::ZERO,
            normal: cell.clipping_planes[clipping_plane_idx].plane.n,
        }
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

/// A Voronoi face between two neighbouring generators.
#[derive(Clone)]
pub struct VoronoiFace {
    inner: FaceIntegrator<VoronoiFaceIntegral>
}

impl VoronoiFace {
    pub(super) fn init<M: ConvexCellMarker>(convex_cell: &ConvexCell<M>, clipping_plane_idx: usize) -> Self {
        Self {
            inner: FaceIntegrator::init(convex_cell, clipping_plane_idx, ())
        }
    }

    pub(super) fn collect(&mut self, v0: DVec3, v1: DVec3, v2: DVec3, gen: DVec3) {
        self.inner.collect(v0, v1, v2, gen);
    }

    pub(super) fn finalize(mut self) -> Self {
        self.inner = self.inner.finalize();
        self
    }

    /// Get the index of the generator on the _left_ of this face.
    pub fn left(&self) -> usize {
        self.inner.left
    }

    /// Get the index of the generator on the _right_ of this face.
    /// Returns `None` if if this is a boundary face (i.e. obtained by clipping
    /// a Voronoi cell with the _simulation volume_).
    pub fn right(&self) -> Option<usize> {
        self.inner.right
    }

    /// Update the index of the generator on the _left_ of this face.
    pub fn set_left(&mut self, left: usize) {
        self.inner.left = left;
    }

    /// Update the index of the generator on the _right_ of this face.
    pub fn set_right(&mut self, right: usize) {
        self.inner.right = Some(right);
    }

    /// Get the area of this face.
    pub fn area(&self) -> f64 {
        self.inner.integral.area
    }

    /// Get the position of the centroid of this face.
    pub fn centroid(&self) -> DVec3 {
        self.inner.integral.centroid
    }

    /// Get a normal vector to this face, pointing away from the _left_
    /// generator.
    pub fn normal(&self) -> DVec3 {
        self.inner.integral.normal
    }

    /// Get the shift vector (if any) to apply to the generator to the right of
    /// this face to bring it to the reference frame of this face.
    /// Can only be `Some` for periodic Voronoi tessellations.
    pub fn shift(&self) -> Option<DVec3> {
        self.inner.shift
    }

    /// Update the shift vector associated with this face
    pub fn set_shift(&mut self, shift: DVec3) {
        self.inner.shift = Some(shift)
    }

    /// Whether this is a face between a particle and a periodic boundary neighbour
    pub fn is_periodic(&self) -> bool {
        // Periodically wrapping faces must have shift set
        self.inner.shift.is_some()
    }

    /// Whether this is a boundary face
    pub fn is_boundary(&self) -> bool {
        self.inner.right.is_none()
    }
}
