use glam::DVec3;

use crate::geometry::Plane;

#[derive(Clone, Debug)]
pub(super) struct HalfSpace {
    pub plane: Plane,
    d: f64,
    pub right_idx: Option<usize>,
    pub shift: Option<DVec3>,
}

impl HalfSpace {
    pub fn new(n: DVec3, p: DVec3, right_idx: Option<usize>, shift: Option<DVec3>) -> Self {
        HalfSpace {
            plane: Plane::new(n, p),
            d: n.dot(p),
            right_idx,
            shift,
        }
    }

    /// Whether a vertex is clipped by this half space
    pub fn clips(&self, vertex: DVec3) -> bool {
        self.plane.n.dot(vertex) < self.d
    }

    pub fn normal(&self) -> DVec3 {
        self.plane.n
    }
}
