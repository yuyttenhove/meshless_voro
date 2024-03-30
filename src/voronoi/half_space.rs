use glam::DVec3;

use crate::geometry::Plane;

use super::Generator;

#[derive(Clone, Debug)]
pub(super) struct HalfSpace {
    pub plane: Plane,
    d: f64,
    errb: f64,
    pub right_idx: Option<usize>,
    pub shift: Option<DVec3>,
}

impl HalfSpace {
    const EPSILON: f64 = 1e-13;

    pub fn new(n: DVec3, p: DVec3, right_idx: Option<usize>, shift: Option<DVec3>) -> Self {
        HalfSpace {
            plane: Plane::new(n, p),
            d: n.dot(p),
            errb: Self::EPSILON * n.abs().dot(p.abs()),
            right_idx,
            shift,
        }
    }

    /// Determine on which side of the plane a given `vertex` lies.
    /// Returns `1.` if the `vertex` lies on the positive half space (direction
    /// of normal), `-1.` when the `vertex` lies on the negative half space
    /// and `0.` when a more precise test is needed
    pub fn clip(&self, vertex: DVec3) -> f64 {
        let clip = self.plane.n.dot(vertex) - self.d;
        if clip.abs() < self.errb {
            0.
        } else {
            clip.signum()
        }
    }

    pub fn normal(&self) -> DVec3 {
        self.plane.n
    }

    pub fn right_loc(&self, left_idx: usize, generators: &[Generator]) -> DVec3 {
        if let Some(right_idx) = self.right_idx {
            let mut loc = generators[right_idx].loc();
            if let Some(shift) = self.shift {
                loc += shift;
            }
            loc
        } else {
            let left_loc = generators[left_idx].loc();
            let projected = self.plane.project_onto(left_loc);
            2. * projected - left_loc
        }
    }
}
