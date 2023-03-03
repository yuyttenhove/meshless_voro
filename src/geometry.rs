use glam::{DMat3, DMat4, DVec3, DVec4};

#[derive(Clone)]
pub(crate) struct Plane {
    pub n: DVec3,
    pub p: DVec3,
}

impl Plane {
    /// Create a plane from a normal vector and a point on the plane.
    pub(crate) fn new(n: DVec3, p: DVec3) -> Self {
        Self { n, p }
    }

    /// Project point onto plane.
    pub fn project_onto(&self, point: DVec3) -> DVec3 {
        point + (self.p - point).project_onto(self.n)
    }

    /// Project the a point on the intersection of two planes.
    pub fn project_onto_intersection(&self, other: &Self, point: DVec3) -> DVec3 {
        // first create a plane through the point perpendicular to both planes
        let p_perp = Plane::new(self.n.cross(other.n), point);

        // The projection is the intersection of the planes self, other and p_perp
        intersect_planes(self, other, &p_perp)
    }

    #[allow(dead_code)]
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        sphere.contains(self.project_onto(sphere.center))
    }

    #[allow(dead_code)]
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        // interval radius of projection of AABB on planes normal
        let r = self.n.abs().dot(aabb.extent);
        // distance from box center to plane
        let d = (self.p - aabb.center).dot(self.n).abs();

        return d <= r * (1. + 1e-10);
    }
}

/// Calculate the intersection of 3 planes.
/// see: https://mathworld.wolfram.com/Plane-PlaneIntersection.html
pub(crate) fn intersect_planes(p0: &Plane, p1: &Plane, p2: &Plane) -> DVec3 {
    let det = DMat3::from_cols(p0.n, p1.n, p2.n).determinant();
    assert!(det != 0., "Degenerate 3-plane intersection!");

    (p0.p.dot(p0.n) * p1.n.cross(p2.n)
        + p1.p.dot(p1.n) * p2.n.cross(p0.n)
        + p2.p.dot(p2.n) * p0.n.cross(p1.n))
        / det
}

#[derive(Clone)]
pub(crate) struct Sphere {
    pub center: DVec3,
    pub radius: f64,
}

impl Sphere {
    pub const EMPTY: Sphere = Sphere {
        center: DVec3::ZERO,
        radius: 0.,
    };

    pub(crate) fn new(center: DVec3, radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn from_boundary_points(points: &[DVec3]) -> Self {
        match points.len() {
            0 | 1 => Self::EMPTY,
            2 => Self::from_two_points(points[0], points[1]),
            3 => Self::from_three_points(points[0], points[1], points[2]),
            4 => Self::from_four_points(points[0], points[1], points[2], points[3]),
            _ => panic!(
                "Invalid number of boundary points for sphere construction: {:}!",
                points.len()
            ),
        }
    }

    /// Create a sphere through two given points with center the midpoint between the two points
    fn from_two_points(a: DVec3, b: DVec3) -> Self {
        Self::new(0.5 * (a + b), 0.5 * a.distance(b))
    }

    /// Circumscribes sphere through 3 given points with center on the plane spanned by the points
    /// See: https://www.wikiwand.com/en/Circumscribed_circle#Higher_dimensions
    fn from_three_points(a: DVec3, b: DVec3, c: DVec3) -> Sphere {
        let a = a - c;
        let b = b - c;

        let a_2 = a.length_squared();
        let b_2 = b.length_squared();
        let a_cross_b = a.cross(b);
        let one_over_a_cross_b_2 = 1. / a_cross_b.length_squared();
        let one_over_sin_theta_2 = (a_2 * b_2) * one_over_a_cross_b_2;

        let radius = 0.5 * (one_over_sin_theta_2 * (a - b).length_squared()).sqrt();
        let center = 0.5 * (a_2 * b - b_2 * a).cross(a_cross_b) * one_over_a_cross_b_2 + c;

        Self::new(center, radius)
    }

    /// Circumscribed sphere through 4 points. See: https://mathworld.wolfram.com/Circumsphere.html
    fn from_four_points(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> Sphere {
        let x = DVec4 {
            x: a.x,
            y: b.x,
            z: c.x,
            w: d.x,
        };
        let y = DVec4 {
            x: a.y,
            y: b.y,
            z: c.y,
            w: d.y,
        };
        let z = DVec4 {
            x: a.z,
            y: b.z,
            z: c.z,
            w: d.z,
        };
        let n2 = x * x + y * y + z * z;

        let a = DMat4::from_cols(x, y, z, DVec4::ONE).determinant();
        let d_x = DMat4::from_cols(n2, y, z, DVec4::ONE).determinant();
        let d_y = -DMat4::from_cols(n2, x, z, DVec4::ONE).determinant();
        let d_z = DMat4::from_cols(n2, x, y, DVec4::ONE).determinant();
        let c = DMat4::from_cols(n2, x, y, z).determinant();

        let one_over_2a = 0.5 / a;
        let radius = (d_x * d_x + d_y * d_y + d_z * d_z - 4. * a * c).sqrt() * one_over_2a.abs();
        let center = DVec3 {
            x: d_x,
            y: d_y,
            z: d_z,
        } * one_over_2a;
        Self::new(center, radius)
    }

    /// Extend this sphere to include x if necessary
    pub fn extend(mut self, x: DVec3) -> Self {
        if !self.contains(x) {
            let opposite = self.center - self.radius * (x - self.center).normalize();
            self.center = 0.5 * (opposite + x);
            self.radius = self.center.distance(x);
        }
        self
    }

    pub fn contains(&self, x: DVec3) -> bool {
        self.radius > 0.
            && x.distance_squared(self.center) <= self.radius * self.radius * (1. + 1e-10)
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct AABB {
    min: DVec3,
    max: DVec3,
    center: DVec3,
    extent: DVec3,
}

#[allow(dead_code)]
impl AABB {
    pub const EMPTY: Self = Self {
        min: DVec3::ZERO,
        max: DVec3::ZERO,
        center: DVec3::ZERO,
        extent: DVec3::ZERO,
    };

    pub fn new(min: DVec3, max: DVec3) -> Self {
        let center = 0.5 * (min + max);
        let extent = max - center;
        AABB {
            min,
            max,
            center,
            extent,
        }
    }

    pub fn from_points(points: &[DVec3]) -> Self {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for point in points {
            min = min.min(*point);
            max = max.max(*point);
        }
        Self::new(min, max)
    }
}

#[cfg(test)]
mod test {
    use glam::DVec3;

    use super::{Plane, Sphere, AABB};

    #[test]
    fn test_sphere_two_points() {
        let a = DVec3::ZERO;
        let b = DVec3::ONE + DVec3::X;
        let sphere = Sphere::from_two_points(a, b);
        assert_eq!(sphere.center.x, 1.);
        assert_eq!(sphere.center.y, 0.5);
        assert_eq!(sphere.center.z, 0.5);
        assert_eq!(sphere.radius, 0.5 * 6f64.sqrt())
    }

    #[test]
    fn test_sphere_three_points() {
        let a = DVec3::ZERO;
        let b = DVec3::ONE;
        let c = DVec3::X;
        let sphere = Sphere::from_three_points(a, b, c);
        assert_eq!(sphere.center.x, 0.5);
        assert_eq!(sphere.center.y, 0.5);
        assert_eq!(sphere.center.z, 0.5);
        assert_eq!(sphere.radius, 0.5 * 3f64.sqrt())
    }

    #[test]
    fn test_sphere_four_points() {
        let a = DVec3::ZERO;
        let b = DVec3::X;
        let c = DVec3::Y;
        let d = DVec3::Z;
        let sphere = Sphere::from_four_points(a, b, c, d);
        assert_eq!(sphere.center.x, 0.5);
        assert_eq!(sphere.center.y, 0.5);
        assert_eq!(sphere.center.z, 0.5);
        assert_eq!(sphere.radius, 0.5 * 3f64.sqrt())
    }

    #[test]
    fn test_extend() {
        let sphere = Sphere::new(DVec3::ZERO, 1.);
        let x = DVec3::X * 2.;
        let sphere = sphere.extend(x);
        assert_eq!(sphere.radius, 1.5);
        assert_eq!(sphere.center.x, 0.5);
        assert_eq!(sphere.center.y, 0.);
        assert_eq!(sphere.center.z, 0.);
    }

    #[test]
    fn test_sphere_plane_intersection() {
        let sphere = Sphere::new(DVec3::ONE, 1.5);
        let n = DVec3 {
            x: 1.,
            y: 0.5,
            z: 0.1,
        }
        .normalize();
        let plane = Plane::new(
            n,
            DVec3 {
                x: 2.5,
                y: 2.,
                z: 0.,
            },
        );
        assert!(!plane.intersects_sphere(&sphere));
        let plane = Plane::new(
            n,
            DVec3 {
                x: 1.,
                y: 1.,
                z: 0.,
            },
        );
        assert!(plane.intersects_sphere(&sphere));
    }

    #[test]
    fn test_aabb_plane_intersection() {
        let aabb = AABB::new(DVec3::NEG_ONE, DVec3::X);
        let n = DVec3 { x: 1., y: 2., z: 1. }.normalize();
        let plane = Plane::new(n, DVec3 { x: 1.1, y: 0., z: 0. });
        assert!(!plane.intersects_aabb(&aabb));
        let plane = Plane::new(n, DVec3 { x: 0.9, y: 0., z: 0. });
        assert!(plane.intersects_aabb(&aabb));
    }
}
