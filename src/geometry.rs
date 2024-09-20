//! A few general-purpose geometry functions and structs,
//! which might also be useful for users of this library.

#[cfg(feature = "dashu")]
use dashu::Integer;
use glam::{DMat3, DMat4, DVec3, DVec4};
#[cfg(feature = "ibig")]
use ibig::IBig as Integer;
#[cfg(feature = "malachite")]
use malachite_base::num::arithmetic::traits::Sign;
#[cfg(feature = "malachite")]
use malachite_nz::integer::Integer;
#[cfg(feature = "num_bigint")]
use num_bigint::{BigInt as Integer, Sign};
#[cfg(feature = "rug")]
use rug::Integer;
#[cfg(feature = "malachite")]
use std::cmp::Ordering;

/// A simple plane struct.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Normal vector
    pub n: DVec3,
    /// Point on the plane
    pub p: DVec3,
}

impl Plane {
    /// Create a plane from a normal vector and a point on the plane.
    pub fn new(n: DVec3, p: DVec3) -> Self {
        Self { n, p }
    }

    /// Project a point onto plane.
    pub fn project_onto(&self, point: DVec3) -> DVec3 {
        point + (self.p - point).project_onto(self.n)
    }

    /// Project a point on the intersection of two planes.
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
    pub(crate) fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        // interval radius of projection of AABB on planes normal
        let r = self.n.abs().dot(aabb.extent);
        // distance from box center to plane
        let d = (self.p - aabb.center).dot(self.n).abs();

        d <= r * (1. + 1e-10)
    }
}

/// Calculate the intersection of 3 planes.
///
/// See <https://mathworld.wolfram.com/Plane-PlaneIntersection.html>.
pub fn intersect_planes(p0: &Plane, p1: &Plane, p2: &Plane) -> DVec3 {
    let det = DMat3::from_cols(p0.n, p1.n, p2.n).determinant();
    assert!(det != 0., "Degenerate 3-plane intersection!");

    (p0.p.dot(p0.n) * p1.n.cross(p2.n)
        + p1.p.dot(p1.n) * p2.n.cross(p0.n)
        + p2.p.dot(p2.n) * p0.n.cross(p1.n))
        / det
}

/// Compute the signed volume of a oriented tetrahedron.
///
/// The volume is positive if `v0`, `v1` and `v2` are ordered counterclockwise,
/// as seen from v3.
pub fn signed_volume_tet(v0: DVec3, v1: DVec3, v2: DVec3, v3: DVec3) -> f64 {
    let v01 = v1 - v0;
    let v02 = v2 - v0;
    let v03 = v3 - v0;

    DMat3::from_cols(v01, v02, v03).determinant() / 6.
}

/// Calculates the signed area of the ground face `v0`, `v1`, `v2` of the
/// tetrahedron with top `t`.
///
/// The area is positive if the the vertices are ordered counterclockwise
/// as seen from t.
pub fn signed_area_tri(v0: DVec3, v1: DVec3, v2: DVec3, t: DVec3) -> f64 {
    // Normal vector with the area of the ground face as length
    let n = 0.5 * (v1 - v0).cross(v2 - v0);
    let sign = (t - v0).dot(n).signum();
    n.length() * sign
}

/// A simple sphere struct.
#[derive(Clone)]
pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
}

impl Sphere {
    /// Zero sized sphere at the origin.
    pub const EMPTY: Sphere = Sphere {
        center: DVec3::ZERO,
        radius: 0.,
    };

    pub fn new(center: DVec3, radius: f64) -> Self {
        Self { center, radius }
    }

    /// Create the smallest sphere through 2, 3 or 4 boundary points.
    /// When 0 or 1 points are given, an empty sphere is returned.
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

    /// Create a sphere through two given points with center the midpoint
    /// between the two points.
    pub fn from_two_points(a: DVec3, b: DVec3) -> Self {
        Self::new(0.5 * (a + b), 0.5 * a.distance(b))
    }

    /// Circumscribes sphere through three given points with center on the plane
    /// spanned by the points. I.e. the smallest sphere trough the three points.
    ///
    /// See <https://www.wikiwand.com/en/Circumscribed_circle#Higher_dimensions>.
    pub fn from_three_points(a: DVec3, b: DVec3, c: DVec3) -> Sphere {
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

    /// Circumscribed sphere through four points.
    ///
    /// See <https://mathworld.wolfram.com/Circumsphere.html>.
    pub fn from_four_points(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> Sphere {
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

    /// Extend this sphere to include `x`, if necessary.
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

/// Test whether `v` lies inside or outside the circumsphere around `a`, `b`,
/// `c` and `d`. See Springel (2010) eq. (3).
///
/// The result is negative when `v` lies inside and positive when `v` lies
/// outside the circumsphere. We work in relative coordinates to simplify the
/// determinant to a 4×4 determinant.
pub(crate) fn in_sphere_test(a: DVec3, b: DVec3, c: DVec3, d: DVec3, v: DVec3) -> f64 {
    let b = (b - a).extend((b - a).length_squared());
    let c = (c - a).extend((c - a).length_squared());
    let d = (d - a).extend((d - a).length_squared());
    let v = (v - a).extend((v - a).length_squared());

    DMat4::from_cols(b, c, d, v).determinant()
}

macro_rules! big_int {
    ($a:expr, $b:expr) => {{
        let mut big_int_diff = [
            Integer::from($a[0] - $b[0]),
            Integer::from($a[1] - $b[1]),
            Integer::from($a[2] - $b[2]),
            Integer::default(),
        ];
        let mut norm2 = Integer::default();
        norm2 += &big_int_diff[0] * &big_int_diff[0];
        norm2 += &big_int_diff[1] * &big_int_diff[1];
        norm2 += &big_int_diff[2] * &big_int_diff[2];
        big_int_diff[3] = norm2;
        big_int_diff
    }};
}

macro_rules! big_int_det2x2 {
    ($a:expr, $b:expr, $c:expr, $d:expr, $det:expr) => {{
        $det = Integer::default();
        $det += &$a * &$d;
        $det -= &$b * &$c;
    }};
}

macro_rules! big_int_det3x3 {
    ($a0:expr, $a1:expr, $a2:expr, $b0:expr, $b1:expr, $b2:expr, $c0:expr, $c1:expr, $c2:expr, $tmp:expr, $det:expr) => {
        $det = Integer::default();
        big_int_det2x2!($b1, $b2, $c1, $c2, $tmp);
        $det += &$a0 * &$tmp;
        big_int_det2x2!($b0, $b2, $c0, $c2, $tmp);
        $det -= &$a1 * &$tmp;
        big_int_det2x2!($b0, $b1, $c0, $c1, $tmp);
        $det += &$a2 * &$tmp;
    };
}

/// Test whether `v` lies inside or outside the circumsphere around `a`, `b`,
/// `c` and `d` using exact integer arithmetic.
pub(crate) fn in_sphere_test_exact(a: &[i64], b: &[i64], c: &[i64], d: &[i64], v: &[i64]) -> f64 {
    let b = big_int!(b, a);
    let c = big_int!(c, a);
    let d = big_int!(d, a);
    let v = big_int!(v, a);

    // We need to compute the sign of the 4×4 determinant with b, c, d, v as rows.
    let mut determinant = Integer::default();

    // Let's do it in 4 steps by developing over the last column
    // Step 1 (b-row):
    let mut tmp1: Integer;
    let mut det: Integer;
    big_int_det3x3!(c[0], c[1], c[2], d[0], d[1], d[2], v[0], v[1], v[2], tmp1, det);
    determinant += &b[3] * &det;
    // Step 2 (c-row)
    big_int_det3x3!(b[0], b[1], b[2], d[0], d[1], d[2], v[0], v[1], v[2], tmp1, det);
    determinant -= &c[3] * &det;
    // Step 3 (d-row)
    big_int_det3x3!(b[0], b[1], b[2], c[0], c[1], c[2], v[0], v[1], v[2], tmp1, det);
    determinant += &d[3] * &det;
    // Step 4 (v-row)
    big_int_det3x3!(b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], tmp1, det);
    determinant -= &v[3] * &det;

    #[cfg(any(feature = "dashu", feature = "ibig", feature = "rug"))]
    let result = determinant.signum().to_f64();
    #[cfg(feature = "dashu")]
    let result = result.value();
    #[cfg(feature = "malachite")]
    let result = match determinant.sign() {
        Ordering::Less => -1.0,
        Ordering::Equal => 0.0,
        Ordering::Greater => 1.0,
    };
    #[cfg(feature = "num_bigint")]
    let result = match determinant.sign() {
        Sign::Minus => -1.0,
        Sign::NoSign => 0.0,
        Sign::Plus => 1.0,
    };

    result
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct Aabb {
    min: DVec3,
    max: DVec3,
    center: DVec3,
    extent: DVec3,
}

#[allow(dead_code)]
impl Aabb {
    pub const EMPTY: Self = Self {
        min: DVec3::ZERO,
        max: DVec3::ZERO,
        center: DVec3::ZERO,
        extent: DVec3::ZERO,
    };

    pub fn new(min: DVec3, max: DVec3) -> Self {
        let center = 0.5 * (min + max);
        let extent = max - center;
        Aabb {
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
mod tests {
    use glam::DVec3;

    use crate::geometry::{signed_area_tri, signed_volume_tet};

    use super::{Aabb, Plane, Sphere};

    #[test]
    fn test_signed_volume() {
        let v0 = DVec3::ZERO;
        let v1 = DVec3::X;
        let v2 = DVec3::Y;
        let v3 = DVec3::Z;

        let volume = signed_volume_tet(v0, v1, v2, v3);
        assert_eq!(volume, 1. / 6.);

        let volume2 = signed_volume_tet(v0, v2, v1, v3);
        assert_eq!(volume, -volume2);
    }

    #[test]
    fn test_signed_area() {
        let v0 = DVec3::ZERO;
        let v1 = DVec3::X;
        let v2 = DVec3::Y;
        let t = DVec3::Z;

        let area = signed_area_tri(v0, v1, v2, t);
        assert_eq!(area, 0.5);

        let area2 = signed_area_tri(v0, v2, v1, t);
        assert_eq!(area, -area2);

        let t = DVec3::Z
            + DVec3 {
                x: 10.,
                y: 10.,
                z: 0.,
            };
        let area3 = signed_area_tri(v0, v1, v2, t);
        assert_eq!(area, area3)
    }

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
        let aabb = Aabb::new(DVec3::NEG_ONE, DVec3::X);
        let n = DVec3 {
            x: 1.,
            y: 2.,
            z: 1.,
        }
        .normalize();
        let plane = Plane::new(
            n,
            DVec3 {
                x: 1.1,
                y: 0.,
                z: 0.,
            },
        );
        assert!(!plane.intersects_aabb(&aabb));
        let plane = Plane::new(
            n,
            DVec3 {
                x: 0.9,
                y: 0.,
                z: 0.,
            },
        );
        assert!(plane.intersects_aabb(&aabb));
    }
}
