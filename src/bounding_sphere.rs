use std::collections::HashSet;

use glam::DVec3;

use crate::geometry::Sphere;

pub(crate) trait BoundingSphereSolver {
    fn bounding_sphere(points: &[DVec3]) -> Sphere;
}

pub struct Welzl;

impl Welzl {
    /// Recursively compute the bounding sphere for the given points with the given boundary points on its surface.
    fn bounding_sphere_recursive(points: &mut Vec<DVec3>, boundary: &mut Vec<DVec3>) -> Sphere {
        if points.is_empty() || boundary.len() == 4 {
            // base case: No other points left or maximal number of boundary points
            return Sphere::from_boundary_points(&boundary);
        }

        // Pop test point from points
        let point = points
            .pop()
            .expect("Points cannot be empty at this point (see base case)");

        // recurse: find solution of remaining points
        let mut solution = Self::bounding_sphere_recursive(points, boundary);
        if !solution.contains(point) {
            // The proposed solution does not contain our test point, add it to the boundary and try again.
            boundary.push(point);
            solution = Self::bounding_sphere_recursive(points, boundary);
            // revert push to not mess up recursion
            boundary.pop();
        }

        // revert pop to not mess up recursion.
        points.push(point);
        solution
    }
}

impl BoundingSphereSolver for Welzl {
    fn bounding_sphere(points: &[DVec3]) -> Sphere {
        let mut boundary = vec![];
        let mut points = points.to_vec();
        Self::bounding_sphere_recursive(&mut points, &mut boundary)
    }
}

pub(crate) struct EPOS6;

/// See: Extremal Points Optimal Sphere, Larsson 2008 (https://ep.liu.se/ecp/034/009/ecp083409.pdf)
impl BoundingSphereSolver for EPOS6 {
    fn bounding_sphere(points: &[DVec3]) -> Sphere {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut idx_min = [0, 0, 0];
        let mut idx_max = [0, 0, 0];
        for (idx, point) in points.iter().enumerate() {
            if point.x < min.x {
                idx_min[0] = idx;
                min.x = point.x;
            }
            if max.x < point.x {
                idx_max[0] = idx;
                max.x = point.x;
            }
            if point.y < min.y {
                idx_min[1] = idx;
                min.y = point.y;
            }
            if max.y < point.y {
                idx_max[1] = idx;
                max.y = point.y;
            }
            if point.z < min.z {
                idx_min[2] = idx;
                min.z = point.z;
            }
            if max.z < point.z {
                idx_max[2] = idx;
                max.z = point.z;
            }
        }

        // Get sphere from extremal points
        let mut extremal_points = HashSet::new();
        extremal_points.extend(idx_min.into_iter());
        extremal_points.extend(idx_max.into_iter());
        let extremal_points = extremal_points
            .into_iter()
            .map(|i| points[i])
            .collect::<Vec<_>>();
        let mut sphere = Welzl::bounding_sphere(&extremal_points);

        // Extend sphere if necessary
        for point in points {
            sphere = sphere.extend(*point);
        }

        sphere
    }
}

#[cfg(test)]
mod test {
    use glam::DVec3;

    use super::{BoundingSphereSolver, Welzl, EPOS6};

    #[test]
    fn test_bound() {
        let points = [
            DVec3 {
                x: 0.415890403454624,
                y: 0.3575877823827407,
                z: 0.04615071873700416,
            },
            DVec3 {
                x: 0.5904207636262682,
                y: 0.3881875556525609,
                z: 0.431685545623832,
            },
            DVec3 {
                x: 0.08615522660010921,
                y: 0.1745114522139155,
                z: 0.04833203182100243,
            },
            DVec3 {
                x: 0.4017866036409746,
                y: 0.9398328313317894,
                z: 0.702150646647231,
            },
            DVec3 {
                x: 0.8696231553786411,
                y: 0.2422068991399332,
                z: 0.832436770396769,
            },
            DVec3 {
                x: 0.9585318105009617,
                y: 0.3793309254294106,
                z: 0.6561882124333864,
            },
            DVec3 {
                x: 0.04774867266809413,
                y: 0.5902320414959258,
                z: 0.28298125411550457,
            },
            DVec3 {
                x: 0.31377085869583643,
                y: 0.13009140821508602,
                z: 0.8451054417972689,
            },
            DVec3 {
                x: 0.24658811345108766,
                y: 0.05225420135982506,
                z: 0.8547081384093004,
            },
            DVec3 {
                x: 0.8646915959398506,
                y: 0.7140367933093176,
                z: 0.11694464267625437,
            },
        ];

        let sphere = Welzl::bounding_sphere(&points);
        for point in &points {
            assert!(sphere.contains(*point));
        }

        let sphere_approx = EPOS6::bounding_sphere(&points);
        for point in &points {
            assert!(sphere_approx.contains(*point));
        }
    }

    #[test]
    fn test_bound_2d() {
        let points = [
            DVec3 {
                x: 0.415890403454624,
                y: 0.3575877823827407,
                z: 0.,
            },
            DVec3 {
                x: 0.5904207636262682,
                y: 0.3881875556525609,
                z: 0.,
            },
            DVec3 {
                x: 0.05615522660010921,
                y: 0.2045114522169155,
                z: 0.,
            },
            DVec3 {
                x: 0.4017866036409746,
                y: 0.9398328313317894,
                z: 0.,
            },
            DVec3 {
                x: 0.8696231553786411,
                y: 0.2422068991399332,
                z: 0.,
            },
            DVec3 {
                x: 0.9385318105009617,
                y: 0.0593309254294106,
                z: 0.,
            },
            DVec3 {
                x: 0.04774867266809413,
                y: 0.5902320414959258,
                z: 0.,
            },
            DVec3 {
                x: 0.31377085869583643,
                y: 0.13009140821508602,
                z: 0.,
            },
            DVec3 {
                x: 0.24658811345108766,
                y: 0.05225420135982506,
                z: 0.,
            },
            DVec3 {
                x: 0.8646915959398506,
                y: 0.7140367933093176,
                z: 0.,
            },
        ];

        let sphere = Welzl::bounding_sphere(&points);
        for point in &points {
            assert!(sphere.contains(*point));
        }

        let sphere_approx = EPOS6::bounding_sphere(&points);
        for point in &points {
            assert!(sphere_approx.contains(*point));
        }
    }
}
