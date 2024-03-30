use glam::DVec3;

use crate::{
    geometry::{in_sphere_test, intersect_planes, Plane},
    simple_cycle::SimpleCycle,
};

use super::{Dimensionality, Generator};

#[derive(Clone)]
pub(super) struct DualVertex {
    pub repr: (usize, usize, usize),
    circumradius2: f64,
}

impl DualVertex {
    fn new(
        _0: usize,
        _1: usize,
        _2: usize,
        loc: DVec3,
        neighbours: &[Neighbour],
        dimensionality: Dimensionality,
    ) -> Self {
        let mut circumcenter =
            intersect_planes(&neighbours[_0].plane, &neighbours[_1].plane, &neighbours[_2].plane);
        match dimensionality {
            Dimensionality::OneD => {
                circumcenter.y = 0.;
                circumcenter.z = 0.;
            }
            Dimensionality::TwoD => circumcenter.z = 0.,
            Dimensionality::ThreeD => (),
        }
        Self {
            repr: (_0, _1, _2),
            circumradius2: (loc - circumcenter).length_squared(),
        }
    }

    fn clipped_by(&self, loc: DVec3, new_neighbour: DVec3, neighbours: &[Neighbour]) -> bool {
        in_sphere_test(
            loc,
            neighbours[self.repr.0].loc,
            neighbours[self.repr.1].loc,
            neighbours[self.repr.2].loc,
            new_neighbour,
        ) < 0.
    }
}

#[derive(Clone)]
pub(super) struct Neighbour {
    pub loc: DVec3,
    pub idx: Option<usize>,
    pub shift: Option<DVec3>,
    pub plane: Plane,
}

impl Neighbour {
    fn new(loc: DVec3, idx: Option<usize>, shift: Option<DVec3>, gen_loc: DVec3) -> Self {
        Neighbour {
            loc,
            idx,
            shift,
            plane: Plane::new((loc - gen_loc).normalize(), 0.5 * (loc + gen_loc)),
        }
    }
}

#[derive(Clone)]
pub(super) struct ConvexCell {
    pub loc: DVec3,
    pub idx: usize,
    pub neighbours: Vec<Neighbour>,
    pub vertices: Vec<DualVertex>,
    pub safety_radius2: f64,
    pub boundary: SimpleCycle,
    pub dimensionality: Dimensionality,
}

impl ConvexCell {
    /// Initialize each Voronoi cell as the bounding box of the simulation
    /// volume.
    pub(super) fn init(
        loc: DVec3,
        idx: usize,
        mut anchor: DVec3,
        mut width: DVec3,
        periodic: bool,
        dimensionality: Dimensionality,
    ) -> Self {
        if periodic {
            anchor.x -= width.x;
            width.x *= 3.;
            if let Dimensionality::TwoD | Dimensionality::ThreeD = dimensionality {
                anchor.y -= width.y;
                width.y *= 3.;
            };
            if let Dimensionality::ThreeD = dimensionality {
                anchor.z -= width.z;
                width.z *= 3.;
            }
        }
        let opposite = anchor + width;

        let neighbours = vec![
            Neighbour::new(DVec3::new(2. * anchor.x - loc.x, loc.y, loc.z), None, None, loc),
            Neighbour::new(DVec3::new(2. * opposite.x - loc.x, loc.y, loc.z), None, None, loc),
            Neighbour::new(DVec3::new(loc.x, 2. * anchor.y - loc.y, loc.z), None, None, loc),
            Neighbour::new(DVec3::new(loc.x, 2. * opposite.y - loc.y, loc.z), None, None, loc),
            Neighbour::new(DVec3::new(loc.x, loc.y, 2. * anchor.z - loc.z), None, None, loc),
            Neighbour::new(DVec3::new(loc.x, loc.y, 2. * opposite.z - loc.z), None, None, loc),
        ];
        let vertices = vec![
            DualVertex::new(2, 5, 0, loc, &neighbours, dimensionality),
            DualVertex::new(5, 3, 0, loc, &neighbours, dimensionality),
            DualVertex::new(1, 5, 2, loc, &neighbours, dimensionality),
            DualVertex::new(5, 1, 3, loc, &neighbours, dimensionality),
            DualVertex::new(4, 2, 0, loc, &neighbours, dimensionality),
            DualVertex::new(4, 0, 3, loc, &neighbours, dimensionality),
            DualVertex::new(2, 4, 1, loc, &neighbours, dimensionality),
            DualVertex::new(4, 3, 1, loc, &neighbours, dimensionality),
        ];
        let mut cell = ConvexCell {
            loc,
            idx,
            boundary: SimpleCycle::new(neighbours.len()),
            neighbours,
            vertices,
            safety_radius2: 0.,
            dimensionality,
        };
        cell.update_safety_radius();
        cell
    }

    /// Build the Convex cell by repeatedly intersecting it with the appropriate
    /// half spaces
    pub(super) fn build(
        &mut self,
        generators: &[Generator],
        mut nearest_neighbours: Box<dyn Iterator<Item = (usize, Option<DVec3>)> + '_>,
        dimensionality: Dimensionality,
    ) {
        // skip the first nearest neighbour (will be this cell)
        assert_eq!(
            nearest_neighbours.next().expect("Nearest neighbours cannot be empty!").0,
            self.idx,
            "First nearest neighbour should be the generator itself!"
        );
        // now loop over the nearest neighbours and clip this cell until the safety
        // radius is reached
        for (idx, shift) in nearest_neighbours {
            let generator = generators[idx];
            let ngb_loc;
            if let Some(shift) = shift {
                ngb_loc = generator.loc() + shift;
            } else {
                ngb_loc = generator.loc();
            }
            let dx = self.loc - ngb_loc;
            let dist = dx.length_squared();
            assert!(dist.is_finite() && dist > 0.0, "Degenerate point set!");
            if self.safety_radius2 < dist {
                return;
            }
            self.clip_by(ngb_loc, idx, shift, dimensionality);
        }
    }

    fn clip_by(
        &mut self,
        ngb_loc: DVec3,
        ngb_idx: usize,
        shift: Option<DVec3>,
        dimensionality: Dimensionality,
    ) {
        // loop over vertices and remove the ones clipped by p
        let mut i = 0;
        let mut num_v = self.vertices.len();
        let mut num_r = 0;
        while i < num_v {
            if self.vertices[i].clipped_by(self.loc, ngb_loc, &self.neighbours) {
                num_v -= 1;
                num_r += 1;
                self.vertices.swap(i, num_v);
            } else {
                i += 1;
            }
        }

        // Were any vertices clipped?
        if num_r > 0 {
            let new_idx = self.neighbours.len();
            self.neighbours.push(Neighbour::new(ngb_loc, Some(ngb_idx), shift, self.loc));
            self.boundary.grow();
            // Compute the boundary of the (dual) topological triangulated disk around the
            // vertices to be removed.
            Self::compute_boundary(&mut self.boundary, &mut self.vertices[num_v..]);
            let mut boundary = self.boundary.iter().take(self.boundary.len + 1);
            // finally we can *realy* remove the vertices.
            self.vertices.truncate(num_v);
            // Add new vertices constructed from the new clipping plane and the boundary
            let mut cur = boundary.next().expect("Boundary contains at least 3 elements");
            for next in boundary {
                self.vertices.push(DualVertex::new(
                    cur,
                    next,
                    new_idx,
                    self.loc,
                    &self.neighbours,
                    dimensionality,
                ));
                cur = next;
            }
            self.update_safety_radius();
        }
    }

    fn update_safety_radius(&mut self) {
        let mut max_circumradius2 = f64::NEG_INFINITY;
        for vertex in self.vertices.iter() {
            max_circumradius2 = max_circumradius2.max(vertex.circumradius2);
        }
        self.safety_radius2 = 4. * max_circumradius2;
    }

    fn compute_boundary(boundary: &mut SimpleCycle, vertices: &mut [DualVertex]) {
        boundary.init(vertices[0].repr.0, vertices[0].repr.1, vertices[0].repr.2);

        for i in 1..vertices.len() {
            // Look for a suitable next vertex to extend the boundary
            let mut idx = i;
            loop {
                assert!(idx < vertices.len(), "No suitable vertex found to extend boundary!");
                let vertex = &vertices[idx];
                match boundary.try_extend(vertex.repr.0, vertex.repr.1, vertex.repr.2) {
                    Ok(()) => {
                        if idx > i {
                            vertices.swap(i, idx);
                        }
                        break;
                    }
                    Err(()) => idx += 1,
                }
            }
        }
    }
}
