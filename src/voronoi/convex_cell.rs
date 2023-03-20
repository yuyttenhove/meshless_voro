use glam::DVec3;

use crate::{geometry::intersect_planes, simple_cycle::SimpleCycle};

use super::{
    boundary::SimulationBoundary, convex_cell_alternative::ConvexCell as ConvexCellAlternative,
    half_space::HalfSpace, Dimensionality, Generator,
};

#[derive(Clone)]
pub(super) struct Vertex {
    pub loc: DVec3,
    pub dual: (usize, usize, usize),
    pub radius2: f64,
}

impl Vertex {
    fn from_dual(
        i: usize,
        j: usize,
        k: usize,
        half_spaces: &[HalfSpace],
        gen_loc: DVec3,
        dimensionality: Dimensionality,
    ) -> Self {
        let loc = intersect_planes(
            &half_spaces[i].plane,
            &half_spaces[j].plane,
            &half_spaces[k].plane,
        );
        let d_loc = match dimensionality {
            Dimensionality::Dimensionality1D => DVec3::new(loc.x, 0., 0.),
            Dimensionality::Dimensionality2D => DVec3::new(loc.x, loc.y, 0.),
            Dimensionality::Dimensionality3D => loc,
        };
        Vertex {
            loc,
            dual: (i, j, k),
            radius2: gen_loc.distance_squared(d_loc),
        }
    }
}

#[derive(Clone)]
pub(super) struct ConvexCell {
    pub loc: DVec3,
    pub clipping_planes: Vec<HalfSpace>,
    pub vertices: Vec<Vertex>,
    boundary: SimpleCycle,
    safety_radius: f64,
    pub idx: usize,
}

impl ConvexCell {
    /// Initialize each voronoi cell as the bounding box of the simulation volume.
    pub fn init(
        loc: DVec3,
        idx: usize,
        simulation_volume: &SimulationBoundary,
        dimensionality: Dimensionality,
    ) -> Self {
        let clipping_planes = simulation_volume.clipping_planes.clone();

        let vertices = vec![
            Vertex::from_dual(2, 5, 0, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(5, 3, 0, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(1, 5, 2, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(5, 1, 3, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(4, 2, 0, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(4, 0, 3, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(2, 4, 1, &clipping_planes, loc, dimensionality),
            Vertex::from_dual(4, 3, 1, &clipping_planes, loc, dimensionality),
        ];

        let mut cell = ConvexCell {
            loc,
            idx,
            boundary: SimpleCycle::new(clipping_planes.len()),
            clipping_planes,
            vertices,
            safety_radius: 0.,
        };
        cell.update_safety_radius();
        cell
    }

    /// Build the Convex cell by repeatedly intersecting it with the appropriate half spaces
    pub(super) fn build(
        &mut self,
        generators: &[Generator],
        mut nearest_neighbours: Box<dyn Iterator<Item = (usize, Option<DVec3>)> + '_>,
        dimensionality: Dimensionality,
    ) {
        // skip the first nearest neighbour (will be this cell)
        assert_eq!(
            nearest_neighbours
                .next()
                .expect("Nearest neighbours cannot be empty!")
                .0,
            self.idx,
            "First nearest neighbour should be the generator itself!"
        );
        // now loop over the nearest neighbours and clip this cell until the safety radius is reached
        for (idx, shift) in nearest_neighbours {
            let generator = generators[idx];
            let ngb_loc;
            if let Some(shift) = shift {
                ngb_loc = generator.loc() + shift;
            } else {
                ngb_loc = generator.loc();
            }
            let dx = self.loc - ngb_loc;
            let dist = dx.length();
            assert!(dist.is_finite() && dist > 0.0, "Degenerate point set!");
            if self.safety_radius < dist {
                return;
            }
            let n = dx / dist;
            let p = 0.5 * (self.loc + ngb_loc);
            self.clip_by_plane(HalfSpace::new(n, p, Some(idx), shift), dimensionality);
        }
    }

    pub fn clip_by_plane(&mut self, p: HalfSpace, dimensionality: Dimensionality) {
        // loop over vertices and remove the ones clipped by p
        let mut i = 0;
        let mut num_v = self.vertices.len();
        let mut num_r = 0;
        while i < num_v {
            if p.clips(self.vertices[i].loc) {
                num_v -= 1;
                num_r += 1;
                self.vertices.swap(i, num_v);
            } else {
                i += 1;
            }
        }

        // Were any vertices clipped?
        if num_r > 0 {
            // Add the new clipping plane
            let p_idx = self.clipping_planes.len();
            self.clipping_planes.push(p);
            self.boundary.grow();
            // Compute the boundary of the (dual) topological triangulated disk around the vertices to be removed.
            Self::compute_boundary(&mut self.boundary, &mut self.vertices[num_v..]);
            let mut boundary = self.boundary.iter().take(self.boundary.len + 1);
            // finally we can *realy* remove the vertices.
            self.vertices.truncate(num_v);
            // Add new vertices constructed from the new clipping plane and the boundary
            let mut cur = boundary
                .next()
                .expect("Boundary contains at least 3 elements");
            for next in boundary {
                self.vertices.push(Vertex::from_dual(
                    cur,
                    next,
                    p_idx,
                    &self.clipping_planes,
                    self.loc,
                    dimensionality,
                ));
                cur = next;
            }
            self.update_safety_radius();
        }
    }

    fn compute_boundary(boundary: &mut SimpleCycle, vertices: &mut [Vertex]) {
        boundary.init(vertices[0].dual.0, vertices[0].dual.1, vertices[0].dual.2);

        for i in 1..vertices.len() {
            // Look for a suitable next vertex to extend the boundary
            let mut idx = i;
            loop {
                assert!(
                    idx < vertices.len(),
                    "No suitable vertex found to extend boundary!"
                );
                let vertex = &vertices[idx].dual;
                match boundary.try_extend(vertex.0, vertex.1, vertex.2) {
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

    fn update_safety_radius(&mut self) {
        let max_dist_2 = self
            .vertices
            .iter()
            .map(|v| v.radius2)
            .max_by(|a, b| a.partial_cmp(b).expect("NaN distance encountered!"))
            .expect("Vertices cannot be empty!");

        // Update safety radius
        self.safety_radius = 2. * max_dist_2.sqrt();
    }
}

impl From<ConvexCellAlternative> for ConvexCell {
    fn from(convex_cell_alt: ConvexCellAlternative) -> Self {
        let clipping_planes = convex_cell_alt
            .neighbours
            .iter()
            .map(|ngb| HalfSpace::new(-ngb.plane.n, ngb.plane.p, ngb.idx, ngb.shift))
            .collect::<Vec<_>>();
        let vertices = convex_cell_alt
            .vertices
            .iter()
            .map(|v| {
                Vertex::from_dual(
                    v.repr.0,
                    v.repr.1,
                    v.repr.2,
                    &clipping_planes,
                    convex_cell_alt.loc,
                    Dimensionality::Dimensionality3D,
                )
            })
            .collect::<Vec<_>>();

        ConvexCell {
            loc: convex_cell_alt.loc,
            clipping_planes,
            vertices,
            boundary: SimpleCycle::new(0),
            safety_radius: 0.,
            idx: convex_cell_alt.idx,
        }
    }
}
