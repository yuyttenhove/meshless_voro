use glam::{DMat3, DVec3};

use crate::space::Space;

struct HalfSpace {
    n: DVec3,
    p: DVec3,
    d: f64,
}

impl HalfSpace {
    fn new(n: DVec3, p: DVec3) -> Self {
        HalfSpace { n, p, d: n.dot(p) }
    }

    /// Whether a vertex is clipped by this half space
    fn clips(&self, vertex: DVec3) -> bool {
        self.n.dot(vertex) < self.d
    }
}

/// Calculate the intersection of 3 planes.
/// see: https://mathworld.wolfram.com/Plane-PlaneIntersection.html
fn intersect_planes(p0: &HalfSpace, p1: &HalfSpace, p2: &HalfSpace) -> DVec3 {
    let det = DMat3::from_cols(p0.n, p1.n, p2.n).determinant();

    (p0.p.dot(p0.n) * p1.n.cross(p2.n)
        + p1.p.dot(p1.n) * p2.n.cross(p0.n)
        + p2.p.dot(p2.n) * p0.n.cross(p1.n))
        / det
}

struct Vertex {
    loc: DVec3,
    dual: (usize, usize, usize),
}

impl Vertex {
    fn from_dual(i: usize, j: usize, k: usize, planes: &[HalfSpace]) -> Self {
        Vertex {
            loc: intersect_planes(&planes[i], &planes[j], &planes[k]),
            dual: (i, j, k),
        }
    }
}

struct VoronoiCell {
    loc: DVec3,
    clipping_planes: Vec<HalfSpace>,
    vertices: Vec<Vertex>,
    safety_radius: f64,
}

impl VoronoiCell {
    /// Initialize each boronoi cell as the bounding box of the simulation volume.
    fn init(loc: DVec3, anchor: DVec3, width: DVec3) -> Self {
        let clipping_planes = vec![
            HalfSpace::new(DVec3::X, anchor),
            HalfSpace::new(DVec3::NEG_X, anchor + width),
            HalfSpace::new(DVec3::Y, anchor),
            HalfSpace::new(DVec3::NEG_Y, anchor + width),
            HalfSpace::new(DVec3::Z, anchor),
            HalfSpace::new(DVec3::NEG_Z, anchor + width),
        ];
        let vertices = vec![
            Vertex::from_dual(2, 5, 0, &clipping_planes),
            Vertex::from_dual(5, 3, 0, &clipping_planes),
            Vertex::from_dual(1, 5, 2, &clipping_planes),
            Vertex::from_dual(5, 1, 3, &clipping_planes),
            Vertex::from_dual(4, 2, 0, &clipping_planes),
            Vertex::from_dual(4, 0, 3, &clipping_planes),
            Vertex::from_dual(2, 4, 1, &clipping_planes),
            Vertex::from_dual(4, 3, 1, &clipping_planes),
        ];
        let mut cell = VoronoiCell {
            loc,
            clipping_planes,
            vertices,
            safety_radius: 0.,
        };
        cell.update_safety_radius();
        cell
    }

    /// Build the voronoi cell by repeatedly intersecting it with the appropriate half spaces
    fn build(&mut self, generators: &[DVec3], nearest_neighbours: &[usize]) {
        for &idx in nearest_neighbours.iter() {
            let generator = generators[idx];
            let dx = self.loc - generator;
            let dist = dx.length();
            assert!(dist.is_finite() && dist > 0.0, "Degenerate point set!");
            if self.safety_radius < dist {
                return;
            }
            let n = dx / dist;
            let p = 0.5 * (self.loc + generator);
            self.clip_by_plane(HalfSpace::new(n, p));
        }
    }

    fn clip_by_plane(&mut self, p: HalfSpace) {
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
            // Compute the boundary of the (dual) topological triangulated disk around the vertices to be removed.
            let boundary = Self::compute_boundary(&mut self.vertices[num_v..]);
            // finally we can *realy* remove the vertices.
            self.vertices.truncate(num_v);
            // Add new vertices constructed from the new clipping plane and the boundary
            for edge in boundary.windows(2) {
                self.vertices.push(Vertex::from_dual(
                    edge[0],
                    edge[1],
                    p_idx,
                    &self.clipping_planes,
                ));
            }
            self.update_safety_radius();
        }
    }

    fn compute_boundary(vertices: &mut [Vertex]) -> Vec<usize> {
        let mut boundary = vec![
            vertices[0].dual.0,
            vertices[0].dual.1,
            vertices[0].dual.2,
            vertices[0].dual.0,
        ];
        for i in 1..vertices.len() {
            // Look for a suitable next vertex to extend the boundary
            let mut idx = i;
            let (insertion_index, insertion_value) = loop {
                assert!(
                    idx < vertices.len(),
                    "No suitable vertex found to extend boundary!"
                );
                let vertex = &vertices[idx].dual;
                let in_boundary = (
                    boundary.iter().position(|v| v == &vertex.0),
                    boundary.iter().position(|v| v == &vertex.1),
                    boundary.iter().position(|v| v == &vertex.2),
                );
                match in_boundary {
                    (Some(idx_0), Some(idx_1), None) => break (idx_0.min(idx_1), vertex.2),
                    (Some(idx_0), None, Some(idx_2)) => break (idx_0.min(idx_2), vertex.1),
                    (None, Some(idx_1), Some(idx_2)) => break (idx_1.min(idx_2), vertex.0),
                    _ => {
                        idx += 1;
                    }
                }
            };
            // swap the suitalble vertex to the front of the array
            vertices.swap(i, idx);
            // insert the next boundary piece
            boundary.insert(insertion_index, insertion_value);
        }

        boundary
    }

    fn update_safety_radius(&mut self) {
        let mut max_dist_2 = 0f64;
        for vertex in self.vertices.iter() {
            max_dist_2 = max_dist_2.max(self.loc.distance_squared(vertex.loc));
        }
        self.safety_radius = max_dist_2.sqrt();
    }
}

pub struct VoronoiFace {
    left: usize,
    right: usize,
    area: f64,
    midpoint: DVec3,
}

pub struct Voronoi {
    anchor: DVec3,
    width: DVec3,
    cells: Vec<VoronoiCell>,
}

impl Voronoi {
    pub fn build(generators: &[DVec3], anchor: DVec3, width: DVec3) -> Self {
        let max_cell_width =
            (5. * (width.x * width.y * width.z) / generators.len() as f64).powf(1. / 2.);
        let mut space = Space::new(anchor, width, max_cell_width);
        space.add_parts(generators);
        let knn = space.knn(40);

        let mut voronoi_cells: Vec<VoronoiCell> = generators
            .iter()
            .map(|g| VoronoiCell::init(*g, anchor, width))
            .collect();
        for (i, voronoi_cell) in voronoi_cells.iter_mut().enumerate() {
            voronoi_cell.build(generators, &knn[i])
        }

        Voronoi {
            anchor,
            width,
            cells: voronoi_cells,
        }
    }

    pub fn volumes(&self) -> Vec<f64> {
        todo!()
    }

    pub fn faces(&self) -> Vec<VoronoiFace> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{distributions::Uniform, prelude::*};

    fn gererators_2d() -> Vec<DVec3> {
        let anchor = DVec3::splat(1.);
        let mut p_x = vec![];
        let mut rng = thread_rng();
        let distr = Uniform::new(0., 0.5);
        for i in 0..4 {
            for j in 0..4 {
                let k = 0;
                let cell_anchor = anchor
                    + DVec3 {
                        x: i as f64 * 0.5,
                        y: j as f64 * 0.5,
                        z: k as f64 * 0.5,
                    };
                for _ in 0..12 {
                    // Add 3 parts per cell
                    let rel_pos = DVec3 {
                        x: rng.sample(distr),
                        y: rng.sample(distr),
                        z: rng.sample(distr),
                    };
                    p_x.push(cell_anchor + rel_pos);
                }
            }
        }
        p_x
    }

    #[test]
    fn test_init_voronoi_cell() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(3.);
        let loc = DVec3::splat(3.);
        let cell = VoronoiCell::init(loc, anchor, width);

        assert_eq!(cell.vertices.len(), 8);
        assert_eq!(cell.clipping_planes.len(), 6);
        assert_eq!(cell.safety_radius, 12f64.sqrt())
    }

    #[test]
    fn test_clipping() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let loc = DVec3::splat(2.);
        let mut cell = VoronoiCell::init(loc, anchor, width);

        let ngb = DVec3::splat(3.5);
        let dx = cell.loc - ngb;
        let dist = dx.length();
        let n = dx / dist;
        let p = 0.5 * (cell.loc + ngb);
        cell.clip_by_plane(HalfSpace::new(n, p));

        assert_eq!(cell.clipping_planes.len(), 7)
    }

    #[test]
    fn test_voronoi() {
        let generators = gererators_2d();
        let voronoi = Voronoi::build(&generators, DVec3::splat(1.), DVec3::splat(2.));
        assert_eq!(voronoi.cells.len(), generators.len());
    }
}
