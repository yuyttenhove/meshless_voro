use glam::{DMat3, DVec3};

use crate::{
    simple_cycle::SimpleCycle,
    space::Space,
    util::{signed_area_tri, signed_volume_tet, GetMutMultiple},
};

#[cfg(test)]
mod test;

struct HalfSpace {
    n: DVec3,
    p: DVec3,
    d: f64,
    right_idx: Option<usize>,
}

impl HalfSpace {
    fn new(n: DVec3, p: DVec3, right_idx: Option<usize>) -> Self {
        HalfSpace {
            n,
            p,
            d: n.dot(p),
            right_idx,
        }
    }

    /// Whether a vertex is clipped by this half space
    fn clips(&self, vertex: DVec3) -> bool {
        self.n.dot(vertex) < self.d
    }

    fn project_onto(&self, vertex: DVec3) -> DVec3 {
        vertex + (self.p - vertex).project_onto(self.n)
    }

    /// Project the a vertex on the intersection of two planes, by first projecting it
    /// on one plane and then project the result on the second plane.
    fn project_onto_intersection(&self, other: &Self, vertex: DVec3) -> DVec3 {
        self.project_onto(other.project_onto(vertex))
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

struct ConvexCell {
    loc: DVec3,
    clipping_planes: Vec<HalfSpace>,
    vertices: Vec<Vertex>,
    safety_radius: f64,
    idx: usize,
}

impl ConvexCell {
    /// Initialize each voronoi cell as the bounding box of the simulation volume.
    fn init(loc: DVec3, anchor: DVec3, width: DVec3, idx: usize) -> Self {
        let clipping_planes = vec![
            HalfSpace::new(DVec3::X, anchor, None),
            HalfSpace::new(DVec3::NEG_X, anchor + width, None),
            HalfSpace::new(DVec3::Y, anchor, None),
            HalfSpace::new(DVec3::NEG_Y, anchor + width, None),
            HalfSpace::new(DVec3::Z, anchor, None),
            HalfSpace::new(DVec3::NEG_Z, anchor + width, None),
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
        let mut cell = ConvexCell {
            loc,
            clipping_planes,
            vertices,
            safety_radius: 0.,
            idx,
        };
        cell.update_safety_radius();
        cell
    }

    /// Build the voronoi cell by repeatedly intersecting it with the appropriate half spaces
    fn build(&mut self, generators: &[DVec3], nearest_neighbours: &[usize]) -> Result<(), ()> {
        for &idx in nearest_neighbours.iter() {
            let generator = generators[idx];
            let dx = self.loc - generator;
            let dist = dx.length();
            assert!(dist.is_finite() && dist > 0.0, "Degenerate point set!");
            if self.safety_radius < dist {
                return Ok(());
            }
            let n = dx / dist;
            let p = 0.5 * (self.loc + generator);
            self.clip_by_plane(HalfSpace::new(n, p, Some(idx)));
        }

        Err(())
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
            for i in 0..boundary.len() {
                self.vertices.push(Vertex::from_dual(
                    boundary[i],
                    boundary[i + 1],
                    p_idx,
                    &self.clipping_planes,
                ));
            }
            self.update_safety_radius();
        }
    }

    fn compute_boundary(vertices: &mut [Vertex]) -> SimpleCycle<usize> {
        let mut boundary =
            SimpleCycle::new(vertices[0].dual.0, vertices[0].dual.1, vertices[0].dual.2);

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

        debug_assert!(boundary.is_valid());
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

impl VoronoiFace {
    fn init(left: usize, right: usize) -> Self {
        VoronoiFace {
            left,
            right,
            area: 0.,
            midpoint: DVec3::ZERO,
        }
    }

    fn finalize(&mut self) {
        self.midpoint /= self.area;
    }

    pub fn left(&self) -> usize {
        self.left
    }

    pub fn right(&self) -> usize {
        self.right
    }

    pub fn area(&self) -> f64 {
        self.area
    }

    pub fn midpoint(&self) -> DVec3 {
        self.midpoint
    }
}

pub struct VoronoiCell {
    loc: DVec3,
    centroid: DVec3,
    volume: f64,
    face_connections_offset: usize,
    face_count: usize,
}

impl VoronoiCell {
    fn init(loc: DVec3, centroid: DVec3, volume: f64) -> Self {
        Self {
            loc,
            centroid,
            volume,
            face_connections_offset: 0,
            face_count: 0,
        }
    }

    fn from_convex_cell(convex_cell: &ConvexCell) -> (VoronoiCell, Vec<VoronoiFace>) {
        let idx = convex_cell.idx;
        let loc = convex_cell.loc;
        let mut centroid = DVec3::ZERO;
        let mut volume = 0.;

        let mut maybe_faces: Vec<Option<VoronoiFace>> = (0..convex_cell.clipping_planes.len())
            .map(|_| None)
            .collect();

        // Loop over vertices and compute the necessary integrals/barycenter calculations
        for vertex in &convex_cell.vertices {
            // Initialize these faces
            let face_idx_0 = vertex.dual.0;
            let face_idx_1 = vertex.dual.1;
            let face_idx_2 = vertex.dual.2;
            let plane_0 = &convex_cell.clipping_planes[face_idx_0];
            let plane_1 = &convex_cell.clipping_planes[face_idx_1];
            let plane_2 = &convex_cell.clipping_planes[face_idx_2];
            let (face_0, face_1, face_2) =
                maybe_faces.get_3_mut(face_idx_0, face_idx_1, face_idx_2);
            // If the faces are None, and we want to create a face (idx < right_idx) initialize it now.
            match plane_0.right_idx {
                Some(right_idx) if right_idx > idx => {
                    face_0.get_or_insert(VoronoiFace::init(idx, right_idx));
                }
                _ => (), // Don't construct boundary faces,
            }
            match plane_1.right_idx {
                Some(right_idx) if right_idx > idx => {
                    face_1.get_or_insert(VoronoiFace::init(idx, right_idx));
                }
                _ => (), // Don't construct boundary faces,
            }
            match plane_2.right_idx {
                Some(right_idx) if right_idx > idx => {
                    face_2.get_or_insert(VoronoiFace::init(idx, right_idx));
                }
                _ => (), // Don't construct boundary faces,
            }

            // Project generator on planes
            let g_on_p0 = plane_0.project_onto(loc);
            let g_on_p1 = plane_1.project_onto(loc);
            let g_on_p2 = plane_2.project_onto(loc);

            // Project generator on edges between planes
            let g_on_p01 = plane_0.project_onto_intersection(&plane_1, loc);
            let g_on_p02 = plane_0.project_onto_intersection(&plane_2, loc);
            let g_on_p12 = plane_1.project_onto_intersection(&plane_2, loc);

            // Project generator on vertex between planes
            let g_on_p012 = vertex.loc;

            // Calculate signed volumes of tetrahedra TODO
            let v_001 = signed_volume_tet(g_on_p012, g_on_p01, g_on_p0, loc);
            let v_002 = signed_volume_tet(g_on_p012, g_on_p0, g_on_p02, loc);
            let v_101 = signed_volume_tet(g_on_p012, g_on_p1, g_on_p01, loc);
            let v_112 = signed_volume_tet(g_on_p012, g_on_p12, g_on_p1, loc);
            let v_202 = signed_volume_tet(g_on_p012, g_on_p02, g_on_p2, loc);
            let v_212 = signed_volume_tet(g_on_p012, g_on_p2, g_on_p12, loc);
            volume += v_001 + v_002 + v_101 + v_112 + v_202 + v_212;

            // Calculate barycenters of the tetrahedra
            centroid += 0.25
                * (v_001 * (g_on_p0 + g_on_p01 + g_on_p012 + loc)
                    + v_002 * (g_on_p0 + g_on_p02 + g_on_p012 + loc)
                    + v_101 * (g_on_p1 + g_on_p01 + g_on_p012 + loc)
                    + v_112 * (g_on_p1 + g_on_p12 + g_on_p012 + loc)
                    + v_202 * (g_on_p2 + g_on_p02 + g_on_p012 + loc)
                    + v_212 * (g_on_p2 + g_on_p12 + g_on_p012 + loc));

            // Calculate the signed areas of the triangles on the faces and update their barycenters
            let frac_1_3 = 1. / 3.;
            face_0.as_mut().map(|f| {
                let v_001 = signed_area_tri(g_on_p012, g_on_p01, g_on_p0, loc);
                let v_002 = signed_area_tri(g_on_p012, g_on_p0, g_on_p02, loc);
                f.area += v_001 + v_002;
                f.midpoint += frac_1_3
                    * (v_001 * (g_on_p0 + g_on_p01 + g_on_p012)
                        + v_002 * (g_on_p0 + g_on_p02 + g_on_p012));
            });
            face_1.as_mut().map(|f| {
                let v_101 = signed_area_tri(g_on_p012, g_on_p1, g_on_p01, loc);
                let v_112 = signed_area_tri(g_on_p012, g_on_p12, g_on_p1, loc);
                f.area += v_101 + v_112;
                f.midpoint += frac_1_3
                    * (v_101 * (g_on_p1 + g_on_p01 + g_on_p012)
                        + v_112 * (g_on_p1 + g_on_p12 + g_on_p012));
            });
            face_2.as_mut().map(|f| {
                let v_202 = signed_area_tri(g_on_p012, g_on_p02, g_on_p2, loc);
                let v_212 = signed_area_tri(g_on_p012, g_on_p2, g_on_p12, loc);
                f.area += v_202 + v_212;
                f.midpoint += frac_1_3
                    * (v_202 * (g_on_p2 + g_on_p02 + g_on_p012)
                        + v_212 * (g_on_p2 + g_on_p12 + g_on_p012));
            });
        }

        // Filter out uninitialized faces and finalize the rest
        let faces = maybe_faces
            .into_iter()
            .filter_map(|f| {
                f.map(|mut v| {
                    v.finalize();
                    v
                })
            })
            .collect();
        (VoronoiCell::init(loc, centroid / volume, volume), faces)
    }

    pub fn loc(&self) -> DVec3 {
        self.loc
    }

    pub fn centroid(&self) -> DVec3 {
        self.centroid
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    pub fn faces<'a>(
        &'a self,
        voronoi: &'a Voronoi,
    ) -> Box<dyn Iterator<Item = &VoronoiFace> + 'a> {
        let indices = &voronoi.cell_face_connections
            [self.face_connections_offset..(self.face_connections_offset + self.face_count)];
        Box::new(indices.iter().map(|&i| &voronoi.faces[i]))
    }
}

pub struct Voronoi {
    anchor: DVec3,
    width: DVec3,
    cells: Vec<VoronoiCell>,
    faces: Vec<VoronoiFace>,
    cell_face_connections: Vec<usize>,
}

impl Voronoi {
    /// Construct a Voronoi tesselation using the VoroGPU algorithm with `k` nearest neighbours.
    pub fn build(generators: &[DVec3], anchor: DVec3, width: DVec3, k: usize) -> Self {
        let max_cell_width =
            (5. * (width.x * width.y * width.z) / generators.len() as f64).powf(1. / 2.);
        let mut space = Space::new(anchor, width, max_cell_width);
        space.add_parts(generators);
        let knn = space.knn(k);

        let mut cells = Vec::with_capacity(generators.len());
        let mut faces = Vec::with_capacity(generators.len());
        for (i, generator) in generators.iter().enumerate() {
            let mut convex_cell = ConvexCell::init(*generator, anchor, width, i);
            let result = convex_cell.build(generators, &knn[i]);
            if k + 1 < generators.len() {
                result.expect("Security radius not reached! Run with larger number of neighbours!");
            }
            let (this_cell, this_faces) = VoronoiCell::from_convex_cell(&convex_cell);
            cells.push(this_cell);
            faces.push(this_faces);
        }

        Voronoi {
            anchor,
            width,
            cells,
            faces: faces.into_iter().flatten().collect(),
            cell_face_connections: vec![],
        }
        .finalize()
    }

    /// Compute the cell_face_connections.
    fn finalize(mut self) -> Self {
        let mut cell_face_connections: Vec<Vec<usize>> =
            (0..self.cells.len()).map(|_| vec![]).collect();

        for (i, face) in self.faces.iter().enumerate() {
            cell_face_connections[face.left].push(i);
            cell_face_connections[face.right].push(i);
        }

        let mut count_total = 0;
        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.face_connections_offset = count_total;
            cell.face_count = cell_face_connections[i].len();
            count_total += cell.face_count;
        }

        self.cell_face_connections = cell_face_connections.into_iter().flatten().collect();

        self
    }

    pub fn anchor(&self) -> DVec3 {
        self.anchor
    }

    pub fn width(&self) -> DVec3 {
        self.width
    }

    pub fn cells(&self) -> &[VoronoiCell] {
        self.cells.as_ref()
    }

    pub fn faces(&self) -> &[VoronoiFace] {
        self.faces.as_ref()
    }
}
