use glam::{DMat3, DVec3};

use crate::{
    simple_cycle::SimpleCycle,
    util::{signed_volume_tet, GetMutMultiple},
    Voronoi, VoronoiFace,
};

use super::{Dimensionality, Generator};

#[derive(Clone)]
pub struct HalfSpace {
    pub n: DVec3,
    p: DVec3,
    d: f64,
    pub right_idx: Option<usize>,
    pub shift: Option<DVec3>,
}

impl HalfSpace {
    fn new(n: DVec3, p: DVec3, right_idx: Option<usize>, shift: Option<DVec3>) -> Self {
        HalfSpace {
            n,
            p,
            d: n.dot(p),
            right_idx,
            shift,
        }
    }

    /// Whether a vertex is clipped by this half space
    fn clips(&self, vertex: DVec3) -> bool {
        self.n.dot(vertex) < self.d
    }

    pub fn project_onto(&self, vertex: DVec3) -> DVec3 {
        vertex + (self.p - vertex).project_onto(self.n)
    }

    /// Project the a vertex on the intersection of two planes.
    pub fn project_onto_intersection(&self, other: &Self, vertex: DVec3) -> DVec3 {
        // first create a plane through the point perpendicular to both planes
        let p_perp = HalfSpace::new(self.n.cross(other.n), vertex, None, None);

        // The projection is the intersection of the planes self, other and p_perp
        intersect_planes(self, other, &p_perp)
    }
}

/// Calculate the intersection of 3 planes.
/// see: https://mathworld.wolfram.com/Plane-PlaneIntersection.html
fn intersect_planes(p0: &HalfSpace, p1: &HalfSpace, p2: &HalfSpace) -> DVec3 {
    let det = DMat3::from_cols(p0.n, p1.n, p2.n).determinant();
    assert!(det != 0., "Degenerate 3-plane intersection!");

    (p0.p.dot(p0.n) * p1.n.cross(p2.n)
        + p1.p.dot(p1.n) * p2.n.cross(p0.n)
        + p2.p.dot(p2.n) * p0.n.cross(p1.n))
        / det
}

#[derive(Clone)]
pub struct Vertex {
    pub loc: DVec3,
    pub dual: (usize, usize, usize),
}

impl Vertex {
    fn from_dual(i: usize, j: usize, k: usize, planes: &[HalfSpace]) -> Self {
        Vertex {
            loc: intersect_planes(&planes[i], &planes[j], &planes[k]),
            dual: (i, j, k),
        }
    }
}

#[derive(Clone)]
pub struct ConvexCell {
    pub loc: DVec3,
    pub clipping_planes: Vec<HalfSpace>,
    pub vertices: Vec<Vertex>,
    safety_radius: f64,
    pub idx: usize,
}

impl ConvexCell {
    /// Initialize each voronoi cell as the bounding box of the simulation volume.
    pub(super) fn init(
        loc: DVec3,
        idx: usize,
        simulation_volume: &ConvexCell,
        dimensionality: Dimensionality,
    ) -> Self {
        let mut cell = simulation_volume.clone();
        cell.idx = idx;
        cell.loc = loc;
        cell.update_safety_radius(dimensionality);
        cell
    }

    pub(super) fn init_simulation_volume(
        mut anchor: DVec3,
        mut width: DVec3,
        periodic: bool,
        dimensionality: Dimensionality,
    ) -> Self {
        if periodic {
            anchor.x -= width.x;
            width.x *= 3.;
            if let Dimensionality::Dimensionality2D | Dimensionality::Dimensionality3D =
                dimensionality
            {
                anchor.y -= width.y;
                width.y *= 3.;
            };
            if let Dimensionality::Dimensionality3D = dimensionality {
                anchor.z -= width.z;
                width.z *= 3.;
            }
        }
        let clipping_planes = vec![
            HalfSpace::new(DVec3::X, anchor, None, None),
            HalfSpace::new(DVec3::NEG_X, anchor + width, None, None),
            HalfSpace::new(DVec3::Y, anchor, None, None),
            HalfSpace::new(DVec3::NEG_Y, anchor + width, None, None),
            HalfSpace::new(DVec3::Z, anchor, None, None),
            HalfSpace::new(DVec3::NEG_Z, anchor + width, None, None),
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
        ConvexCell {
            loc: DVec3::ZERO,
            clipping_planes,
            vertices,
            safety_radius: 0.,
            idx: 0,
        }
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

    fn clip_by_plane(&mut self, p: HalfSpace, dimensionality: Dimensionality) {
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
            self.update_safety_radius(dimensionality);
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

    fn update_safety_radius(&mut self, dimensionality: Dimensionality) {
        let mut max_dist_2 = 0f64;
        for vertex in self.vertices.iter() {
            let mut v_loc = vertex.loc;
            // Ignore the unused dimensions fo the safety radius!
            match dimensionality {
                Dimensionality::Dimensionality1D => {
                    v_loc.y = self.loc.y;
                    v_loc.z = self.loc.z;
                }
                Dimensionality::Dimensionality2D => v_loc.z = self.loc.z,
                _ => (),
            }
            max_dist_2 = max_dist_2.max(self.loc.distance_squared(v_loc));
        }
        self.safety_radius = 2. * max_dist_2.sqrt();
    }
}

/// A Voronoi cell.
#[derive(Default, Debug, Clone, Copy)]
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

    /// Build a Voronoi cell from a ConvexCell by computing the relevant integrals.
    ///
    /// Any Voronoi faces that are created by the construction of this cell are stored in the `faces` vector.
    pub fn from_convex_cell(
        convex_cell: &ConvexCell,
        faces: &mut Vec<VoronoiFace>,
        mask: Option<&[bool]>,
    ) -> Self {
        let idx = convex_cell.idx;
        let loc = convex_cell.loc;
        let mut centroid = DVec3::ZERO;
        let mut volume = 0.;

        let mut maybe_faces: Vec<Option<VoronoiFace>> = (0..convex_cell.clipping_planes.len())
            .map(|_| None)
            .collect();

        fn maybe_init_face(
            maybe_face: &mut Option<VoronoiFace>,
            plane: &HalfSpace,
            left_idx: usize,
            mask: Option<&[bool]>,
        ) {
            match plane {
                // Don't construct faces twice.
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    ..
                } if *right_idx <= left_idx && mask.map_or(true, |mask| mask[*right_idx]) => (),
                _ => {
                    maybe_face.get_or_insert(VoronoiFace::init(
                        left_idx,
                        plane.right_idx,
                        -plane.n,
                        plane.shift,
                    ));
                }
            }
        }

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
            maybe_init_face(face_0, plane_0, idx, mask);
            maybe_init_face(face_1, plane_1, idx, mask);
            maybe_init_face(face_2, plane_2, idx, mask);

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

            // Calculate signed volumes of tetrahedra
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
            face_0.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p01, g_on_p0, loc);
                f.extend(g_on_p012, g_on_p0, g_on_p02, loc);
            });
            face_1.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p1, g_on_p01, loc);
                f.extend(g_on_p012, g_on_p12, g_on_p1, loc);
            });
            face_2.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p02, g_on_p2, loc);
                f.extend(g_on_p012, g_on_p2, g_on_p12, loc);
            });
        }

        // Filter out uninitialized faces and finalize the rest
        for maybe_face in maybe_faces {
            if let Some(mut face) = maybe_face {
                face.finalize();
                faces.push(face);
            }
        }

        VoronoiCell::init(loc, centroid / volume, volume)
    }

    pub(super) fn finalize(&mut self, face_connections_offset: usize, face_count: usize) {
        self.face_connections_offset = face_connections_offset;
        self.face_count = face_count;
    }

    /// Get the position of the generator of this Voronoi cell.
    pub fn loc(&self) -> DVec3 {
        self.loc
    }

    /// Get the position of the centroid of this cell
    pub fn centroid(&self) -> DVec3 {
        self.centroid
    }

    /// Get the volume of this cell
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Get an `Iterator` over the Voronoi faces that have this cell as their left _or_ right generator.
    pub fn faces<'a>(
        &'a self,
        voronoi: &'a Voronoi,
    ) -> Box<dyn Iterator<Item = &VoronoiFace> + 'a> {
        let indices = &voronoi.cell_face_connections
            [self.face_connections_offset..(self.face_connections_offset + self.face_count)];
        Box::new(indices.iter().map(|&i| &voronoi.faces[i]))
    }

    /// Get the offset of the slice of the indices of this cell's faces in the `Voronoi::cell_face_connections` array.
    pub fn face_connections_offset(&self) -> usize {
        self.face_connections_offset
    }

    /// Get the length of the slice of the indices of this cell's faces in the `Voronoi::cell_face_connections` array.
    pub fn face_count(&self) -> usize {
        self.face_count
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const DIM3D: usize = 3;

    #[test]
    fn test_init_simulation_volume() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(4.);
        let cell = ConvexCell::init_simulation_volume(anchor, width, false, DIM3D.into());

        assert_eq!(cell.vertices.len(), 8);
        assert_eq!(cell.clipping_planes.len(), 6);
    }

    #[test]
    fn test_clipping() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let loc = DVec3::splat(2.);
        let volume = ConvexCell::init_simulation_volume(anchor, width, false, DIM3D.into());
        let mut cell = ConvexCell::init(loc, 0, &volume, DIM3D.into());

        let ngb = DVec3::splat(2.5);
        let dx = cell.loc - ngb;
        let dist = dx.length();
        let n = dx / dist;
        let p = 0.5 * (cell.loc + ngb);
        cell.clip_by_plane(
            HalfSpace::new(n, p, Some(1), Some(DVec3::ZERO)),
            DIM3D.into(),
        );

        assert_eq!(cell.clipping_planes.len(), 7)
    }
}
