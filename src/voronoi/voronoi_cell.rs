use glam::DVec3;

use crate::{
    geometry::{intersect_planes, Plane},
    integrators::{
        ScalarVoronoiFaceIntegrator, VectorVoronoiFaceIntegrator, VolumeCentroidIntegrator,
        VoronoiCellIntegrator,
    },
    simple_cycle::SimpleCycle,
    util::GetMutMultiple,
    voronoi::{
        voronoi_face::{VoronoiFace, VoronoiFaceBuilder},
        ConvexCellAlternative, Voronoi,
    },
};

use super::{Dimensionality, Generator};

#[derive(Clone)]
pub struct HalfSpace {
    plane: Plane,
    d: f64,
    pub right_idx: Option<usize>,
    pub shift: Option<DVec3>,
}

impl HalfSpace {
    fn new(n: DVec3, p: DVec3, right_idx: Option<usize>, shift: Option<DVec3>) -> Self {
        HalfSpace {
            plane: Plane::new(n, p),
            d: n.dot(p),
            right_idx,
            shift,
        }
    }

    /// Whether a vertex is clipped by this half space
    fn clips(&self, vertex: DVec3) -> bool {
        self.plane.n.dot(vertex) < self.d
    }

    pub fn normal(&self) -> DVec3 {
        self.plane.n
    }

    pub fn project_onto(&self, point: DVec3) -> DVec3 {
        self.plane.project_onto(point)
    }
}

#[derive(Clone)]
pub struct Vertex {
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
pub(crate) struct SimulationBoundary {
    clipping_planes: Vec<HalfSpace>,
}

impl SimulationBoundary {
    pub fn cuboid(
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

        Self { clipping_planes }
    }
}

#[derive(Clone)]
pub struct ConvexCell {
    pub loc: DVec3,
    pub clipping_planes: Vec<HalfSpace>,
    pub vertices: Vec<Vertex>,
    boundary: SimpleCycle,
    safety_radius: f64,
    pub idx: usize,
}

impl ConvexCell {
    /// Initialize each voronoi cell as the bounding box of the simulation volume.
    pub(super) fn init(
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

struct VoronoiCellBuilder {
    loc: DVec3,
    volume_centroid: VolumeCentroidIntegrator,
}

impl VoronoiCellBuilder {
    fn new(loc: DVec3) -> Self {
        Self {
            loc,
            volume_centroid: VolumeCentroidIntegrator::init(),
        }
    }

    fn extend(&mut self, v0: DVec3, v1: DVec3, v2: DVec3) {
        self.volume_centroid.collect(v0, v1, v2, self.loc);
    }

    fn build(self) -> VoronoiCell {
        let (volume, centroid) = self.volume_centroid.finalize();
        VoronoiCell::init(self.loc, centroid, volume)
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
    pub fn from_convex_cell<'a>(
        convex_cell: &'a ConvexCell,
        faces: &mut Vec<VoronoiFaceBuilder<'a>>,
        mask: Option<&[bool]>,
        vector_face_integrators: &[Box<
            dyn Fn() -> Box<dyn VectorVoronoiFaceIntegrator> + Send + Sync,
        >],
        scalar_face_integrators: &[Box<
            dyn Fn() -> Box<dyn ScalarVoronoiFaceIntegrator> + Send + Sync,
        >],
    ) -> Self {
        let idx = convex_cell.idx;
        let loc = convex_cell.loc;
        let mut cell = VoronoiCellBuilder::new(loc);

        let mut maybe_faces: Vec<Option<VoronoiFaceBuilder>> =
            (0..convex_cell.clipping_planes.len())
                .map(|_| None)
                .collect();

        fn maybe_init_face<'a>(
            maybe_face: &mut Option<VoronoiFaceBuilder<'a>>,
            half_space: &'a HalfSpace,
            left_idx: usize,
            left_loc: DVec3,
            mask: Option<&[bool]>,
            vector_face_integrators: &[Box<
                dyn Fn() -> Box<dyn VectorVoronoiFaceIntegrator> + Send + Sync,
            >],
            scalar_face_integrators: &[Box<
                dyn Fn() -> Box<dyn ScalarVoronoiFaceIntegrator> + Send + Sync,
            >],
        ) {
            match half_space {
                // Don't construct faces twice in case the voronoi cell of right_idx is also being constructed.
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    ..
                } if *right_idx <= left_idx && mask.map_or(true, |mask| mask[*right_idx]) => (),
                _ => {
                    maybe_face.get_or_insert(VoronoiFaceBuilder::new(
                        left_idx,
                        left_loc,
                        half_space,
                        vector_face_integrators,
                        scalar_face_integrators,
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
            let half_space_0 = &convex_cell.clipping_planes[face_idx_0];
            let half_space_1 = &convex_cell.clipping_planes[face_idx_1];
            let half_space_2 = &convex_cell.clipping_planes[face_idx_2];
            let (maybe_face_0, maybe_face_1, maybe_face_2) =
                maybe_faces.get_3_mut(face_idx_0, face_idx_1, face_idx_2);
            maybe_init_face(
                maybe_face_0,
                half_space_0,
                idx,
                loc,
                mask,
                vector_face_integrators,
                scalar_face_integrators,
            );
            maybe_init_face(
                maybe_face_1,
                half_space_1,
                idx,
                loc,
                mask,
                vector_face_integrators,
                scalar_face_integrators,
            );
            maybe_init_face(
                maybe_face_2,
                half_space_2,
                idx,
                loc,
                mask,
                vector_face_integrators,
                scalar_face_integrators,
            );

            // Project generator on planes
            let plane_0 = &half_space_0.plane;
            let plane_1 = &half_space_1.plane;
            let plane_2 = &half_space_2.plane;
            let g_on_p0 = plane_0.project_onto(loc);
            let g_on_p1 = plane_1.project_onto(loc);
            let g_on_p2 = plane_2.project_onto(loc);

            // Project generator on edges between planes
            let g_on_p01 = plane_0.project_onto_intersection(&plane_1, loc);
            let g_on_p02 = plane_0.project_onto_intersection(&plane_2, loc);
            let g_on_p12 = plane_1.project_onto_intersection(&plane_2, loc);

            // Project generator on vertex determined by planes
            let g_on_p012 = vertex.loc;

            // Calculate signed volumes of tetrahedra
            cell.extend(g_on_p012, g_on_p01, g_on_p0);
            cell.extend(g_on_p012, g_on_p0, g_on_p02);
            cell.extend(g_on_p012, g_on_p1, g_on_p01);
            cell.extend(g_on_p012, g_on_p12, g_on_p1);
            cell.extend(g_on_p012, g_on_p02, g_on_p2);
            cell.extend(g_on_p012, g_on_p2, g_on_p12);

            // Calculate the signed areas of the triangles on the faces and update their barycenters
            maybe_face_0.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p01, g_on_p0);
                f.extend(g_on_p012, g_on_p0, g_on_p02);
            });
            maybe_face_1.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p1, g_on_p01);
                f.extend(g_on_p012, g_on_p12, g_on_p1);
            });
            maybe_face_2.as_mut().map(|f| {
                f.extend(g_on_p012, g_on_p02, g_on_p2);
                f.extend(g_on_p012, g_on_p2, g_on_p12);
            });
        }

        // Filter out uninitialized faces and finalize the rest
        for maybe_face in maybe_faces {
            if let Some(face) = maybe_face {
                faces.push(face);
            }
        }

        cell.build()
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

    /// Get the indices of the faces that have this cell as its left or right neighbour.
    pub fn face_indices<'a>(&'a self, voronoi: &'a Voronoi) -> &[usize] {
        &voronoi.cell_face_connections
            [self.face_connections_offset..(self.face_connections_offset + self.face_count)]
    }

    /// Get an `Iterator` over the Voronoi faces that have this cell as their left _or_ right generator.
    pub fn faces<'a>(&'a self, voronoi: &'a Voronoi) -> impl Iterator<Item = &VoronoiFace> + 'a {
        self.face_indices(voronoi)
            .iter()
            .map(|&i| &voronoi.faces[i])
    }

    /// Get an `Iterator` over the extra vector face integral with given `id` of the faces that have this cell as neighbours.
    pub fn vector_face_integrals<'a>(
        &'a self,
        id: usize,
        voronoi: &'a Voronoi,
    ) -> impl Iterator<Item = DVec3> + 'a {
        self.face_indices(voronoi)
            .iter()
            .map(move |i| voronoi.vector_face_integrals[id][*i])
    }

    /// Get an `Iterator` over the extra scalar face integral with given `id` of the faces that have this cell as neighbours.
    pub fn scalar_face_integrals<'a>(
        &'a self,
        id: usize,
        voronoi: &'a Voronoi,
    ) -> impl Iterator<Item = DVec3> + 'a {
        self.face_indices(voronoi)
            .iter()
            .map(move |i| voronoi.vector_face_integrals[id][*i])
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
    fn test_init_cuboid() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(4.);
        let cell = SimulationBoundary::cuboid(anchor, width, false, DIM3D.into());

        assert_eq!(cell.clipping_planes.len(), 6);
    }

    #[test]
    fn test_clipping() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let loc = DVec3::splat(2.);
        let volume = SimulationBoundary::cuboid(anchor, width, false, DIM3D.into());
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
