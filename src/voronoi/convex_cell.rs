use super::{
    boundary::SimulationBoundary, convex_cell_alternative::ConvexCell as ConvexCellAlternative,
    half_space::HalfSpace, Dimensionality, Generator,
};
use crate::{
    geometry::{in_sphere_test_exact, intersect_planes, Plane},
    integrals::{CellIntegralWithData, FaceIntegralWithData},
    simple_cycle::SimpleCycle,
};
use glam::DVec3;
use std::marker::PhantomData;
use std::any::TypeId;

/// A vertex of a [`ConvexCell`].
#[derive(Clone, Debug)]
pub struct Vertex {
    /// The location of the [`Vertex`] (in global coordinates).
    pub loc: DVec3,
    /// The dual representation: the indices of the three half spaces in the corresponding 
    /// vector in the [`ConvexCell`], that intersect at this [`Vertex`] (in counterclockwise 
    /// order around the vertex).
    pub dual: [usize; 3],
    /// The safety radius of this vertex.
    pub (super) radius2: f64,
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
        let loc =
            intersect_planes(&half_spaces[i].plane, &half_spaces[j].plane, &half_spaces[k].plane);
        let d_loc = match dimensionality {
            Dimensionality::OneD => DVec3::new(loc.x, 0., 0.),
            Dimensionality::TwoD => DVec3::new(loc.x, loc.y, 0.),
            Dimensionality::ThreeD => loc,
        };
        Vertex {
            loc,
            dual: [i, j, k],
            radius2: gen_loc.distance_squared(d_loc),
        }
    }

    fn plane_idx(&self, clipping_plane_idx: usize) -> Option<usize> {
        let mut p_idx = 0;
        while self.dual[p_idx] != clipping_plane_idx {
            p_idx += 1;
            if p_idx == 3 {
                return None;
            }
        }
        Some(p_idx)
    }
}

/// An oriented tetrahedron, part of a decomposition of a [`ConvexCell`].
/// The generator of the corresponding [`ConvexCell`] is forms the top of the
/// tetrahedron.
///
/// We use the following orientation convention:
///
/// - If the three vertices are ordered counterclockwise as seen from the top,
///   the tetrahedron is assumed to be part of the `ConvexCell` and should
///   contribute from any integrals that will be computed.
///
/// - If the vertices are ordered clockwise, the tetrahedron should subtract
///   from the integrals, to correct for another tetrahedron that was not fully
///   contained within the `ConvexCell`.
pub(super) struct ConvexCellTet {
    pub plane_idx: usize,
    pub vertices: [DVec3; 3],
}

impl ConvexCellTet {
    pub fn new(v0: DVec3, v1: DVec3, v2: DVec3, plane_idx: usize) -> Self {
        Self {
            vertices: [v0, v1, v2],
            plane_idx,
        }
    }
}

struct DecompositionWithFaces<'a> {
    cur_face_idx: usize,
    cur_vertex_idx: usize,
    convex_cell: &'a ConvexCell<WithFaces>
}

impl<'a> DecompositionWithFaces<'a> {
    fn new(convex_cell: &'a ConvexCell<WithFaces>) -> Self {
        Self { cur_face_idx: 0, cur_vertex_idx: 1, convex_cell }
    }

    fn next(&mut self) -> Option<ConvexCellTet> {
        if self.cur_face_idx >= self.convex_cell.face_count() {
            return None;
        }

        // Get face vertices
        let vertices = self.convex_cell.face_vertices(self.cur_face_idx);

        // Construct next tet
        let next = ConvexCellTet::new(
            self.convex_cell.vertices[vertices[0]].loc,
            self.convex_cell.vertices[vertices[self.cur_vertex_idx]].loc,
            self.convex_cell.vertices[vertices[self.cur_vertex_idx + 1]].loc,
            self.convex_cell.faces()[self.cur_face_idx].clipping_plane,
        );

        // update variables
        self.cur_vertex_idx += 1;
        if self.cur_vertex_idx > self.convex_cell.faces()[self.cur_face_idx].vertex_count - 2 {
            self.cur_vertex_idx = 1;
            self.cur_face_idx += 1;
        }

        Some(next)
    }
}

struct DecompositionWithoutFaces {
    cur_vertex_idx: usize,
    cur_vertex: Vertex,
    cur_tet_idx: usize,
    projections: [DVec3; 6],
}

impl DecompositionWithoutFaces {
    fn new<M: ConvexCellMarker>(convex_cell: &ConvexCell<M>) -> Self {
        let mut decomposition = Self {
            cur_vertex_idx: 0,
            cur_vertex: convex_cell.vertices[0].clone(),
            cur_tet_idx: 0,
            projections: [DVec3::ZERO; 6],
        };
        decomposition.load_vertex(convex_cell);
        decomposition
    }

    fn load_vertex<M: ConvexCellMarker>(&mut self, convex_cell: &ConvexCell<M>) {
        self.cur_vertex = convex_cell.vertices[self.cur_vertex_idx].clone();
        for i in 0..3 {
            let cur_plane = &convex_cell.clipping_planes[self.cur_vertex.dual[i]].plane;
            let next_plane =
                &convex_cell.clipping_planes[self.cur_vertex.dual[(i + 1) % 3]].plane;
            self.projections[2 * i] = cur_plane.project_onto(convex_cell.loc);
            self.projections[2 * i + 1] =
                next_plane.project_onto_intersection(cur_plane, convex_cell.loc);
        }
    }

    fn next<M: ConvexCellMarker>(&mut self, convex_cell: &ConvexCell<M>) -> Option<ConvexCellTet> {
        // Any vertices left to treat?
        if self.cur_vertex_idx >= convex_cell.vertices.len() {
            return None;
        }

        // Construct next tet
        let next = ConvexCellTet::new(
            self.projections[self.cur_tet_idx],
            self.projections[(self.cur_tet_idx + 5) % 6],
            self.cur_vertex.loc,
            self.cur_vertex.dual[self.cur_tet_idx / 2],
        );

        // Update indices for constructiong the next valid tetrahedron
        self.cur_tet_idx += 1;
        if self.cur_tet_idx == 6 {
            self.cur_tet_idx = 0;
            self.cur_vertex_idx += 1;
            if self.cur_vertex_idx < convex_cell.vertices.len() {
                self.load_vertex(convex_cell);
            }
        }

        Some(next)
    }
}

enum Decomposition<'a> {
    WithFaces(DecompositionWithFaces<'a>),
    WithoutFaces(DecompositionWithoutFaces),
}

/// Decompose a [`ConvexCell`] into an iterator of oriented tetrahedra.
///
/// Useful to compute integrals, such as volume, area, centroid, etc. for
/// [`ConvexCell`]s.
pub(super) struct ConvexCellDecomposition<'a, M: ConvexCellMarker> {
    convex_cell: &'a ConvexCell<M>,
    inner: Decomposition<'a>,
}

impl<'a,  M: ConvexCellMarker + 'static> ConvexCellDecomposition<'a, M> {
    fn new(convex_cell: &'a ConvexCell<M>) -> Self {
        let decomposition = if TypeId::of::<M>() == TypeId::of::<WithFaces>() {
            // Safety: we know that the convex_cell is with faces 
            unsafe {
                let convex_cell_ref: &ConvexCell<WithFaces> = std::mem::transmute(convex_cell);
                Decomposition::WithFaces(DecompositionWithFaces::new(convex_cell_ref))
            }
        } else {
            Decomposition::WithoutFaces(DecompositionWithoutFaces::new(convex_cell))
        };
        Self {
            convex_cell, inner: decomposition,
        }
    }
}

impl<M: ConvexCellMarker> Iterator for ConvexCellDecomposition<'_, M> {
    type Item = ConvexCellTet;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            Decomposition::WithFaces(ref mut decomposition) => decomposition.next(),
            Decomposition::WithoutFaces(ref mut decomposition) => decomposition.next(self.convex_cell),
        }
    }
}

pub(crate) trait ConvexCellMarker: Clone + Send + Sync + Default {}

#[derive(Copy, Clone, Default)]
pub struct WithoutFaces;
impl ConvexCellMarker for WithoutFaces {}
#[derive(Copy, Clone, Default)]
pub struct WithFaces;
impl ConvexCellMarker for WithFaces {}

/// Meshless representation of a Voronoi cell as an intersection of
/// [`HalfSpace`]s.
///
/// Can be used to compute integrated cell and face quantities. 
/// In this representation, the vertices of the Voronoi cell are also available.
#[derive(Clone, Debug)]
pub struct ConvexCell<T: ConvexCellMarker> {
    /// The index (label) of the generator of this [`ConvexCell`]
    pub idx: usize,
    /// The location of the generator of this [`ConvexCell`].
    pub loc: DVec3,
    /// [`HalfSpace`]s that intersect to form this [`ConvexCell`]. Their normals
    /// are pointed inwards.
    pub clipping_planes: Vec<HalfSpace>,
    /// The vertices of this cell (in global coordinates).
    pub vertices: Vec<Vertex>,
    /// The actual faces
    faces: Option<Vec<ConvexCellFace>>,
    /// List containing the indices of the vertices of the faces
    face_vertex_connections: Option<Vec<usize>>,
    boundary: SimpleCycle,
    pub(super) safety_radius: f64,
    pub(super) dimensionality: Dimensionality,
    _phantom: PhantomData<T>,
}

impl ConvexCell<WithoutFaces> {
    pub(crate) fn new(loc: DVec3, idx: usize, clipping_planes: Vec<HalfSpace>, vertices: Vec<Vertex>, dimensionality: Dimensionality) -> Self {
        let n_planes = clipping_planes.len();
        Self {
            loc, idx, clipping_planes, vertices, dimensionality,
            boundary: SimpleCycle::new(n_planes),
            faces: None,
            face_vertex_connections: None,
            safety_radius: 0.,
            _phantom: PhantomData,
        }
    }

    /// Initialize each Voronoi cell as the bounding box of the simulation
    /// volume.
    pub(super) fn init(loc: DVec3, idx: usize, simulation_boundary: &SimulationBoundary) -> Self {
        let clipping_planes = simulation_boundary.clipping_planes.clone();

        let dimensionality = simulation_boundary.dimensionality;
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

        let mut cell = ConvexCell::new(loc, idx, clipping_planes, vertices, dimensionality);
        cell.update_safety_radius();
        cell
    }

    /// Build the Convex cell by repeatedly intersecting it with the appropriate
    /// half spaces
    pub(super) fn build(
        loc: DVec3,
        idx: usize,
        generators: &[Generator],
        mut nearest_neighbours: Box<dyn Iterator<Item = (usize, Option<DVec3>)> + '_>,
        simulation_boundary: &SimulationBoundary,
    ) -> Self {
        let mut cell = ConvexCell::init(loc, idx, simulation_boundary);
        // skip the first nearest neighbour (will be this cell)
        assert_eq!(
            nearest_neighbours.next().expect("Nearest neighbours cannot be empty!").0,
            cell.idx,
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
            let dx = cell.loc - ngb_loc;
            let dist = dx.length();
            assert!(dist.is_finite() && dist > 0.0, "Degenerate point set!");
            if cell.safety_radius < dist {
                return cell;
            }
            let n = dx / dist;
            let p = 0.5 * (cell.loc + ngb_loc);
            cell.clip_by_plane(
                HalfSpace::new(n, p, Some(idx), shift),
                generators,
                simulation_boundary,
            );
        }

        cell
    }

    pub(super) fn clip_by_plane(
        &mut self,
        p: HalfSpace,
        generators: &[Generator],
        simulation_boundary: &SimulationBoundary,
    ) {
        // loop over vertices and remove the ones clipped by p
        let mut i = 0;
        let mut num_v = self.vertices.len();
        let mut num_r = 0;
        while i < num_v {
            let mut clip = p.clip(self.vertices[i].loc);
            if clip == 0. {
                // Do the equivalent in-sphere test to determine whether a vertex is clipped
                let dual = self.vertices[i].dual;
                let a = simulation_boundary.iloc(self.loc);
                let b = simulation_boundary
                    .iloc(self.clipping_planes[dual[0]].right_loc(self.idx, generators));
                let c = simulation_boundary
                    .iloc(self.clipping_planes[dual[1]].right_loc(self.idx, generators));
                let d = simulation_boundary
                    .iloc(self.clipping_planes[dual[2]].right_loc(self.idx, generators));
                let v = simulation_boundary.iloc(p.right_loc(self.idx, generators));
                clip = in_sphere_test_exact(&a, &b, &c, &d, &v);
            }
            if clip < 0. {
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
            // Compute the boundary of the (dual) topological triangulated disk around the
            // vertices to be removed.
            Self::compute_boundary(&mut self.boundary, &mut self.vertices[num_v..]);
            let mut boundary = self.boundary.iter().take(self.boundary.len + 1);
            // finally we can *realy* remove the vertices.
            self.vertices.truncate(num_v);
            // Add new vertices constructed from the new clipping plane and the boundary
            let mut cur = boundary.next().expect("Boundary contains at least 3 elements");
            for next in boundary {
                self.vertices.push(Vertex::from_dual(
                    cur,
                    next,
                    p_idx,
                    &self.clipping_planes,
                    self.loc,
                    simulation_boundary.dimensionality,
                ));
                cur = next;
            }
            self.update_safety_radius();
        }
    }

    fn compute_boundary(boundary: &mut SimpleCycle, vertices: &mut [Vertex]) {
        boundary.init(vertices[0].dual[0], vertices[0].dual[1], vertices[0].dual[2]);

        for i in 1..vertices.len() {
            // Look for a suitable next vertex to extend the boundary
            let mut idx = i;
            loop {
                assert!(idx < vertices.len(), "No suitable vertex found to extend boundary!");
                let vertex = &vertices[idx].dual;
                match boundary.try_extend(vertex[0], vertex[1], vertex[2]) {
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

    /// Convert this [`ConvexCell`] into one with face information stored. 
    /// This makes it impossible to clip this [`ConvexCell`] with additional half spaces.
    /// 
    /// NOTE: this is only supported for 3D Voronoi meshes, since the underlying 
    /// representation is always 3D,  this method would lead to additional, nonsensical vertices 
    /// along the extra dimensions in 1D or 2D. The function panics when invoked in 1D or 2D.
    pub fn with_faces(mut self) -> ConvexCell<WithFaces> {
        assert_eq!(self.dimensionality, Dimensionality::ThreeD, "Can only convert to WithFaces in 3D!");
        
        // Collect the vertices on all clipping planes (if any)
        let mut face_vertex_connections = vec![vec![]; self.clipping_planes.len()];
        for (idx, vertex) in self.vertices.iter().enumerate() {
            face_vertex_connections[vertex.dual[0]].push(idx);
            face_vertex_connections[vertex.dual[1]].push(idx);
            face_vertex_connections[vertex.dual[2]].push(idx);
        }
        // Sort the face vertices in counterclockwise order for each face
        face_vertex_connections.iter_mut().enumerate().for_each(|(clipping_plane_idx, vertices)| self.sort_face_vertices(vertices, clipping_plane_idx));

        // Create a new face if the corresponding half space contains in some vertices
        let mut offset = 0;
        let faces = face_vertex_connections.iter().enumerate().filter_map(|(id, vertices)| {
            if vertices.len() == 0 {
                return None;
            }
            let face = ConvexCellFace {
                clipping_plane: id,
                vertex_count: vertices.len(),
                vertex_offset: offset,
            };
            offset += face.vertex_count;
            Some(face)
        }).collect();

        // Set faces and face_vertex_connections
        self.faces = Some(faces);
        self.face_vertex_connections = Some(face_vertex_connections.into_iter().flatten().collect());

        ConvexCell::transition(self)

    }

    /// Sort the vertices on the given clipping plane, given the current vertex
    fn sort_face_vertices(&self, vert_idx: &mut [usize], clipping_plane_idx: usize) {
        // Empty face?
        if vert_idx.len() == 0 {
            return;
        }

        let mut cur_v = &self.vertices[vert_idx[0]];
        let mut p_idx_in_cur_v = cur_v.plane_idx(clipping_plane_idx).expect("Plane contained in vertex by construction");
        // Get the other plane contained in the next vertex
        let mut next_plane = cur_v.dual[(p_idx_in_cur_v + 2) % 3];
        
        let mut cur_idx = 1;
        while cur_idx < vert_idx.len() - 1 {
            let mut test_idx = cur_idx;
            // loop through the tail of the list and swap the next vertex to cur_idx
            while test_idx < vert_idx.len() {
                cur_v = &self.vertices[vert_idx[test_idx]];
                p_idx_in_cur_v = cur_v.plane_idx(clipping_plane_idx).expect("All given vertices must contain clipping plane");
                if cur_v.dual.contains(&next_plane) {
                    next_plane = cur_v.dual[(p_idx_in_cur_v + 2) % 3];
                    vert_idx.swap(cur_idx, test_idx);
                    cur_idx += 1;
                    break;
                }
                test_idx += 1;
            }
            assert!(test_idx < vert_idx.len(), "There always must be a next vertex connected to the current one!");
        }
    }
}

impl<M: ConvexCellMarker + 'static> ConvexCell<M> {

    /// Safely transition between two states
    fn transition<N: ConvexCellMarker>(self) -> ConvexCell<N> {
        let ConvexCell {
            idx, loc, clipping_planes, vertices, faces, face_vertex_connections, boundary, safety_radius, dimensionality, _phantom: _
        } = self;

        ConvexCell::<N> {
            idx, loc, clipping_planes, vertices, faces, face_vertex_connections, boundary, safety_radius, dimensionality, _phantom: PhantomData,
        }
    }

    pub(super) fn decompose(&self) -> ConvexCellDecomposition<M> {
        ConvexCellDecomposition::<M>::new(self)
    }

    /// Compute a custom integrated quantity for this cell.
    pub fn compute_cell_integral<D: Copy, T: CellIntegralWithData<D>>(&self, extra_data: D) -> T {
        // Compute integral from decomposition of convex cell
        let mut integrator = T::init_with_data(self, extra_data);
        for tet in self.decompose() {
            integrator.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }
        integrator.finalize()
    }

    fn clipping_plane_has_valid_dimensionality(&self, plane_idx: usize) -> bool {
        self.dimensionality.vector_is_valid(self.clipping_planes[plane_idx].normal())
    }

    /// Compute a custom integrated quantity for the faces of this cell
    /// (non-symmetric version).
    pub fn compute_face_integrals<D: Copy, T: FaceIntegralWithData<D>>(
        &self,
        extra_data: D,
    ) -> Vec<T> {
        // Compute integrals from decomposition of convex cell
        let mut integrals = vec![None; self.clipping_planes.len()];
        for tet in self.decompose() {
            // Only compute integrals for faces of valid dimensionality
            if !self.clipping_plane_has_valid_dimensionality(tet.plane_idx) {
                continue;
            }
            let integral = &mut integrals[tet.plane_idx];
            let integral =
                integral.get_or_insert_with(|| T::init_with_data(self, tet.plane_idx, extra_data));
            integral.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }

        integrals
            .into_iter()
            .filter_map(|maybe_integral| maybe_integral.map(|integral| integral.finalize()))
            .collect()
    }

    /// Compute a custom integrated quantity for the faces of this cell.
    ///
    /// Symmetric version: skips faces that are shared with active cells with a
    /// smaller idx.
    ///
    /// - `mask`: A mask indicating which for which generators convex cells are
    ///   actually constructed.
    pub fn compute_face_integrals_sym<D: Copy, T: FaceIntegralWithData<D>>(
        &self,
        extra_data: D,
        mask: &[bool],
    ) -> Vec<T> {
        // Compute integrals from decomposition of convex cell
        let mut integrals = vec![None; self.clipping_planes.len()];
        for tet in self.decompose() {
            // Only compute integrals for faces of valid dimensionality
            if !self.clipping_plane_has_valid_dimensionality(tet.plane_idx) {
                continue;
            }
            let integral = &mut integrals[tet.plane_idx];
            if integral.is_none() {
                match &self.clipping_planes[tet.plane_idx] {
                    // If the face is no boundary face and right_idx < this cell's idx corresponds
                    // to an active cell, we already treated this face.
                    &HalfSpace {
                        shift: None,
                        right_idx: Some(right_idx),
                        ..
                    } if right_idx < self.idx && mask[right_idx] => continue,
                    _ => (),
                }
            }
            let integral =
                integral.get_or_insert_with(|| T::init_with_data(self, tet.plane_idx, extra_data));
            integral.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }

        integrals
            .into_iter()
            .filter_map(|maybe_integral| maybe_integral.map(|integral| integral.finalize()))
            .collect()
    }
}

impl ConvexCell<WithFaces> {
    /// Discard the computed `faces` and `face_vertex_connections`, making this [`ConvexCell`] 
    /// safe for clipping by additional half spaces again. 
    pub fn discard_faces(mut self) -> ConvexCell<WithoutFaces> {
        self.faces = None;
        self.face_vertex_connections = None;
        ConvexCell::transition(self)
    }

    /// The number of *actual* faces (might be less than the number of [`HalfSpace`]s used 
    /// in the construction of this cell).
    pub fn face_count(&self) -> usize {
        self.faces().len()
    }

    fn faces(&self) -> &[ConvexCellFace] {
        // Safety: This state of the ConvexCell can only exist if the faces are initialized
        unsafe{
            self.faces.as_ref().unwrap_unchecked()
        }
    }

    fn face_vertex_connections(&self) -> &[usize] {
        // Safety: This state of the ConvexCell can only exist if the faces are initialized
        unsafe{
            self.face_vertex_connections.as_ref().unwrap_unchecked()
        }
    }

    fn half_space(&self, face_idx: usize) -> &HalfSpace {
        &self.clipping_planes[self.faces()[face_idx].clipping_plane]
    }

    /// Get the corresponding clipping plane associated with a given face.
    pub fn clipping_plane(&self, face_idx: usize) -> &Plane {
        &self.half_space(face_idx).plane
    }

    /// Get the index of the generator on the opposite side of a face.
    /// Returns None for boundary faces
    pub fn neighbour(&self, face_idx: usize) -> Option<usize> {
        self.half_space(face_idx).right_idx
    }

    /// Get the shift (if any) associated with a given face (for
    /// applying periodic boundary conditions).
    pub fn shift(&self, face_idx: usize) -> Option<DVec3> {
        self.half_space(face_idx).shift
    }

    /// Get the count of the face vertices 
    pub fn face_vertex_count(&self, face_idx: usize) -> usize {
        self.faces()[face_idx].vertex_count
    }

    /// Get the indices of the vertices of a given face in this cell's [`Vertex`] list. 
    pub fn face_vertices(&self, face_idx: usize) -> &[usize] {
        let face = &self.faces()[face_idx];
        &self.face_vertex_connections()[face.vertex_offset..face.vertex_offset+face.vertex_count]
    }
}

#[derive(Clone, Debug)]
struct ConvexCellFace {
    /// The index of the clipping plane/half-space of the [`ConvexCell`] corresponding to this face.
    clipping_plane: usize,
    vertex_count: usize,
    vertex_offset: usize,
}

impl From<ConvexCellAlternative> for ConvexCell<WithoutFaces> {
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
                    Dimensionality::ThreeD,
                )
            })
            .collect::<Vec<_>>();

        ConvexCell::new(
            convex_cell_alt.loc,convex_cell_alt.idx, clipping_planes, vertices, convex_cell_alt.dimensionality 
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_cuboid() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(4.);
        let cell = SimulationBoundary::cuboid(anchor, width, false, Dimensionality::ThreeD.into());

        assert_eq!(cell.clipping_planes.len(), 6);
    }

    #[test]
    fn test_clipping() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let loc = DVec3::splat(2.);
        let volume =
            SimulationBoundary::cuboid(anchor, width, false, Dimensionality::ThreeD.into());
        let mut cell = ConvexCell::init(loc, 0, &volume);

        let ngb = DVec3::splat(2.5);
        let generators = [
            Generator::new(0, loc, Dimensionality::ThreeD.into()),
            Generator::new(1, ngb, Dimensionality::ThreeD.into()),
        ];
        let dx = cell.loc - ngb;
        let dist = dx.length();
        let n = dx / dist;
        let p = 0.5 * (cell.loc + ngb);
        cell.clip_by_plane(HalfSpace::new(n, p, Some(1), Some(DVec3::ZERO)), &generators, &volume);

        assert_eq!(cell.clipping_planes.len(), 7)
    }

    #[test]
    fn test_get_face_vertices_cuboid() {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let loc = DVec3::splat(2.);
        let volume = SimulationBoundary::cuboid(anchor, width, false, Dimensionality::ThreeD.into());
        let cell = ConvexCell::init(loc, 0, &volume).with_faces();

        assert_eq!(cell.face_count(), 6);
        for i in 0..6 {
            let face_vertices = cell.face_vertices(i);
            assert!(face_vertices.len() == 4);
            for vi in face_vertices {
                assert!(cell.vertices[*vi].dual.contains(&i));
                assert!(face_vertices.iter().filter(|&vj| *vi == *vj).count() == 1);
                println!("{:?}", cell.vertices[*vi].loc)
            }
            println!("------------")
        }
    }
}
