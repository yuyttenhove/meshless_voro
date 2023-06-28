use glam::DVec3;

use crate::{
    geometry::{in_sphere_test_exact, intersect_planes, Plane},
    simple_cycle::SimpleCycle,
};

use super::{
    boundary::SimulationBoundary,
    convex_cell_alternative::ConvexCell as ConvexCellAlternative,
    half_space::HalfSpace,
    integrators::{CellIntegral, FaceIntegral},
    Dimensionality, Generator,
};

#[derive(Clone, Debug)]
pub(super) struct Vertex {
    pub loc: DVec3,
    pub dual: [usize; 3],
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
            dual: [i, j, k],
            radius2: gen_loc.distance_squared(d_loc),
        }
    }
}

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

pub(super) struct ConvexCellDecomposition<'a> {
    convex_cell: &'a ConvexCell,
    cur_vertex_idx: usize,
    cur_vertex: &'a Vertex,
    cur_tet_idx: usize,
    projections: [DVec3; 6],
}

impl<'a> ConvexCellDecomposition<'a> {
    fn new(convex_cell: &'a ConvexCell) -> Self {
        let mut decomposition = Self {
            convex_cell,
            cur_vertex_idx: 0,
            cur_vertex: &convex_cell.vertices[0],
            cur_tet_idx: 0,
            projections: [DVec3::ZERO; 6],
        };
        decomposition.load_vertex();
        decomposition
    }

    fn load_vertex(&mut self) {
        self.cur_vertex = &self.convex_cell.vertices[self.cur_vertex_idx];
        for i in 0..3 {
            let cur_plane = &self.convex_cell.clipping_planes[self.cur_vertex.dual[i]].plane;
            let next_plane =
                &self.convex_cell.clipping_planes[self.cur_vertex.dual[(i + 1) % 3]].plane;
            self.projections[2 * i] = cur_plane.project_onto(self.convex_cell.loc);
            self.projections[2 * i + 1] =
                next_plane.project_onto_intersection(&cur_plane, self.convex_cell.loc);
        }
    }
}

impl Iterator for ConvexCellDecomposition<'_> {
    type Item = ConvexCellTet;

    fn next(&mut self) -> Option<Self::Item> {
        // Any vertices left to treat?
        if self.cur_vertex_idx >= self.convex_cell.vertices.len() {
            return None;
        }

        // Get next tet
        let next = ConvexCellTet::new(
            self.projections[self.cur_tet_idx],
            self.projections[(self.cur_tet_idx + 5) % 6],
            self.cur_vertex.loc,
            self.cur_vertex.dual[self.cur_tet_idx / 2],
        );

        self.cur_tet_idx += 1;
        if self.cur_tet_idx == 6 {
            self.cur_tet_idx = 0;
            self.cur_vertex_idx += 1;
            if self.cur_vertex_idx < self.convex_cell.vertices.len() {
                self.load_vertex();
            }
        }

        Some(next)
    }
}

/// Meshless representation of a Voronoi cell as an intersection of halfspaces.
///
/// Can be used to compute integrated cell and face quantities
#[derive(Clone, Debug)]
pub struct ConvexCell {
    pub loc: DVec3,
    pub(super) clipping_planes: Vec<HalfSpace>,
    pub(super) vertices: Vec<Vertex>,
    boundary: SimpleCycle,
    pub(super) safety_radius: f64,
    pub idx: usize,
}

impl ConvexCell {
    /// Initialize each voronoi cell as the bounding box of the simulation volume.
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
        loc: DVec3,
        idx: usize,
        generators: &[Generator],
        mut nearest_neighbours: Box<dyn Iterator<Item = (usize, Option<DVec3>)> + '_>,
        simulation_boundary: &SimulationBoundary,
    ) -> Self {
        let mut cell = ConvexCell::init(loc, idx, simulation_boundary);
        // skip the first nearest neighbour (will be this cell)
        assert_eq!(
            nearest_neighbours
                .next()
                .expect("Nearest neighbours cannot be empty!")
                .0,
            cell.idx,
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

    pub(super) fn decompose(&self) -> ConvexCellDecomposition {
        ConvexCellDecomposition::new(self)
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
                    simulation_boundary.dimensionality,
                ));
                cur = next;
            }
            self.update_safety_radius();
        }
    }

    fn compute_boundary(boundary: &mut SimpleCycle, vertices: &mut [Vertex]) {
        boundary.init(
            vertices[0].dual[0],
            vertices[0].dual[1],
            vertices[0].dual[2],
        );

        for i in 1..vertices.len() {
            // Look for a suitable next vertex to extend the boundary
            let mut idx = i;
            loop {
                assert!(
                    idx < vertices.len(),
                    "No suitable vertex found to extend boundary!"
                );
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

    /// Get the index of the generator on the opposite side of a clipping plane.
    pub fn get_neighbour(&self, clipping_plane_idx: usize) -> Option<usize> {
        self.clipping_planes[clipping_plane_idx].right_idx
    }

    /// Get the shift (if any) associated with a given clipping plane (for applying periodic boundary conditions).
    pub fn get_shift(&self, clipping_plane_idx: usize) -> Option<DVec3> {
        self.clipping_planes[clipping_plane_idx].shift
    }

    pub fn get_plane(&self, clipping_plane_idx: usize) -> Plane {
        self.clipping_planes[clipping_plane_idx].plane.clone()
    }

    /// Compute a custom integrated quantity for this cell.
    pub fn compute_cell_integral<T: CellIntegral>(&self) -> T {
        // Compute integral from decomposition of convex cell
        let mut integrator = T::init(&self);
        for tet in self.decompose() {
            integrator.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }
        integrator.finalize()
    }

    /// Compute a custom integrated quantity for the faces of this cell (non-symmetric version).
    pub fn compute_face_integrals<T: FaceIntegral>(&self) -> Vec<T> {
        // Compute integrals from decomposition of convex cell
        let mut integrals = vec![None; self.clipping_planes.len()];
        for tet in self.decompose() {
            let integral = &mut integrals[tet.plane_idx];
            let integral = integral.get_or_insert_with(|| T::init(self, tet.plane_idx));
            integral.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }

        integrals
            .into_iter()
            .filter_map(|maybe_integral| maybe_integral.map(|integral| integral.finalize()))
            .collect()
    }

    /// Compute a custom integrated quantity for the faces of this cell.
    ///
    /// Symmetric version: skips faces that are shared with active cells with a smaller idx.
    ///
    /// - `mask`: A mask indicating which for which generators convex_cells are actually constructed.
    pub fn compute_face_integrals_sym<T: FaceIntegral>(&self, mask: &[bool]) -> Vec<T> {
        // Compute integrals from decomposition of convex cell
        let mut integrals = vec![None; self.clipping_planes.len()];
        for tet in self.decompose() {
            let integral = &mut integrals[tet.plane_idx];
            if integral.is_none() {
                match &self.clipping_planes[tet.plane_idx] {
                    // If the face is no boundary face and right_idx < this cell's idx corresponds to an active cell, we already treated this face.
                    &HalfSpace {
                        shift: None,
                        right_idx: Some(right_idx),
                        ..
                    } if right_idx < self.idx && mask[right_idx] => continue,
                    _ => (),
                }
            }
            let integral = integral.get_or_insert_with(|| T::init(self, tet.plane_idx));
            integral.collect(tet.vertices[0], tet.vertices[1], tet.vertices[2], self.loc);
        }

        integrals
            .into_iter()
            .filter_map(|maybe_integral| maybe_integral.map(|integral| integral.finalize()))
            .collect()
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
