use std::{fs::File, io::Write};

use glam::{DMat3, DVec3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{
    rtree_nn::{build_rtree, nn_iter, wrapping_nn_iter},
    simple_cycle::SimpleCycle,
    util::{signed_area_tri, signed_volume_tet, GetMutMultiple},
};

#[cfg(test)]
mod test;

struct HalfSpace {
    n: DVec3,
    p: DVec3,
    d: f64,
    right_idx: Option<usize>,
    shift: Option<DVec3>,
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

    fn project_onto(&self, vertex: DVec3) -> DVec3 {
        vertex + (self.p - vertex).project_onto(self.n)
    }

    /// Project the a vertex on the intersection of two planes.
    fn project_onto_intersection(&self, other: &Self, vertex: DVec3) -> DVec3 {
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
    fn init(
        loc: DVec3,
        anchor: DVec3,
        width: DVec3,
        idx: usize,
        dimensionality: Dimensionality,
    ) -> Self {
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
        let mut cell = ConvexCell {
            loc,
            clipping_planes,
            vertices,
            safety_radius: 0.,
            idx,
        };
        cell.update_safety_radius(dimensionality);
        cell
    }

    /// Build the Convex cell by repeatedly intersecting it with the appropriate half spaces
    fn build(
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
                ngb_loc = generator.loc + shift;
            } else {
                ngb_loc = generator.loc;
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

/// A Voronoi generator
#[derive(Clone, Copy, Debug)]
pub struct Generator {
    loc: DVec3,
    id: usize,
}

impl Generator {
    fn new(id: usize, loc: DVec3, dimensionality: Dimensionality) -> Self {
        let mut loc = loc;
        match dimensionality {
            Dimensionality::Dimensionality1D => {
                loc.y = 0.;
                loc.z = 0.;
            }
            Dimensionality::Dimensionality2D => loc.z = 0.,
            _ => (),
        }
        Self { loc, id }
    }

    /// Get the id of this generator
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the position of this generator
    pub fn loc(&self) -> DVec3 {
        self.loc
    }
}

/// A Voronoi face between two neighbouring generators.
pub struct VoronoiFace {
    left: usize,
    right: Option<usize>,
    area: f64,
    centroid: DVec3,
    normal: DVec3,
    shift: Option<DVec3>,
}

impl VoronoiFace {
    fn init(left: usize, right: Option<usize>, normal: DVec3, shift: Option<DVec3>) -> Self {
        VoronoiFace {
            left,
            right,
            area: 0.,
            centroid: DVec3::ZERO,
            normal,
            shift,
        }
    }

    fn finalize(&mut self) {
        let area_inv = if self.area != 0. { 1. / self.area } else { 0. };
        self.centroid *= area_inv;
    }

    fn has_valid_dimensionality(&self, dimensionality: Dimensionality) -> bool {
        match dimensionality {
            Dimensionality::Dimensionality1D => self.normal.y == 0. && self.normal.z == 0.,
            Dimensionality::Dimensionality2D => self.normal.z == 0.,
            Dimensionality::Dimensionality3D => true,
        }
    }

    /// Get the index of the generator on the _left_ of this face.
    pub fn left(&self) -> usize {
        self.left
    }

    /// Get the index of the generator on the _right_ of this face.
    /// Returns `None` if if this is a boundary face (i.e. obtained by clipping a Voronoi cell with the _simulation volume_).
    pub fn right(&self) -> Option<usize> {
        self.right
    }

    /// Get the area of this face.
    pub fn area(&self) -> f64 {
        self.area
    }

    /// Get the position of the centroid of this face.
    pub fn centroid(&self) -> DVec3 {
        self.centroid
    }

    /// Get a normal vector to this face, pointing away from the _left_ generator.
    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    /// Get the shift vector (if any) to apply to the generator to the right of this face to bring it to the reference frame of this face.
    /// Can only be `Some` for periodic Voronoi tesselations.
    pub fn shift(&self) -> Option<DVec3> {
        self.shift
    }
}

/// A Voronoi cell.
#[derive(Debug, Clone, Copy)]
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
    fn from_convex_cell(convex_cell: &ConvexCell, faces: &mut Vec<VoronoiFace>) -> Self {
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
            // If the faces are None and we want to create a face, initialize it now.
            match plane_0 {
                // Don't construct faces twice
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    ..
                } if *right_idx <= idx => (),
                _ => {
                    face_0.get_or_insert(VoronoiFace::init(
                        idx,
                        plane_0.right_idx,
                        -plane_0.n,
                        plane_0.shift,
                    ));
                }
            }
            match plane_1 {
                // Don't construct faces twice
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    ..
                } if *right_idx <= idx => (),
                _ => {
                    face_1.get_or_insert(VoronoiFace::init(
                        idx,
                        plane_1.right_idx,
                        -plane_1.n,
                        plane_1.shift,
                    ));
                }
            }
            match plane_2 {
                // Don't construct faces twice
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    ..
                } if *right_idx <= idx => (),
                _ => {
                    face_2.get_or_insert(VoronoiFace::init(
                        idx,
                        plane_2.right_idx,
                        -plane_2.n,
                        plane_2.shift,
                    ));
                }
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
            let frac_1_3 = 1. / 3.;
            face_0.as_mut().map(|f| {
                let v_001 = signed_area_tri(g_on_p012, g_on_p01, g_on_p0, loc, v_001);
                let v_002 = signed_area_tri(g_on_p012, g_on_p0, g_on_p02, loc, v_002);
                f.area += v_001 + v_002;
                f.centroid += frac_1_3
                    * (v_001 * (g_on_p0 + g_on_p01 + g_on_p012)
                        + v_002 * (g_on_p0 + g_on_p02 + g_on_p012));
            });
            face_1.as_mut().map(|f| {
                let v_101 = signed_area_tri(g_on_p012, g_on_p1, g_on_p01, loc, v_101);
                let v_112 = signed_area_tri(g_on_p012, g_on_p12, g_on_p1, loc, v_112);
                f.area += v_101 + v_112;
                f.centroid += frac_1_3
                    * (v_101 * (g_on_p1 + g_on_p01 + g_on_p012)
                        + v_112 * (g_on_p1 + g_on_p12 + g_on_p012));
            });
            face_2.as_mut().map(|f| {
                let v_202 = signed_area_tri(g_on_p012, g_on_p02, g_on_p2, loc, v_202);
                let v_212 = signed_area_tri(g_on_p012, g_on_p2, g_on_p12, loc, v_212);
                f.area += v_202 + v_212;
                f.centroid += frac_1_3
                    * (v_202 * (g_on_p2 + g_on_p02 + g_on_p012)
                        + v_212 * (g_on_p2 + g_on_p12 + g_on_p012));
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

#[derive(Clone, Copy)]
pub enum Dimensionality {
    Dimensionality1D,
    Dimensionality2D,
    Dimensionality3D,
}

impl From<usize> for Dimensionality {
    fn from(u: usize) -> Self {
        match u {
            1 => Dimensionality::Dimensionality1D,
            2 => Dimensionality::Dimensionality2D,
            3 => Dimensionality::Dimensionality3D,
            _ => panic!("Invalid Voronoi dimensionality!"),
        }
    }
}

/// The main Voronoi struct
pub struct Voronoi {
    anchor: DVec3,
    width: DVec3,
    cells: Vec<VoronoiCell>,
    faces: Vec<VoronoiFace>,
    cell_face_connections: Vec<usize>,
    dimensionality: Dimensionality,
}

impl Voronoi {
    /// Construct the Voronoi tesselation. This method runs in parallel if the `"rayon"` feature is enabled.
    ///
    /// Iteratively construct each Voronoi cell independently of each other by repeatedly clipping it by the nearest generators until a safety criterion is reached.
    /// For non-periodic Voronoi tesselations, all Voronoi cells are clipped by the simulation volume with given `anchor` and `width` if necessary.
    ///
    /// * `generators` - The seed points of the Voronoi cells.
    /// * `anchor` - The lower left corner of the simulation volume.
    /// * `width` - The width of the simulation volume. Also determines the period of periodic Voronoi tesselations.
    /// * `dimensionality` - The dimensionality of the Voronoi tesselation. The algorithm is mainly aimed at constructiong 3D Voronoi tesselations, but can be used for 1 or 2D as well.
    /// * `periodic` - Whether to apply periodic boundary conditions to the Voronoi tesselation.
    pub fn build(
        generators: &[DVec3],
        anchor: DVec3,
        width: DVec3,
        dimensionality: usize,
        periodic: bool,
    ) -> Self {
        let dimensionality = dimensionality.into();

        // Normalize the unused components of the simulation volume, so that the lower dimensional volumes will be correct.
        let mut this_anchor = anchor;
        let mut this_width = width;
        if periodic {
            this_anchor -= width;
            this_width *= 3.;
        }

        match dimensionality {
            Dimensionality::Dimensionality1D => {
                this_anchor.y = -0.5;
                this_anchor.z = -0.5;
                this_width.y = 1.;
                this_width.z = 1.
            }
            Dimensionality::Dimensionality2D => {
                this_anchor.z = -0.5;
                this_width.z = 1.;
            }
            _ => (),
        }

        let generators: Vec<Generator> = generators
            .iter()
            .enumerate()
            .map(|(id, &loc)| Generator::new(id, loc, dimensionality))
            .collect();

        let rtree = build_rtree(&generators);

        let mut faces: Vec<Vec<VoronoiFace>> = generators.iter().map(|_| vec![]).collect();
        #[cfg(feature = "rayon")]
        let cells = generators
            .par_iter()
            .zip(faces.par_iter_mut())
            .map(|(generator, faces)| {
                let mut convex_cell = ConvexCell::init(
                    generator.loc,
                    this_anchor,
                    this_width,
                    generator.id,
                    dimensionality,
                );
                let nearest_neighbours = if periodic {
                    wrapping_nn_iter(&rtree, generator.loc, width, dimensionality)
                } else {
                    nn_iter(&rtree, generator.loc)
                };
                convex_cell.build(&generators, nearest_neighbours, dimensionality);
                VoronoiCell::from_convex_cell(&convex_cell, faces)
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let cells = generators
            .iter()
            .zip(faces.iter_mut())
            .map(|(generator, faces)| {
                let mut convex_cell = ConvexCell::init(
                    generator.loc,
                    this_anchor,
                    this_width,
                    generator.id,
                    dimensionality,
                );
                let nearest_neighbours = if periodic {
                    wrapping_nn_iter(&rtree, generator.loc, width, dimensionality)
                } else {
                    nn_iter(&rtree, generator.loc)
                };
                convex_cell.build(&generators, nearest_neighbours, dimensionality);
                VoronoiCell::from_convex_cell(&convex_cell, faces)
            })
            .collect();

        Voronoi {
            anchor: this_anchor,
            width: this_width,
            cells,
            faces: faces
                .into_iter()
                .flatten()
                .filter(|f| f.has_valid_dimensionality(dimensionality))
                .collect(),
            cell_face_connections: vec![],
            dimensionality,
        }
        .finalize()
    }

    /// Link the Voronoi faces to their respective cells.
    fn finalize(mut self) -> Self {
        let mut cell_face_connections: Vec<Vec<usize>> =
            (0..self.cells.len()).map(|_| vec![]).collect();

        for (i, face) in self.faces.iter().enumerate() {
            cell_face_connections[face.left].push(i);
            match face {
                // Only add if non boundary face (periodic/box)
                VoronoiFace {
                    right: Some(right_idx),
                    shift: None,
                    ..
                } => cell_face_connections[*right_idx].push(i),
                _ => (),
            }
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

    /// The anchor of the simulation volume. All generators are assumed to be contained in this simulation volume.
    pub fn anchor(&self) -> DVec3 {
        self.anchor
    }

    /// The width of the simulation volume. All generators are assumed to be contained in this simulation volume.
    pub fn width(&self) -> DVec3 {
        self.width
    }

    /// Get the voronoi cells.
    pub fn cells(&self) -> &[VoronoiCell] {
        self.cells.as_ref()
    }

    /// Get the voronoi faces.
    pub fn faces(&self) -> &[VoronoiFace] {
        self.faces.as_ref()
    }

    /// Get a vector of the Voronoi faces by consuming the Voronoi struct.
    pub fn into_faces(self) -> Vec<VoronoiFace> {
        self.faces
    }

    /// Get the links between the cells and their faces.
    pub fn cell_face_connections(&self) -> &[usize] {
        self.cell_face_connections.as_ref()
    }

    /// Dump the cell and face info to 2 files called "faces.txt" and "cells.txt".
    ///
    /// Mainly for debugging purposes.
    pub fn save(&self) -> Result<(), std::io::Error> {
        let mut file = File::create("faces.txt")?;
        for face in &self.faces {
            let centroid = face.centroid;
            let n = face.normal;
            match self.dimensionality {
                Dimensionality::Dimensionality1D => {
                    writeln!(file, "{}\t({})\t({})", face.area, centroid.x, n.x,)?
                }
                Dimensionality::Dimensionality2D => writeln!(
                    file,
                    "{}\t({}, {})\t({}, {})",
                    face.area, centroid.x, centroid.y, n.x, n.y,
                )?,
                Dimensionality::Dimensionality3D => writeln!(
                    file,
                    "{}\t({}, {}, {})\t({}, {}, {})",
                    face.area, centroid.x, centroid.y, centroid.z, n.x, n.y, n.z,
                )?,
            }
        }
        let mut file = File::create("cells.txt").unwrap();
        for cell in &self.cells {
            writeln!(
                file,
                "{}\t({}, {}, {})\t({}, {}, {})",
                cell.volume,
                cell.loc.x,
                cell.loc.y,
                cell.loc.z,
                cell.centroid.x,
                cell.centroid.y,
                cell.centroid.z
            )?;
        }

        Ok(())
    }
}
