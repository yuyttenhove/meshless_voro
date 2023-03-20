use glam::DVec3;

use crate::{
    geometry::Plane,
    integrators::{VolumeCentroidIntegrator, VoronoiCellIntegrator},
    util::GetMutMultiple,
    voronoi::{
        half_space::HalfSpace,
        voronoi_face::{VoronoiFace, VoronoiFaceBuilder},
        Voronoi,
    },
};

use super::{convex_cell::ConvexCell, Dimensionality};

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
    pub(super) fn from_convex_cell<'a>(
        convex_cell: &'a ConvexCell,
        faces: &mut Vec<VoronoiFace>,
        mask: Option<&[bool]>,
        dimensionality: Dimensionality,
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
            dimensionality: Dimensionality,
        ) {
            let should_construct_face = match half_space {
                // Don't construct non-boundary faces twice.
                HalfSpace {
                    right_idx: Some(right_idx),
                    shift: None,
                    plane: Plane { n, .. },
                    ..
                } => {
                    // Only construct face if:
                    // - normal has right dimensionality
                    // - other neighbour has not been treated yet or is inactive
                    dimensionality.vector_is_valid(*n)
                        && (*right_idx > left_idx || mask.map_or(false, |mask| !mask[*right_idx]))
                }
                _ => true,
            };
            if should_construct_face {
                maybe_face.get_or_insert(VoronoiFaceBuilder::new(left_idx, left_loc, half_space));
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
            maybe_init_face(maybe_face_0, half_space_0, idx, loc, mask, dimensionality);
            maybe_init_face(maybe_face_1, half_space_1, idx, loc, mask, dimensionality);
            maybe_init_face(maybe_face_2, half_space_2, idx, loc, mask, dimensionality);

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
                faces.push(face.build());
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
    use crate::voronoi::boundary::SimulationBoundary;

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
