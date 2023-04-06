use glam::DVec3;

use crate::{
    geometry::Plane,
    voronoi::{
        half_space::HalfSpace,
        voronoi_face::{VoronoiFace, VoronoiFaceBuilder},
        Voronoi,
    },
};

use super::{
    convex_cell::ConvexCell,
    integrators::{VolumeCentroidIntegrator, VoronoiIntegrator},
    Dimensionality,
};

/// A Voronoi cell.
#[derive(Default, Debug, Clone)]
pub struct VoronoiCell {
    loc: DVec3,
    centroid: DVec3,
    volume: f64,
    safety_radius: f64,
    face_connections_offset: usize,
    face_count: usize,
}

impl VoronoiCell {
    fn init(loc: DVec3, centroid: DVec3, volume: f64, safety_radius: f64) -> Self {
        Self {
            loc,
            centroid,
            volume,
            safety_radius,
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
        build_all_faces: bool,
        dimensionality: Dimensionality,
    ) -> Self {
        let idx = convex_cell.idx;
        let loc = convex_cell.loc;
        let mut volume_centroid_integral = VolumeCentroidIntegrator::init();

        let mut maybe_faces: Vec<Option<VoronoiFaceBuilder<'a>>> =
            (0..convex_cell.clipping_planes.len())
                .map(|_| None)
                .collect();

        let maybe_init_face = |maybe_face: &mut Option<VoronoiFaceBuilder<'a>>,
                               half_space: &'a HalfSpace| {
            let should_construct_face = build_all_faces
                || match half_space {
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
                            && (*right_idx > idx || mask.map_or(false, |mask| !mask[*right_idx]))
                    }
                    _ => true,
                };
            if should_construct_face {
                maybe_face.get_or_insert(VoronoiFaceBuilder::new(idx, loc, half_space));
            }
        };

        // Loop over the decomposition of this convex cell into tetrahedra to compute the necessary integrals/barycenter calculations
        for tet in convex_cell.decompose() {
            // Update the volume and centroid of the cell
            volume_centroid_integral.collect(
                tet.vertices[0],
                tet.vertices[1],
                tet.vertices[2],
                loc,
            );

            // Initialize a new face if necessary
            let maybe_face = &mut maybe_faces[tet.plane_idx];
            maybe_init_face(maybe_face, &convex_cell.clipping_planes[tet.plane_idx]);
            // Update this face's area and centroid if necessary
            if let Some(face) = maybe_face {
                face.extend(tet.vertices[0], tet.vertices[1], tet.vertices[2])
            }
        }
        // Filter out uninitialized faces and finalize the rest
        for maybe_face in maybe_faces {
            if let Some(face) = maybe_face {
                faces.push(face.build());
            }
        }

        let (volume, centroid) = volume_centroid_integral.finalize();

        VoronoiCell::init(loc, centroid, volume, convex_cell.safety_radius)
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

    /// Get the safety radius of this cell
    pub fn safety_radius(&self) -> f64 {
        self.safety_radius
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
    use crate::voronoi::{boundary::SimulationBoundary, half_space::HalfSpace, Generator};

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
        let mut cell = ConvexCell::init(loc, 0, &volume);

        let ngb = DVec3::splat(2.5);
        let generators = [
            Generator::new(0, loc, DIM3D.into()),
            Generator::new(1, ngb, DIM3D.into()),
        ];
        let dx = cell.loc - ngb;
        let dist = dx.length();
        let n = dx / dist;
        let p = 0.5 * (cell.loc + ngb);
        cell.clip_by_plane(
            HalfSpace::new(n, p, Some(1), Some(DVec3::ZERO)),
            &generators,
            &volume,
        );

        assert_eq!(cell.clipping_planes.len(), 7)
    }
}
