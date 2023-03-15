use glam::DVec3;

use crate::integrators::{
    AreaCentroidIntegrator, ScalarVoronoiFaceIntegrator, VectorVoronoiFaceIntegrator,
    VoronoiFaceIntegrator,
};

use super::{voronoi_cell::HalfSpace, Dimensionality};

pub struct VoronoiFaceBuilder<'a> {
    left_idx: usize,
    left_loc: DVec3,
    right_loc: DVec3,
    half_space: &'a HalfSpace,
    area_centroid: AreaCentroidIntegrator,
    vector_face_integrators: Vec<Box<dyn VectorVoronoiFaceIntegrator>>,
    scalar_face_integrators: Vec<Box<dyn ScalarVoronoiFaceIntegrator>>,
}

impl<'a> VoronoiFaceBuilder<'a> {
    pub fn new(
        left_idx: usize,
        left_loc: DVec3,
        half_space: &'a HalfSpace,
        vector_face_integrals: &[Box<
            dyn Fn() -> Box<dyn VectorVoronoiFaceIntegrator> + Send + Sync,
        >],
        scalar_face_integrals: &[Box<
            dyn Fn() -> Box<dyn ScalarVoronoiFaceIntegrator> + Send + Sync,
        >],
    ) -> Self {
        let half_loc = half_space.project_onto(left_loc);
        let vector_face_integrators = vector_face_integrals
            .iter()
            .map(|get_integrator| get_integrator())
            .collect();
        let scalar_face_integrators = scalar_face_integrals
            .iter()
            .map(|get_integrator| get_integrator())
            .collect();
        Self {
            left_idx,
            left_loc,
            right_loc: 2. * half_loc - left_loc,
            half_space,
            area_centroid: AreaCentroidIntegrator::init(),
            vector_face_integrators,
            scalar_face_integrators,
        }
    }

    pub fn extend(&mut self, v0: DVec3, v1: DVec3, v2: DVec3) {
        self.area_centroid
            .collect(v0, v1, v2, self.left_loc, self.right_loc);
        for integrator in self.vector_face_integrators.iter_mut() {
            integrator.collect(v0, v1, v2, self.left_loc, self.right_loc)
        }
        for integrator in self.scalar_face_integrators.iter_mut() {
            integrator.collect(v0, v1, v2, self.left_loc, self.right_loc)
        }
    }

    pub(crate) fn has_valid_dimensionality(&self, dimensionality: Dimensionality) -> bool {
        let normal = self.half_space.normal();
        match dimensionality {
            Dimensionality::Dimensionality1D => normal.y == 0. && normal.z == 0.,
            Dimensionality::Dimensionality2D => normal.z == 0.,
            Dimensionality::Dimensionality3D => true,
        }
    }

    pub fn build(&self) -> (VoronoiFace, Vec<DVec3>, Vec<f64>) {
        let (area, centroid) = self.area_centroid.finalize();
        let vector_integrals = self
            .vector_face_integrators
            .iter()
            .map(|integrator| integrator.finalize())
            .collect();
        let scalar_integrals = self
            .scalar_face_integrators
            .iter()
            .map(|integrator| integrator.finalize())
            .collect();
        (
            VoronoiFace::new(
                self.left_idx,
                self.half_space.right_idx,
                area,
                centroid,
                -self.half_space.normal(),
                self.half_space.shift,
            ),
            vector_integrals,
            scalar_integrals,
        )
    }
}

/// A Voronoi face between two neighbouring generators.
#[derive(Debug, Clone)]
pub struct VoronoiFace {
    left: usize,
    right: Option<usize>,
    area: f64,
    centroid: DVec3,
    normal: DVec3,
    shift: Option<DVec3>,
}

impl VoronoiFace {
    pub(super) fn new(
        left: usize,
        right: Option<usize>,
        area: f64,
        centroid: DVec3,
        normal: DVec3,
        shift: Option<DVec3>,
    ) -> Self {
        VoronoiFace {
            left,
            right,
            area,
            centroid,
            normal,
            shift,
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
