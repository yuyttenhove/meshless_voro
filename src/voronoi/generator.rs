use glam::{DVec3};

use super::Dimensionality;

/// A Voronoi generator
#[derive(Clone, Copy, Debug)]
pub struct Generator {
    loc: DVec3,
    id: usize,
}

impl Generator {
    pub(super) fn new(id: usize, loc: DVec3, dimensionality: Dimensionality) -> Self {
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
