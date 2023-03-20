use glam::DVec3;

use super::{half_space::HalfSpace, Dimensionality};

#[derive(Clone)]
pub(super) struct SimulationBoundary {
    pub clipping_planes: Vec<HalfSpace>,
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
