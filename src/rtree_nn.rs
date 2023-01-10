use glam::DVec3;
use rstar::{RTree, RTreeObject, AABB, PointDistance};

use crate::voronoi::Generator;

impl RTreeObject for Generator {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
            AABB::from_point([self.loc().x, self.loc().y, self.loc().z])
    }
}

impl PointDistance for Generator {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.loc().distance_squared(DVec3 { x: point[0], y: point[1], z: point[2] })
    }
}

pub fn build_rtree(generators: &[Generator]) -> RTree<Generator> {
    RTree::bulk_load(generators.to_vec())
}

pub fn nn_iter<'a>(rtree: &'a RTree<Generator>, loc: DVec3) -> Box<dyn Iterator<Item = usize> + 'a> {
    Box::new(rtree.nearest_neighbor_iter(&[loc.x, loc.y, loc.z]).map(|g| g.id()))
}