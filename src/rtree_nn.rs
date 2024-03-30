use std::collections::BinaryHeap;

use glam::DVec3;
use rstar::{Envelope, ParentNode, Point, PointDistance, RTree, RTreeNode, RTreeObject, AABB};

use crate::voronoi::{Dimensionality, Generator};

pub(crate) fn build_rtree(generators: &[Generator]) -> RTree<Generator> {
    RTree::bulk_load(generators.to_vec())
}

pub fn nn_iter<'a>(
    rtree: &'a RTree<Generator>,
    loc: DVec3,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    Box::new(rtree.nearest_neighbor_iter(&[loc.x, loc.y, loc.z]).map(|g| (g.id(), None)))
}

pub(crate) fn wrapping_nn_iter<'a>(
    rtree: &'a RTree<Generator>,
    loc: DVec3,
    width: DVec3,
    dimensionality: Dimensionality,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    let query_point = [loc.x, loc.y, loc.z];
    let width = [width.x, width.y, width.z];
    Box::new(
        RTreeWrappingNearestNeighbourIter::new(rtree.root(), query_point, width, dimensionality)
            .map(move |(g, _distance, shift)| {
                let shift = if shift[0] == 0. && shift[1] == 0. && shift[2] == 0. {
                    None
                } else {
                    Some(-DVec3::from_array(shift))
                };
                (g.id(), shift)
            }),
    )
}

macro_rules! point {
    ($Self:ident) => {
        <<$Self as RTreeObject>::Envelope as Envelope>::Point
    };
}

struct RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance + 'a,
{
    node: &'a RTreeNode<T>,
    distance: <point!(T) as Point>::Scalar,
    shift: point!(T),
}

struct RTreeWrappingNearestNeighbourIter<'a, T>
where
    T: WrappingPointDistance + 'a,
{
    nodes: BinaryHeap<RTreeNodeDistanceWrapper<'a, T>>,
    query_point: point!(T),
}

impl<'a> RTreeWrappingNearestNeighbourIter<'a, Generator> {
    pub fn new(
        root: &'a ParentNode<Generator>,
        query_point: [f64; 3],
        width: [f64; 3],
        dimensionality: Dimensionality,
    ) -> Self {
        let mut result = RTreeWrappingNearestNeighbourIter {
            nodes: BinaryHeap::with_capacity(27),
            query_point,
        };

        // Add the children of this node to the heap and also shifted versions of it for
        // all directions
        let j_range = match dimensionality {
            Dimensionality::Dimensionality2D | Dimensionality::Dimensionality3D => -1..=1,
            Dimensionality::Dimensionality1D => 0..=0,
        };
        let k_range = match dimensionality {
            Dimensionality::Dimensionality3D => -1..=1,
            Dimensionality::Dimensionality1D | Dimensionality::Dimensionality2D => 0..=0,
        };
        for i in -1..=1 {
            for j in j_range.clone() {
                for k in k_range.clone() {
                    let shift = [i as f64 * width[0], j as f64 * width[1], k as f64 * width[2]];
                    result.extend_heap(root.children(), shift);
                }
            }
        }
        result
    }

    fn extend_heap(&mut self, children: &'a [RTreeNode<Generator>], shift: [f64; 3]) {
        let &mut RTreeWrappingNearestNeighbourIter {
            ref mut nodes,
            ref query_point,
        } = self;
        nodes.extend(children.iter().map(|child| {
            let distance = match child {
                RTreeNode::Parent(ref data) => {
                    data.envelope().wrapping_distance_2(query_point, &shift)
                }
                RTreeNode::Leaf(ref t) => t.wrapping_distance_2(query_point, &shift),
            };

            RTreeNodeDistanceWrapper {
                node: child,
                distance,
                shift,
            }
        }));
    }
}

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
        self.loc().distance_squared(DVec3 {
            x: point[0],
            y: point[1],
            z: point[2],
        })
    }
}

impl WrappingPointDistance for Generator {
    fn wrapping_distance_2(&self, point: &[f64; 3], shift: &[f64; 3]) -> f64 {
        let dx = [
            point[0] + shift[0] - self.loc().x,
            point[1] + shift[1] - self.loc().y,
            point[2] + shift[2] - self.loc().z,
        ];

        dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
    }
}

impl WrappingEnvelope for AABB<[f64; 3]> {
    fn wrapping_distance_2(&self, point: &[f64; 3], shift: &[f64; 3]) -> f64 {
        fn clamp(x: f64, min: f64, max: f64) -> f64 {
            x.max(min).min(max)
        }

        let lower = self.lower();
        let upper = self.upper();
        let mut dx = [0., 0., 0.];
        for i in 0..3 {
            dx[i] = clamp(point[i] + shift[i], lower[i], upper[i]) - point[i] - shift[i];
        }

        dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
    }
}

trait WrappingPointDistance: PointDistance {
    fn wrapping_distance_2(
        &self,
        point: &point!(Self),
        shift: &point!(Self),
    ) -> <point!(Self) as Point>::Scalar;
}

trait WrappingEnvelope: Envelope {
    fn wrapping_distance_2(
        &self,
        point: &Self::Point,
        shift: &Self::Point,
    ) -> <Self::Point as Point>::Scalar;
}

impl<'a, T> PartialEq for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<'a, T> PartialOrd for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn partial_cmp(&self, other: &Self) -> Option<::core::cmp::Ordering> {
        // Inverse comparison creates a min heap
        other.distance.partial_cmp(&self.distance)
    }
}

impl<'a, T> Eq for RTreeNodeDistanceWrapper<'a, T> where T: WrappingPointDistance {}

impl<'a, T> Ord for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a> Iterator for RTreeWrappingNearestNeighbourIter<'a, Generator> {
    type Item = (&'a Generator, f64, [f64; 3]);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(current) = self.nodes.pop() {
            match current {
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Parent(ref data),
                    shift,
                    ..
                } => {
                    self.extend_heap(data.children(), shift);
                }
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Leaf(ref t),
                    distance,
                    shift,
                } => {
                    return Some((t, distance, shift));
                }
            }
        }

        None
    }
}
