use crate::part::Part;

use glam::{DVec3, UVec3};
use std::{cmp::Ordering, collections::BinaryHeap};

struct Cell {
    loc: DVec3,
    width: DVec3,
    offset: usize,
    count: usize,
}

impl Cell {
    /// Closest point within a cell to a given point. If the point is in the
    /// cell, the point itself is returned.
    fn closest_loc(&self, pos: DVec3) -> DVec3 {
        let mut res = self.loc;

        if pos.x > self.loc.x {
            res.x = pos.x.min(self.loc.x + self.width.x);
        }
        if pos.y > self.loc.y {
            res.y = pos.y.min(self.loc.y + self.width.y);
        }
        if pos.z > self.loc.z {
            res.z = pos.z.min(self.loc.z + self.width.z);
        }

        res
    }

    fn min_distance_squared(&self, pos: DVec3) -> f64 {
        self.closest_loc(pos).distance_squared(pos)
    }

    /// The minimal distance to the faces of a cell from a point within that
    /// cell.
    fn min_distance_to_face(&self, pos: DVec3) -> f64 {
        (pos.x - self.loc.x)
            .min(self.loc.x + self.width.x - pos.x)
            .min(pos.y - self.loc.y)
            .min(self.loc.y + self.width.y - pos.y)
            .min(pos.z - self.loc.z)
            .min(self.loc.z + self.width.z - pos.z)
    }
}

pub struct Space {
    anchor: DVec3,
    width: DVec3,
    cdim: UVec3,
    cells: Vec<Cell>,
    parts: Vec<Part>,
}

impl Space {
    pub fn new(anchor: DVec3, width: DVec3, max_cell_width: f64) -> Self {
        let mut cells = vec![];
        let cdim = (width / max_cell_width).ceil();
        let c_width = width / cdim;
        let cdim = cdim.as_uvec3();
        for i in 0..cdim.x {
            for j in 0..cdim.y {
                for k in 0..cdim.z {
                    cells.push(Cell {
                        loc: anchor
                            + DVec3 {
                                x: i as f64 * c_width.x,
                                y: j as f64 * c_width.x,
                                z: k as f64 * c_width.x,
                            },
                        width: c_width,
                        offset: 0,
                        count: 0,
                    })
                }
            }
        }

        Space {
            anchor,
            width,
            cdim,
            cells,
            parts: vec![],
        }
    }

    pub fn add_parts(&mut self, positions: &[DVec3]) {
        // Determine the cid of all positions and create parts
        let mut parts: Vec<Part> = positions
            .iter()
            .enumerate()
            .map(|(pid, p_x)| {
                let rel_pos = *p_x - self.anchor;
                assert!(
                    rel_pos.x >= 0.
                        && rel_pos.x < self.width.x
                        && rel_pos.y >= 0.
                        && rel_pos.y < self.width.y
                        && rel_pos.z >= 0.
                        && rel_pos.z < self.width.z,
                    "Part falls outside domain!"
                );

                let i = (rel_pos.x / self.width.x * self.cdim.x as f64).floor() as i32;
                let j = (rel_pos.y / self.width.y * self.cdim.y as f64).floor() as i32;
                let k = (rel_pos.z / self.width.z * self.cdim.z as f64).floor() as i32;
                let cid = self.get_cid(i, j, k).expect("Index out of bounds during construction!");
                Part::new(*p_x, cid, pid)
            })
            .collect();

        // sort by cid
        parts.sort_by_key(|p_a| p_a.cid());

        // add parts to space and set cell offsets and counts
        let mut offset = 0;
        let mut count = 0;
        let mut prev_cid = parts[0].cid();
        for part in parts.iter() {
            if part.cid() != prev_cid {
                self.cells[prev_cid].offset = offset;
                self.cells[prev_cid].count = count;
                offset += count;
                count = 0;
                prev_cid = part.cid();
            }
            count += 1;
        }
        self.cells[prev_cid].offset = offset;
        self.cells[prev_cid].count = count;

        self.parts = parts;
    }

    pub fn knn(&self, k: usize) -> Vec<Vec<usize>> {
        /// HeapEntry struct used to build MaxHeap of nearest neighbours of
        /// particles.
        #[derive(PartialEq)]
        struct HeapEntry {
            idx: usize,
            d_2: f64,
        }

        impl Eq for HeapEntry {}

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.d_2.partial_cmp(&other.d_2)
            }
        }

        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }

        // Check that this is possible
        assert!(k < self.count());

        // Initialize nearest-neighbour matrix
        let mut nn: Vec<Vec<usize>> = (0..self.count()).map(|_| vec![0; k]).collect();

        // loop over parts and find their nearest neighbours
        for part in self.parts.iter() {
            let mut h = BinaryHeap::<HeapEntry>::new();
            let cid = part.cid();
            let dist_to_face = self.cells[cid].min_distance_to_face(part.x());

            let mut r = 0;
            loop {
                if k == 0 {
                    // Nothing to do here
                    break;
                }

                let r_ring = self.get_r_ring(cid, r);
                for ngb_cid in r_ring {
                    let ngb_cell = &self.cells[ngb_cid];

                    // Can we safely skip this cell?
                    if h.len() == k
                        && h.peek().expect("Heap cannot be empty!").d_2
                            < ngb_cell.min_distance_squared(part.x())
                    {
                        continue;
                    }

                    // Check the parts of this cell
                    let ngb_parts =
                        &self.parts[ngb_cell.offset..(ngb_cell.offset + ngb_cell.count)];

                    for ngb_part in ngb_parts.iter() {
                        // Skip the part itself
                        if part.id() == ngb_part.id() {
                            continue;
                        }

                        let d_2 = part.distance_squared(ngb_part);
                        if h.len() < k {
                            h.push(HeapEntry {
                                idx: ngb_part.id(),
                                d_2,
                            });
                        } else {
                            let max_d_2 = h.peek().expect("Heap cannot be empty!").d_2;
                            if d_2 < max_d_2 {
                                h.pop();
                                h.push(HeapEntry {
                                    idx: ngb_part.id(),
                                    d_2,
                                });
                            }
                        }
                    }
                }

                let min_dist_to_ring = dist_to_face + r as f64 * self.cells[0].width.min_element();
                if h.len() == k
                    && min_dist_to_ring * min_dist_to_ring
                        > h.peek().expect("Heap cannot be emtpy!").d_2
                {
                    break;
                }
                r += 1;
            }

            // now collect the particles nearest neighbours from the MaxHeap in increasing
            // distance
            debug_assert_eq!(h.len(), k);
            for i in (0..k).rev() {
                let entry =
                    h.pop().expect("We should be able to pop k entries from a Heap of length k");
                nn[part.id()][i] = entry.idx;
            }
        }

        nn
    }

    fn get_r_ring(&self, cid: usize, r: i32) -> Vec<usize> {
        if r == 0 {
            return vec![cid];
        }

        let mut r_ring = vec![];
        let cid = cid as u32;
        let i = cid / (self.cdim.y * self.cdim.z);
        let j = (cid % (self.cdim.y * self.cdim.z)) / self.cdim.z;
        let k = cid % self.cdim.z;
        for di in -r..=r {
            for dj in -r..=r {
                for dk in -r..=r {
                    if di.abs().max(dj.abs()).max(dk.abs()) < r {
                        continue;
                    }
                    if let Some(r_cid) = self.get_cid(i as i32 + di, j as i32 + dj, k as i32 + dk) {
                        r_ring.push(r_cid);
                    }
                }
            }
        }

        r_ring
    }

    fn get_cid(&self, i: i32, j: i32, k: i32) -> Option<usize> {
        if i < 0 || j < 0 || k < 0 {
            return None;
        }
        let (i, j, k) = (i as u32, j as u32, k as u32);
        if i >= self.cdim.x || j >= self.cdim.y || k >= self.cdim.z {
            return None;
        }
        Some((i * self.cdim.y * self.cdim.z + j * self.cdim.z + k) as usize)
    }

    fn count(&self) -> usize {
        self.parts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions::Uniform, prelude::*};

    fn example_space() -> Space {
        let anchor = DVec3::splat(1.);
        let width = DVec3::splat(2.);
        let mut space = Space::new(anchor, width, 0.5);
        let mut p_x = vec![];
        let mut rng = thread_rng();
        let distr = Uniform::new(0., 0.5);
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let cell_anchor = anchor
                        + DVec3 {
                            x: i as f64 * 0.5,
                            y: j as f64 * 0.5,
                            z: k as f64 * 0.5,
                        };
                    for _ in 0..3 {
                        // Add 3 parts per cell
                        let rel_pos = DVec3 {
                            x: rng.sample(distr),
                            y: rng.sample(distr),
                            z: rng.sample(distr),
                        };
                        p_x.push(cell_anchor + rel_pos);
                    }
                }
            }
        }
        space.add_parts(&p_x);
        space
    }

    #[test]
    fn test_construction() {
        let space = example_space();

        let mut tot_count = 0;
        for (cid, cell) in space.cells.iter().enumerate() {
            tot_count += cell.count;
            assert!(cell.offset + cell.count <= space.count());
            assert_eq!(cell.count, 3);
            for part in space.parts[cell.offset..cell.offset + cell.count].iter() {
                assert_eq!(part.cid(), cid);
                let p_x = part.x();
                assert!(p_x.x >= cell.loc.x);
                assert!(p_x.y >= cell.loc.y);
                assert!(p_x.z >= cell.loc.z);
                assert!(p_x.x < cell.loc.x + cell.width.x);
                assert!(p_x.y < cell.loc.y + cell.width.y);
                assert!(p_x.z < cell.loc.z + cell.width.z);
            }
        }
        assert!(tot_count == space.count());
    }

    #[test]
    fn test_get_r_ring() {
        let space = Space::new(DVec3::ZERO, DVec3::splat(2.5), 0.5);

        let center_cid = space.get_cid(2, 2, 2).unwrap();

        let ring_0 = space.get_r_ring(center_cid, 0);
        assert_eq!(ring_0.len(), 1);
        assert_eq!(ring_0[0], center_cid);

        let ring_1 = space.get_r_ring(center_cid, 1);
        assert_eq!(ring_1.len(), 26);

        let ring_2 = space.get_r_ring(center_cid, 2);
        assert_eq!(ring_2.len(), 98);

        let edge_cid = space.get_cid(0, 1, 1).unwrap();
        let ring_1 = space.get_r_ring(edge_cid, 1);
        assert_eq!(ring_1.len(), 17);
    }

    #[test]
    fn test_knn() {
        let space = example_space();

        let k = 25;
        let knn = space.knn(k);

        // brute force check
        for (i, part) in space.parts.iter().enumerate() {
            let nn = &knn[i];

            // check that the neighbours are ordered in increasing distance:
            for j in 1..k {
                assert!(
                    part.distance_squared(&space.parts[nn[j]])
                        > part.distance_squared(&space.parts[nn[j - 1]])
                )
            }

            // Get max distance to knn
            let max_d_2 = part.distance_squared(&space.parts[nn[k - 1]]);

            // loop over the other parts and check that they are either in the nearest
            // neighbours or farther away than max_d_2
            for (j, other) in space.parts.iter().enumerate() {
                if j == i {
                    continue;
                }
                if part.distance_squared(other) > max_d_2 {
                    assert!(!nn.contains(&j));
                } else {
                    assert!(nn.contains(&j));
                }
            }
        }
    }
}
