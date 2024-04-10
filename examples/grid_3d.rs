//! Example that builds a Voronoi tesselation for a (perturbed) grid of n^3 points in 3 dimensions.
//!
//! Run with: `cargo run --release --example grid_3d`
//!
//! The number of points and the size of perturbations can optionally be given as command line
//! arguments: `cargo run --release --example grid_3d -- [n] [p]`
//!
//! Their default values are:
//! - `n`: 64
//! - `p`: 0.95

extern crate glam;
extern crate meshless_voronoi;
extern crate rand;

use glam::DVec3;
use meshless_voronoi::Voronoi;
use rand::{distributions::Uniform, prelude::*};
use std::convert::TryInto;
use std::env;

fn perturbed_grid(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
    let mut generators = vec![];
    let mut rng = thread_rng();
    let distr = Uniform::new(-0.5, 0.5);
    for n in 0..count.pow(3) {
        let i = n / count.pow(2);
        let j = (n % count.pow(2)) / count;
        let k = n % count;
        let pos = DVec3 {
            x: i as f64 + 0.5 + pert * rng.sample(distr),
            y: j as f64 + 0.5 + pert * rng.sample(distr),
            z: k as f64 + 0.5 + pert * rng.sample(distr),
        } * width
            / count as f64
            + anchor;
        generators.push(pos.clamp(anchor, anchor + width));
    }

    generators
}

fn main() {
    let mut args = env::args().skip(1);
    let count = match args.next() {
        Some(n) => n.parse::<usize>().expect(
            "The first argument should be an integer denoting the grid size along one dimension!",
        ),
        None => 64,
    };
    let pert = match args.next() {
        Some(p) => p.parse::<f64>().expect(
            "The second argument should be a number between 0 and 1 denoting the size of the grid perturbations!"
        ),
        None => 0.95,
    };

    let anchor = DVec3::splat(0.);
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, count, pert);
    let _voronoi = Voronoi::build(&generators, anchor, width, 3.try_into().unwrap(), false);
}
