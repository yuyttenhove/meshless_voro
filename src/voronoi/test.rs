use super::*;
use float_cmp::assert_approx_eq;
use rand::{distributions::Uniform, prelude::*};

fn perturbed_grid(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
    let mut generators = vec![];
    let mut rng = thread_rng();
    let distr = Uniform::new(-0.5, 0.5);
    for n in 0..count.pow(3) {
        let i = n / count.pow(2);
        let j = (n % count.pow(2)) / count;
        let k = n % count;
        generators.push(
            DVec3 {
                x: i as f64 + 0.5 + pert * rng.sample(distr),
                y: j as f64 + 0.5 + pert * rng.sample(distr),
                z: k as f64 + 0.5 + pert * rng.sample(distr),
            } * width
                / count as f64
                + anchor,
        );
    }

    generators
}

fn perturbed_plane(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
    let mut generators = vec![];
    let mut rng = thread_rng();
    let distr = Uniform::new(-0.5, 0.5);
    for n in 0..count.pow(2) {
        let i = n / count;
        let j = n % count;
        generators.push(
            DVec3 {
                x: i as f64 + 0.5 + pert * rng.sample(distr),
                y: j as f64 + 0.5 + pert * rng.sample(distr),
                z: 0.5 * count as f64,
            } * width
                / count as f64
                + anchor,
        );
    }

    generators
}

#[test]
fn test_init_voronoi_cell() {
    let anchor = DVec3::splat(1.);
    let width = DVec3::splat(4.);
    let loc = DVec3::splat(3.);
    let cell = ConvexCell::init(loc, anchor, width, 0);

    assert_eq!(cell.vertices.len(), 8);
    assert_eq!(cell.clipping_planes.len(), 6);
    assert_eq!(cell.safety_radius, 2. * 12f64.sqrt())
}

#[test]
fn test_clipping() {
    let anchor = DVec3::splat(1.);
    let width = DVec3::splat(2.);
    let loc = DVec3::splat(2.);
    let mut cell = ConvexCell::init(loc, anchor, width, 0);

    let ngb = DVec3::splat(2.5);
    let dx = cell.loc - ngb;
    let dist = dx.length();
    let n = dx / dist;
    let p = 0.5 * (cell.loc + ngb);
    cell.clip_by_plane(HalfSpace::new(n, p, Some(1)));

    assert_eq!(cell.clipping_planes.len(), 7)
}

#[test]
fn test_single_cell() {
    let generators = vec![DVec3::splat(0.5)];
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let voronoi = Voronoi::build(&generators, anchor, width, 0);
    assert_approx_eq!(f64, voronoi.cells[0].volume, 1.);
}

#[test]
fn test_two_cells() {
    let generators = vec![
        DVec3 {
            x: 0.3,
            y: 0.4,
            z: 0.25,
        },
        DVec3 {
            x: 0.7,
            y: 0.6,
            z: 0.75,
        },
    ];
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let voronoi = Voronoi::build(&generators, anchor, width, 1);
    assert_approx_eq!(f64, voronoi.cells[0].volume, 0.5);
    assert_approx_eq!(f64, voronoi.cells[1].volume, 0.5);
}

#[test]
fn test_4_cells() {
    let generators = vec![
        DVec3 {
            x: 0.2,
            y: 0.3,
            z: 0.5,
        },
        DVec3 {
            x: 0.8,
            y: 0.2,
            z: 0.5,
        },
        DVec3 {
            x: 0.3,
            y: 0.8,
            z: 0.5,
        },
        DVec3 {
            x: 0.7,
            y: 0.7,
            z: 0.5,
        },
    ];
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let voronoi = Voronoi::build(&generators, anchor, width, 3);
    voronoi.save();
}

#[test]
fn test_five_cells() {
    let delta = 0.1f64.sqrt();
    let generators = vec![
        DVec3 {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        },
        DVec3 {
            x: 0.5 - delta,
            y: 0.5 - delta,
            z: 0.5,
        },
        DVec3 {
            x: 0.5 - delta,
            y: 0.5 + delta,
            z: 0.5,
        },
        DVec3 {
            x: 0.5 + delta,
            y: 0.5 + delta,
            z: 0.5,
        },
        DVec3 {
            x: 0.5 + delta,
            y: 0.5 - delta,
            z: 0.5,
        },
    ];
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let voronoi = Voronoi::build(&generators, anchor, width, 4);
    assert_approx_eq!(f64, voronoi.cells[0].volume, 0.2);
    assert_approx_eq!(f64, voronoi.cells[1].volume, 0.2);
    assert_approx_eq!(f64, voronoi.cells[2].volume, 0.2);
    assert_approx_eq!(f64, voronoi.cells[3].volume, 0.2);
    assert_approx_eq!(f64, voronoi.cells[4].volume, 0.2);
}

#[test]
fn test_eight_cells() {
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, 2, 0.);
    let voronoi = Voronoi::build(&generators, anchor, width, 7);
    for cell in &voronoi.cells {
        assert_approx_eq!(f64, cell.volume, 0.125);
    }
}

#[test]
fn test_27_cells() {
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, 3, 0.);
    let voronoi = Voronoi::build(&generators, anchor, width, 26);
    for cell in &voronoi.cells {
        assert_approx_eq!(f64, cell.volume, 1. / 27.);
    }
}

#[test]
fn test_64_cells() {
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, 4, 0.);
    let voronoi = Voronoi::build(&generators, anchor, width, 27);
    for cell in &voronoi.cells {
        assert_approx_eq!(f64, cell.volume, 1. / 64.);
    }
}

#[test]
fn test_125_cells() {
    let pert = 0.5;
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, 5, pert);
    let voronoi = Voronoi::build(&generators, anchor, width, 40);
    let mut total_volume = 0.;
    for cell in &voronoi.cells {
        total_volume += cell.volume;
    }
    assert_approx_eq!(f64, total_volume, 1.)
}

#[test]
fn test_3_d() {
    let pert = 0.5;
    let count = 32;
    let anchor = DVec3::ZERO;
    let width = DVec3::splat(2.);
    let generators = perturbed_grid(anchor, width, count, pert);
    let voronoi = Voronoi::build(&generators, anchor, width, 60);
    let total_volume: f64 = voronoi.cells.iter().map(|c| c.volume).sum();
    assert_eq!(voronoi.cells.len(), generators.len());
    assert_approx_eq!(f64, total_volume, 8., epsilon=1e-10, ulps=8);
}

#[test]
fn test_2_d() {
    let pert = 0.75;
    let count = 10;
    let anchor = DVec3::splat(2.);
    let width = DVec3{ x: 2., y: 2., z: 1.};
    let generators = perturbed_plane(anchor, width, count, pert);
    let voronoi = Voronoi::build(&generators, anchor, width, count * count - 1);
    voronoi.save();
    
    assert_approx_eq!(f64, voronoi.cells.iter().map(|c| c.volume).sum(), 4.);
}
