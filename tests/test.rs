use float_cmp::assert_approx_eq;
use glam::DVec3 as Vec3;
use meshless_voronoi::{integrals::{AreaIntegral, VolumeIntegral}, Voronoi, VoronoiIntegrator};
use rand::prelude::*;
use rand::rngs::StdRng;

#[macro_export]
macro_rules! log_time {
    ($msg:expr, $x:expr) => {{
        let t0 = std::time::Instant::now();
        let _result = $x;
        let t1 = std::time::Instant::now();
        let dt = t1 - t0;
        println!("{}: {:?}", $msg, dt);
        _result
    }};
}

#[derive(Debug)]
pub struct VCellRaw {
    pub id: i32,
    pub vs: Vec<Vec3>,
    pub faces: Vec<Vec<i32>>,
    pub volume: f64,
    pub position: Vec3,
    pub centroid: Vec3,
    pub face_normals: Vec<Vec3>,
}

pub fn impl_meshless_voro(size: Vec3, points: Vec<Vec3>) -> Vec<VCellRaw> {
    let anchor = Vec3::splat(0.);

    let _voronoi_integrator = log_time!(
        "VoronoiIntegrator::build",
        VoronoiIntegrator::build(
            &points,
            None,
            anchor,
            size,
            3.try_into().unwrap(),
            false,
        )
    );

    let _voronoi2 = log_time!(
        "Voronoi::from(&_voronoi_integrator)",
        Voronoi::from(&_voronoi_integrator)
    );

    
    let (volumes, areas) = log_time!(
    "Manually compute integrals -- sym",
    {
        let volumes = _voronoi_integrator.compute_cell_integrals::<VolumeIntegral>();
        let areas = _voronoi_integrator.compute_face_integrals_sym::<AreaIntegral>();
        (volumes, areas)
    });


    let _voronoi_integrator = log_time!("With faces", _voronoi_integrator.with_faces());
    let _voronoi_raw = log_time!("extract raw info", {
        _voronoi_integrator
            .cells_iter()
            .zip(_voronoi2.cells().into_iter())
            .into_iter()
            .enumerate()
            .map(|(cell_id, (convex_cell, v_cell))| {
                assert!(cell_id == convex_cell.idx);

                assert!(convex_cell.loc == v_cell.loc());

                let vs = convex_cell
                    .vertices
                    .iter()
                    .map(|v| (v.loc - convex_cell.loc))
                    .collect();
                let face_count = v_cell.face_count();

                let v_faces = (0..face_count)
                    .map(|face_idx| {
                        convex_cell
                            .face_vertices(face_idx as usize)
                            .iter()
                            .map(|f| *f as i32)
                            .collect()
                    })
                    .collect();
                let face_norm = v_cell
                    .faces(&_voronoi2)
                    .map(|f| f.normal())
                    .collect();

                // for vx in cell.get_face_vertices(face_idx)
                let cell_position = v_cell.loc();
                VCellRaw {
                    id: cell_id as i32,
                    vs,
                    faces: v_faces,
                    volume: v_cell.volume(),
                    position: cell_position,
                    centroid: v_cell.centroid() - cell_position,
                    face_normals: face_norm,
                }
            })
            .collect()
    });

    let _voronoi3 = log_time!(
        "Voronoi::from(&_voronoi_integrator) -- with faces",
        Voronoi::from(&_voronoi_integrator)
    );

    
    let (volumes2, areas2) = log_time!(
    "Manually compute integrals -- with faces -- sym",
    {
        let volumes = _voronoi_integrator.compute_cell_integrals::<VolumeIntegral>();
        let areas = _voronoi_integrator.compute_face_integrals_sym::<AreaIntegral>();
        (volumes, areas)
    });

    for (volume, volume2) in volumes.iter().zip(volumes2.iter()) {
        assert_approx_eq!(f64, volume.volume, -volume2.volume,  epsilon=1e-10);
    }
    for (area, area2) in areas.iter().zip(areas2.iter()) {
        assert_eq!(area.left(), area2.left());
        assert_eq!(area.right(), area2.right());
        assert_approx_eq!(f64, area.integral().area, -area2.integral().area, epsilon=1e-10)
    }

    assert_eq!(_voronoi2.cells().len(), _voronoi3.cells().len());
    assert_eq!(_voronoi2.cells().len(), volumes.len());
    for (i, (cell, cell2)) in _voronoi2.cells().iter().zip(_voronoi3.cells().iter()).enumerate() {
        assert_approx_eq!(f64, cell.volume(), -cell2.volume(), epsilon=1e-10);
        assert_approx_eq!(f64, cell.volume(), volumes[i].volume);
        assert_eq!(cell.face_count(), cell2.face_count());
        for (face, face2) in cell.faces(&_voronoi2).zip(cell2.faces(&_voronoi3)) {
            // println!("{:}, {:}", face.area(), area.area);
            assert_approx_eq!(f64, face.area(), -face2.area(), epsilon=1e-10);
            // assert_approx_eq!(f64, face.area(), area.area, epsilon=1e-10);
        }
    }

    _voronoi_raw
}

fn random_point(size: Vec3, rng: &mut StdRng) -> Vec3 {
    size * Vec3::new(rng.gen(), rng.gen(), rng.gen())
}

#[test]
fn test_extract_vertices() {
    let count = 4000;
    let size = Vec3::splat(3.);
    let mut rng = StdRng::seed_from_u64(2);
    let generators = (0..count).map(|_| random_point(size, &mut rng)).collect();
    // let generators = vec![Vec3::new(1., 1., 1.), Vec3::new(2., 2., 2.)];
    // let generators = vec![Vec3::new(1., 1., 1.)];

    impl_meshless_voro(size, generators);
}