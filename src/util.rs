use glam::{DVec3, DMat3};

pub trait GetMutMultiple {
    type Output;

    /// Creates mutable references to elements at 3 indices `i`, `j` and `k`.
    /// These indices are assumed to be distinct, but this is not checked by this function!
    unsafe fn get_3_mut_unchecked(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
    ) -> (&mut Self::Output, &mut Self::Output, &mut Self::Output);

    /// Creates mutable references to elements at 3 indices `i`, `j` and `k`, after checking that the indices are indeed all different.
    fn get_3_mut(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
    ) -> (&mut Self::Output, &mut Self::Output, &mut Self::Output) {
        assert!(i != j, "Attempting to index {i} mutably twice!");
        assert!(j != k, "Attempting to index {j} mutably twice!");
        assert!(k != i, "Attempting to index {k} mutably twice!");
        unsafe { self.get_3_mut_unchecked(i, j, k) }
    }
}

impl<T> GetMutMultiple for Vec<T> {
    type Output = T;

    unsafe fn get_3_mut_unchecked(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
    ) -> (&mut Self::Output, &mut Self::Output, &mut Self::Output) {
        let v_i = &mut self[i] as *mut T;
        let v_j = &mut self[j] as *mut T;
        let v_k = &mut self[k] as *mut T;

        (&mut (*v_i), &mut (*v_j), &mut (*v_k))
    }
}

pub fn signed_volume_tet(v0: DVec3, v1: DVec3, v2: DVec3, v3: DVec3) -> f64 {
    let v01 = v1 - v0;
    let v02 = v2 - v0;
    let v03 = v3 - v0;

    DMat3::from_cols(v01, v02, v03).determinant() / 6.
}

/// Calculates the signed area of the ground face `v0`, `v1`, `v2` of the tetrahedron with top `t` and given `volume`.
pub fn signed_area_tri(v0: DVec3, v1: DVec3, v2: DVec3, t: DVec3, volume: f64) -> f64 {
    let n = match (v1 - v0).cross(v2 - v0).try_normalize() {
        Some(n) => n,
        None => return 0.,
    };
    let height = (t - v0).dot(n).abs();
    assert!(height > 0., "Cannot determine signed area of ground face of tetrahedron with height 0!");

    3. * volume / height
}


#[cfg(test)]
mod test {
    use glam::DVec3;

    use super::*;


    #[test]
    #[should_panic]
    fn test_get_3_mut_panic() {
        let mut v = vec![1, 2, 3];
        v.get_3_mut(0, 1, 1);
    }

    #[test]
    fn test_get_3_mut() {
        let mut v = vec![1, 2, 3];
        let (a, b, c) = v.get_3_mut(0, 1, 2);
        *a = 4;
        *b = 5;
        *c = 6;
        assert_eq!(v[0], 4);
        assert_eq!(v[1], 5);
        assert_eq!(v[2], 6);
    }

    #[test]
    fn test_signed_volume() {
        let v0 = DVec3::ZERO;
        let v1 = DVec3::X;
        let v2 = DVec3::Y;
        let v3 = DVec3::Z;

        let volume = signed_volume_tet(v0, v1, v2, v3);
        assert_eq!(volume, 1. / 6.);

        let volume2 = signed_volume_tet(v0, v2, v1, v3);
        assert_eq!(volume, -volume2);
    }

    #[test]
    fn test_signed_area() {
        let v0 = DVec3::ZERO;
        let v1 = DVec3::X;
        let v2 = DVec3::Y;
        let t = DVec3::Z;

        let area = signed_area_tri(v0, v1, v2, t, signed_volume_tet(v0, v1, v2, t));
        assert_eq!(area, 0.5);

        let area2 = signed_area_tri(v0, v2, v1, t, signed_volume_tet(v0, v2, v1, t));
        assert_eq!(area, -area2);

        let t = DVec3::Z + DVec3{ x: 10., y: 10., z: 0. };
        let area3 = signed_area_tri(v0, v1, v2, t, signed_volume_tet(v0, v1, v2, t));
        assert_eq!(area, area3)
    }
}

