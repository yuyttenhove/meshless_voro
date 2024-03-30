pub trait GetMutMultiple {
    type Output;

    /// Creates mutable references to elements at 3 indices `i`, `j` and `k`.
    /// These indices are assumed to be distinct, but this is not checked by
    /// this function!
    unsafe fn get_3_mut_unchecked(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
    ) -> (&mut Self::Output, &mut Self::Output, &mut Self::Output);

    /// Creates mutable references to elements at 3 indices `i`, `j` and `k`,
    /// after checking that the indices are indeed all different.
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

#[cfg(test)]
mod test {
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
}
