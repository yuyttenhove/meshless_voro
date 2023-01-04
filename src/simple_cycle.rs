use std::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Debug, PartialEq, Eq)]
pub struct SimpleCycle<T> {
    values: Vec<T>,
}

impl<T> Index<usize> for SimpleCycle<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index % self.values.len()]
    }
}

impl<T> IndexMut<usize> for SimpleCycle<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T> Deref for SimpleCycle<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.values.deref()
    }
}

impl<T> DerefMut for SimpleCycle<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.values.deref_mut()
    }
}

impl<T> SimpleCycle<T> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn new(a: T, b: T, c: T) -> Self {
        Self {
            values: vec![a, b, c],
        }
    }

    #[cfg(test)]
    fn from_vec(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn push(&mut self, value: T) {
        self.values.push(value);
    }

    pub fn insert(&mut self, index: usize, element: T) {
        if index == self.len() {
            self.push(element);
        } else {
            self.values.insert((index) % self.len(), element);
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        self.values.remove(index % self.len())
    }
}

impl<T: Eq + Copy> SimpleCycle<T> {
    pub fn is_valid(&self) -> bool {
        for (i, v) in self.iter().enumerate() {
            for (j, w) in self.iter().enumerate() {
                if i != j && v == w {
                    return false;
                }
            }
        }
        true
    }

    /// Extend the boundary of a topological triangulated disk with a new triangle.
    ///
    /// Vertices of the boundary and of the new triangle are assumed to be ordered counterclockwise.
    pub fn try_extend(&mut self, a: T, b: T, c: T) -> Result<(), ()> {
        let tri = SimpleCycle::new(a, b, c);

        // find matching vertices and disolvable edges
        let mut disolvable_edges = vec![];
        let mut num_matching_vertices = 0;
        for i in 0..3 {
            // look for this edge in self
            for j in 0..self.len() {
                if self[j + 1] == tri[i] && self[j] == tri[i + 1] {
                    disolvable_edges.push((i, j));
                }
                if self[j] == tri[i] {
                    num_matching_vertices += 1;
                }
            }
        }

        //      A                                       A
        //     / \     +                     =         / \
        //    B - C       - E - B - C - D -     - E - B   C - D -
        if disolvable_edges.len() == 1 && num_matching_vertices == 2 {
            let insertion_index = disolvable_edges[0].1 + 1;
            let insertion_value = tri[disolvable_edges[0].0 + 2];
            self.insert(insertion_index, insertion_value);
            return Ok(());
        }

        //      A       - D - A             - D - A
        //     / \   +       /           =         \
        //    B - C         B - C - E -             C - E -
        if disolvable_edges.len() == 2 && num_matching_vertices == 3 {
            if disolvable_edges[1].1 == (disolvable_edges[0].1 + 1) % self.len() {
                self.remove(disolvable_edges[1].1);
                return Ok(());
            }
            if disolvable_edges[0].1 == (disolvable_edges[1].1 + 1) % self.len() {
                self.remove(disolvable_edges[0].1);
                return Ok(());
            }
        }

        // Could not compute a valid simple cycle from the existing boundary and the new triangle
        Err(())
    }
}

#[cfg(test)]
mod test {
    use super::SimpleCycle;

    #[test]
    fn test_extend() {
        let mut tris = vec![
            (2, 4, 1),
            (1, 5, 2),
            (5, 1, 3),
            (5, 3, 6),
            (3, 4, 6),
            (4, 3, 1),
        ];

        let mut boundary = SimpleCycle::new(tris[0].0, tris[0].1, tris[0].2);

        for i in 1..tris.len() {
            let mut idx = i;
            loop {
                assert!(idx < tris.len());
                match boundary.try_extend(tris[idx].0, tris[idx].1, tris[idx].2) {
                    Ok(()) => {
                        if idx > i {
                            tris.swap(i, idx);
                        }
                        break;
                    }
                    Err(()) => idx += 1,
                }
            }
            assert!(boundary.is_valid())
        }
        assert_eq!(boundary, SimpleCycle::from_vec(vec![2, 4, 6, 5]));
    }
}
