#[derive(Clone, Debug)]
pub struct SimpleCycle {
    ptrs: Vec<usize>,
    start: usize,
    pub len: usize,
}

impl SimpleCycle {
    pub fn new(capacity: usize) -> Self {
        Self {
            ptrs: (0..capacity).collect(),
            start: 0,
            len: 0,
        }
    }

    pub fn grow(&mut self) {
        self.ptrs.push(self.ptrs.len());
    }

    pub fn init(&mut self, a: usize, b: usize, c: usize) {
        let mut current = self.start;
        let mut next;
        // reset the state of the cycle
        for _ in 0..self.len {
            next = self.ptrs[current];
            self.ptrs[current] = current;
            current = next;
        }
        // init the cycle from a triangle
        self.len = 3;
        self.start = a;
        self.ptrs[a] = b;
        self.ptrs[b] = c;
        self.ptrs[c] = a;
    }

    fn contains(&self, idx: usize) -> bool {
        self.ptrs[idx] != idx
    }

    pub fn try_extend(&mut self, a: usize, b: usize, c: usize) -> Result<(), ()> {
        let tri = [a, b, c];
        let contained = [self.contains(a), self.contains(b), self.contains(c)];
        for i in 0..3 {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;

            //      A                                       A
            //     / \     +                     =         / \
            //    B - C       - E - B - C - D -     - E - B   C - D -
            if !contained[i] && contained[j] && contained[k] && self.ptrs[tri[k]] == tri[j] {
                self.ptrs[tri[k]] = tri[i];
                self.ptrs[tri[i]] = tri[j];
                self.len += 1;
                return Ok(());
            }
            //      A       - D - A             - D - A
            //     / \   +       /           =         \
            //    B - C         B - C - E -             C - E -
            else if contained[i]
                && contained[j]
                && contained[k]
                && self.ptrs[tri[k]] == tri[j]
                && self.ptrs[tri[j]] == tri[i]
            {
                self.ptrs[tri[k]] = tri[i];
                self.ptrs[tri[j]] = tri[j];
                if self.start == tri[j] {
                    self.start = tri[i];
                }
                self.len -= 1;
                return Ok(());
            }
        }

        Err(())
    }

    pub fn iter(&self) -> SimpleCycle2Iterator {
        SimpleCycle2Iterator {
            simple_cycle: self,
            next: self.start,
        }
    }
}

pub struct SimpleCycle2Iterator<'a> {
    simple_cycle: &'a SimpleCycle,
    next: usize,
}

impl<'a> Iterator for SimpleCycle2Iterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let res = Some(self.next);
        self.next = self.simple_cycle.ptrs[self.next];
        res
    }
}

#[cfg(test)]
mod test {
    use crate::simple_cycle::SimpleCycle;

    #[test]
    fn test_extend() {
        let mut tris = [(2, 4, 1), (1, 5, 2), (5, 1, 3), (5, 3, 6), (3, 4, 6), (4, 3, 1)];

        let mut boundary = SimpleCycle::new(7);
        boundary.init(tris[0].0, tris[0].1, tris[0].2);

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
        }
        assert_eq!(boundary.len, 4);
        let res = boundary.iter().take(boundary.len).collect::<Vec<_>>();
        assert_eq!(res, vec![2, 4, 6, 5])
    }
}
