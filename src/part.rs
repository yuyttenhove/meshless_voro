use glam::DVec3;

pub struct Part {
    x: DVec3,
    cid: usize,
}

impl Part {
    pub fn new(x: DVec3, cid: usize) -> Self { Self { x, cid } }

    pub fn cid(&self) -> usize {
        self.cid
    }

    pub fn x(&self) -> DVec3 {
        self.x
    }

    pub fn distance_squared(&self, other: &Self) -> f64 {
        self.x.distance_squared(other.x)
    }
}