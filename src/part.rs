use glam::DVec3;

pub struct Part {
    x: DVec3,
    cid: usize,
    id: usize,
}

impl Part {
    pub fn new(x: DVec3, cid: usize, pid: usize) -> Self {
        Self {
            x,
            cid,
            id: pid,
        }
    }

    pub fn cid(&self) -> usize {
        self.cid
    }

    pub fn x(&self) -> DVec3 {
        self.x
    }

    pub fn distance_squared(&self, other: &Self) -> f64 {
        self.x.distance_squared(other.x)
    }

    pub fn id(&self) -> usize {
        self.id
    }
}
