use std::f64::consts::PI;

// Units:
// - Distance: AU
// - Time: years
// - Mass: solar masses
// In these units Newton's gravitational constant becomes:
//   G = 4 π²
// which makes Kepler's third law simple: a body at 1 AU around 1 M☉ has period 1 year.
//
// Newton's law (force):
//   F = G * m_i * m_j / r^2
// Acceleration on body i due to body j (vector form):
//   a_i = G * m_j * (r_j - r_i) / |r_j - r_i|^3
// This implementation computes the sum over j of the RHS above.
pub const G: f64 = 4.0 * PI * PI;

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn mul(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub mass: f64, // solar masses
    pub pos: Vec3, // AU
    pub vel: Vec3, // AU / year
}

impl Body {
    pub fn new(name: &str, mass: f64, pos: Vec3, vel: Vec3) -> Self {
        Self {
            name: name.to_string(),
            mass,
            pos,
            vel,
        }
    }
}

// O(N^2) Newtonian gravity
// Computes accelerations for all bodies using pairwise Newtonian gravity.
// Implementation detail / formula used:
// For each pair (i,j) define dr = r_j - r_i and r = |dr|.
// inv_dist3 = 1 / r^3
// factor = G * inv_dist3
// a_i += factor * m_j * dr
// a_j -= factor * m_i * dr
// This yields the vector acceleration a_i = G * sum_j m_j * (r_j - r_i) / |r_j - r_i|^3
pub fn compute_accelerations(bodies: &[Body]) -> Vec<Vec3> {
    let n = bodies.len();
    let mut acc = vec![Vec3::zero(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dr = bodies[j].pos.sub(bodies[i].pos);
            let dist_sq = dr.length_sq();
            let dist = dist_sq.sqrt();
            if dist == 0.0 {
                continue;
            }

            let inv_dist3 = 1.0 / (dist_sq * dist);
            let factor = G * inv_dist3;

            let a_i = dr.mul(factor * bodies[j].mass);
            let a_j = dr.mul(factor * bodies[i].mass);

            acc[i] = acc[i].add(a_i);
            acc[j] = acc[j].sub(a_j);
        }
    }

    acc
}

// Semi-implicit (a.k.a. symplectic) Euler integrator
// Update formulas used:
//   v_{n+1} = v_n + a_n * dt
//   x_{n+1} = x_n + v_{n+1} * dt
// Note: position update uses the updated velocity (semi-implicit), which is
// more stable for energy conservation in many Hamiltonian systems than
// explicit Euler.
pub fn step(bodies: &mut [Body], dt: f64) {
    let acc = compute_accelerations(bodies);

    for (b, a) in bodies.iter_mut().zip(acc.iter()) {
        b.vel = b.vel.add(a.mul(dt));
        b.pos = b.pos.add(b.vel.mul(dt));
    }
}
