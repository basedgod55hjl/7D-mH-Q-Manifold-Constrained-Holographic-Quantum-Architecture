// 7D Crystal LLM Builder - Optimizer Module
// Implements advanced optimizers: Sophia, Lion, Shampoo

pub struct SophiaG {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub rho: f32,
    pub epsilon: f32,
}

impl SophiaG {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.95,
            rho: 0.04,
            epsilon: 1e-8,
        }
    }

    pub fn update(&self, param: &mut f32, grad: f32, hessian: f32, m: &mut f32, h: &mut f32) {
        *m = self.beta1 * (*m) + (1.0 - self.beta1) * grad;
        *h = self.beta2 * (*h) + (1.0 - self.beta2) * hessian;

        let ratio = (*m).abs() / ((*h) + self.epsilon);
        let clip = ratio.min(self.rho);
        let delta = clip * (*m).signum();

        *param -= self.lr * delta;
    }
}

pub struct Lion {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
}

impl Lion {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
        }
    }

    pub fn update(&self, param: &mut f32, grad: f32, m: &mut f32) {
        let update = (self.beta1 * (*m) + (1.0 - self.beta1) * grad).signum();
        *param -= self.lr * update;
        *m = self.beta2 * (*m) + (1.0 - self.beta2) * grad;
    }
}
