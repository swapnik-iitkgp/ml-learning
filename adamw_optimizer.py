"""
AdamW Optimizer implementation from scratch.

Reference:
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.

This implementation is meant for educational purposes.
"""

import numpy as np


class AdamW:
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer.

        Parameters:
        - lr (float): Learning rate
        - betas (tuple): Coefficients for computing running averages of gradient and its square
        - eps (float): Term added to denominator for numerical stability
        - weight_decay (float): Weight decay coefficient
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using AdamW optimization rule.

        Parameters:
        - params (dict): Dictionary of parameters {name: value}
        - grads (dict): Dictionary of gradients {name: grad}
        """
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Decoupled weight decay
            params[key] -= self.lr * self.weight_decay * params[key]

            # Parameter update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return params


# Example usage
if __name__ == "__main__":
    params = {"w": np.array([1.0, 2.0]), "b": np.array([0.5])}
    grads = {"w": np.array([0.1, 0.1]), "b": np.array([0.01])}

    optimizer = AdamW(lr=0.01)
    for step in range(5):
        params = optimizer.update(params, grads)
        print(f"Step {step + 1}: {params}")
