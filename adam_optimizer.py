import numpy as np

class AdamOptimizer:
    """
    Basic implementation of the Adam optimization algorithm.
    Reference: https://arxiv.org/abs/1412.6980
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters using Adam optimizer.

        params: numpy array of parameters
        grads: numpy array of gradients (same shape as params)
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params
