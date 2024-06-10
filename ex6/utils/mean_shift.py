import torch


class MeanShift(torch.nn.Module):
    """Mean Shift clustering algorithm."""

    def __init__(self, x, chan, n_seeds, bandwidth=None, eps=1e-2, max_steps=1000):
        """Constructor.

        Args:
            x:
                The data to cluster.
            chan:
                The number of channels.
            n_seeds:
                The number of seeds to use.
            bandwidth:
                The bandwidth of the kernel.
            eps:
                The stopping criterion. If the mean shift update is smaller than eps, the algorithm stops.
            max_steps:
                The maximum number of steps to perform.

        """
        X_flat = torch.reshape(x, [chan, -1])  # chan x pixels
        X_flat = X_flat.transpose_(1, 0)  # pixels x chan
        X_flat = X_flat[X_flat.sum(1) != 0]

        # check for empty tensor:
        if X_flat.shape[0] == 0:
            raise ValueError('Empty tensor provided. Change seed, try again!')

        perm = torch.randperm(X_flat.shape[0])
        idx = perm[:n_seeds]
        self.C = X_flat[idx, :]  # n_seeds x chan
        self.X1 = torch.unsqueeze(X_flat.transpose(1, 0), 0)  # 1 x chan x pixels
        self.X2 = torch.unsqueeze(X_flat, 0)  # 1  x pixels x chan
        self.diff = 10
        self.eps = eps
        self.bandwidth = bandwidth
        self.max_steps = max_steps

    def _mean_shift_step(self, C):
        # data: emb_dim x pixels
        C = torch.unsqueeze(C, 2)  # num_centroids x emb_dim x 1
        dist = torch.sum((C - self.X1).square_() / self.bandwidth, dim=1)  # n_seeds x num_pixels
        dist = (-dist).exp_()
        num = torch.sum(torch.unsqueeze(dist, 2) * self.X2, dim=1)  # n_seeds x chan
        denom = torch.sum(dist, dim=1, keepdims=True)  # n_seeds x 1
        C = num / denom
        return C

    def forward(self):
        """Forward pass."""
        c = self.C
        step = 0
        while self.diff > self.eps:
            C_old = c
            c = self._mean_shift_step(c)
            self.diff = torch.max(torch.sum(torch.abs(C_old - c), dim=1))
            step += 1
            if step > self.max_steps:
                print('No convergence after {}'.format(self.max_steps), 'steps')
                break
        return c
