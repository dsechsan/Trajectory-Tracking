"""Periodic stochastic tracking MDP (ECE276B project 3, part 2)."""

import argparse
from time import perf_counter

import numpy as np
from numba import njit


PERIOD = 100
DT = 0.5
SIGMA = np.array([0.04, 0.04, 0.004])


def lissajous(k):
    a = 2 * np.pi / 50
    phase = a * DT * (np.asarray(k) % PERIOD)
    return np.stack((2 * np.cos(phase), 2 * np.sin(3 * phase),
                     np.arctan2(6 * a * np.cos(3 * phase), -2 * a * np.sin(phase))), axis=-1)


def _position_grid(n):
    # Include zero and concentrate resolution where accurate tracking matters most.
    left = np.linspace(-1, 0, n // 2 + 1)[:-1]
    points = np.r_[left, np.linspace(0, 1, n - len(left))]
    return 3 * points * np.abs(points)


def _neighbors(grid, mean, sigma, periodic=False):
    """Lower index and upper probability on a two-point Gaussian stencil."""
    if periodic:
        mean = (mean + np.pi) % (2 * np.pi) - np.pi
        axis = np.r_[grid, np.pi]  # upper seam neighbor is grid[0]
    else:
        mean = np.clip(mean, grid[0], grid[-1])
        axis = grid
    lo = np.clip(np.searchsorted(axis, mean, side='right') - 1, 0, len(axis) - 2)
    a, b = axis[lo], axis[lo + 1]
    # Logistic of the log likelihood ratio avoids underflow for small sigma.
    log_ratio = (b - a) * (mean - (a + b) / 2) / sigma**2
    upper = np.exp(-np.logaddexp(0, -log_ratio))
    return lo.astype(np.int32), upper


@njit
def _sweep(value, policy, state_cost, effort, xt, yt, ht, gamma, evaluate, update):
    """Reverse-phase sweep: fixed-policy evaluation or greedy Bellman backup."""
    nt, nx, ny, nh = value.shape
    nv, nw = effort.shape
    xi, xp = xt
    yi, yp = yt
    hi, hp = ht
    delta = 0.0
    for t in range(nt - 1, -1, -1):
        nxt = value[(t + 1) % nt]
        for i in range(nx):
            for j in range(ny):
                for h in range(nh):
                    action = policy[t, i, j, h]
                    best = np.inf
                    for v in range(action // nw if evaluate else 0,
                                   action // nw + 1 if evaluate else nv):
                        x, px = xi[t, h, i, v], xp[t, h, i, v]
                        y, py = yi[t, h, j, v], yp[t, h, j, v]
                        for w in range(action % nw if evaluate else 0,
                                       action % nw + 1 if evaluate else nw):
                            z, pz = hi[t, h, w], hp[t, h, w]
                            zz = (z + 1) % nh
                            # The joint weight factors across independent noise axes.
                            ev = 0.0
                            for dx in range(2):
                                for dy in range(2):
                                    row = nxt[x + dx, y + dy]
                                    ev += ((px if dx else 1 - px) * (py if dy else 1 - py)
                                           * ((1 - pz) * row[z] + pz * row[zz]))
                            q = effort[v, w] + gamma * ev
                            if q < best:
                                best = q
                                best_action = v * nw + w
                    best += state_cost[t, i, j, h]
                    delta = max(delta, abs(best - value[t, i, j, h]))
                    if not evaluate:
                        policy[t, i, j, h] = best_action
                    if update:
                        value[t, i, j, h] = best
    return delta


class GPI:
    def __init__(self, nx=20, ny=20, nth=20, nv=10, nw=20, gamma=0.95):
        if min(nx, ny) < 3 or min(nth, nv, nw) < 2 or not 0 < gamma < 1:
            raise ValueError("need >=3 position bins, >=2 other bins, and 0 < gamma < 1")
        self.gamma = gamma
        self.x, self.y = _position_grid(nx), _position_grid(ny)
        self.heading = np.linspace(-np.pi, np.pi, nth, endpoint=False)
        self.v, self.w = np.linspace(0, 1, nv), np.linspace(-1, 1, nw)
        self.effort = self.v[:, None]**2 + self.w**2
        self.ref = lissajous(np.arange(PERIOD))
        drift = self.ref - np.roll(self.ref, -1, axis=0)
        theta = self.ref[:, 2, None] + self.heading

        def spatial(grid, direction, dim):
            shift = drift[:, dim, None, None] + DT * direction[:, :, None] * self.v
            return _neighbors(grid, grid[None, None, :, None] + shift[:, :, None, :], SIGMA[dim])

        self.xt = spatial(self.x, np.cos(theta), 0)
        self.yt = spatial(self.y, np.sin(theta), 1)
        self.ht = _neighbors(self.heading, self.heading[None, :, None]
                             + drift[:, 2, None, None] + DT * self.w, SIGMA[2], periodic=True)

        wx = self.x[None, :, None] + self.ref[:, 0, None, None]
        wy = self.y[None, None, :] + self.ref[:, 1, None, None]
        unsafe = (np.abs(wx) > 3) | (np.abs(wy) > 3)
        for cx, cy in [(-2, -2), (1, 2)]:
            unsafe |= (wx - cx)**2 + (wy - cy)**2 <= 0.5**2
        position_cost = self.x[:, None]**2 + self.y**2 + 1000 * unsafe
        self.state_cost = position_cost[..., None] + (1 - np.cos(self.heading))**2
        self.policy = None

    def solve(self, method='value', tol=1e-4, max_iter=100):
        if method not in ('value', 'policy') or not np.isfinite(tol) or tol <= 0 or max_iter < 1:
            raise ValueError("invalid solver method, tolerance, or iteration limit")
        value = np.zeros_like(self.state_cost)
        policy = np.zeros(value.shape, dtype=np.int32)
        args = (value, policy, self.state_cost, self.effort, self.xt, self.yt, self.ht, self.gamma)
        for iteration in range(1, max_iter + 1):
            if method == 'value':
                if _sweep(*args, False, True) > tol:
                    continue
            else:
                # Evaluate only the chosen action before improving over all actions.
                for _ in range(max_iter):
                    if _sweep(*args, True, True) <= tol:
                        break
                else:
                    raise RuntimeError("fixed-policy evaluation did not converge")
            residual = _sweep(*args, False, False)  # greedy policy for frozen V
            if residual <= tol:
                self.policy, self.value, self.iterations = policy, value, iteration
                return policy
        raise RuntimeError(f"{method} iteration did not converge in {max_iter} sweeps")

    def control(self, t, state):
        if self.policy is None:
            raise RuntimeError("solve or load a policy first")
        phase = t % PERIOD
        x, y, h = state - self.ref[phase]
        angle_distance = np.abs((h - self.heading + np.pi) % (2 * np.pi) - np.pi)
        action = self.policy[phase, np.argmin(abs(x - self.x)), np.argmin(abs(y - self.y)),
                             np.argmin(angle_distance)]
        v, w = divmod(int(action), len(self.w))
        return np.array([self.v[v], self.w[w]])


def rollout(model, steps=240, seed=10):
    rng = np.random.default_rng(seed)
    state = np.array([1.5, 0, np.pi / 2])
    states = np.empty((steps, 3))
    for t in range(steps):
        states[t] = state
        v, w = model.control(t, state)
        h = state[2]
        state += DT * np.array([v * np.cos(h), v * np.sin(h), w]) + rng.normal(0, SIGMA)
    return states


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['value', 'policy'], default='value')
    parser.add_argument('--output', default='policy.npy')
    args = parser.parse_args()
    start = perf_counter()
    model = GPI()
    np.save(args.output, model.solve(args.method))
    print(f'{args.method}: {model.iterations} iterations, training and save: {perf_counter()-start:.2f} s')
    start = perf_counter()
    states = rollout(model)
    error = np.linalg.norm(states[:, :2] - lissajous(np.arange(len(states)))[:, :2], axis=1)
    print(f'240-step rollout: {perf_counter()-start:.3f} s; mean position error: {error.mean():.3f}')
