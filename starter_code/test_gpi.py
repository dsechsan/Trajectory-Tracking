import itertools
import unittest

import numpy as np

from GPI_main import (DT, PERIOD, SIGMA, OBSTACLES, SAFETY_MARGIN, GPI,
                      _neighbors, _sweep, lissajous, rollout)


class GPITests(unittest.TestCase):
    def test_noise_stencil_interpolates_and_wraps(self):
        grid = np.array([-1, -0.05, 0, 0.05, 1])
        indices, weights = _neighbors(grid, np.array([0.01]), 0.04)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertGreater(np.count_nonzero(weights), 2)
        self.assertAlmostEqual(np.sum(grid[indices] * weights), 0.01)

        heading = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        indices, weights = _neighbors(heading, np.array([np.pi - 0.01]), 0.1, periodic=True)
        unwrapped = np.where(heading[indices] < 0, heading[indices] + 2 * np.pi,
                             heading[indices])
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertIn(0, indices[weights > 0])
        self.assertAlmostEqual(np.sum(unwrapped * weights), np.pi - 0.01)

        indices, weights = _neighbors(grid, np.array([-5.0, 5.0]), 0.04)
        np.testing.assert_array_equal(indices[:, 0], [0, len(grid) - 1])
        np.testing.assert_allclose(weights[:, 0], 1)

    def test_backup_against_explicit_quadrature_at_phase_wrap(self):
        m = GPI(nx=7, ny=7, nth=12, nv=3, nw=5)
        t, i, j, h = PERIOD - 1, 3, 2, 0
        values = np.random.default_rng(42).normal(size=m.state_cost.shape)
        policy = np.zeros(values.shape, dtype=np.int32)
        e = np.array([m.x[i], m.y[j], m.heading[h]])
        theta = e[2] + m.ref[t, 2]
        nodes = (-np.sqrt(3), 0, np.sqrt(3))
        node_weights = (1/6, 2/3, 1/6)
        returns = []
        for vi, wi in itertools.product(range(len(m.v)), range(len(m.w))):
            v, w = m.v[vi], m.w[wi]
            mean = e + DT * np.array([v*np.cos(theta), v*np.sin(theta), w]) + m.ref[t] - m.ref[0]
            expected = 0.0
            for samples in itertools.product(range(3), repeat=3):
                point = mean + SIGMA * np.array([nodes[s] for s in samples])
                point[:2] = np.clip(point[:2], -3, 3)
                point[2] = (point[2] + np.pi) % (2*np.pi) - np.pi
                axes = (m.x, m.y, np.r_[m.heading, np.pi])
                choices = []
                for axis, coord in zip(axes, point):
                    lower = np.clip(np.searchsorted(axis, coord, side='right') - 1, 0, len(axis) - 2)
                    upper_weight = (coord - axis[lower]) / (axis[lower+1] - axis[lower])
                    choices.append(((lower, 1-upper_weight), (lower+1, upper_weight)))
                sample_value = 0.0
                for x, y, z in itertools.product(*choices):
                    sample_value += x[1] * y[1] * z[1] * values[0, x[0], y[0], z[0] % len(m.heading)]
                expected += np.prod([node_weights[s] for s in samples]) * sample_value
            returns.append(m.state_cost[t, i, j, h] + m.effort[vi, wi] + m.gamma*expected)
        _sweep(values, policy, m.state_cost, m.effort, m.xt, m.yt, m.ht, m.gamma, False, True)
        self.assertAlmostEqual(values[t, i, j, h], min(returns))
        self.assertEqual(policy[t, i, j, h], np.argmin(returns))

    def test_default_policy_tracks_and_avoids_obstacles_for_seeded_rollout(self):
        model = GPI()
        wx = model.ref[:, 0, None, None] + model.x[None, :, None]
        wy = model.ref[:, 1, None, None] + model.y[None, None, :]
        cx, cy, radius = OBSTACLES[0]
        distance = np.hypot(wx - cx, wy - cy)
        t, i, j = np.argwhere((distance > radius) & (distance < radius + SAFETY_MARGIN))[0]
        tracking_cost = model.x[i]**2 + model.y[j]**2 + (1 - np.cos(model.heading[0]))**2
        self.assertAlmostEqual(model.state_cost[t, i, j, 0] - tracking_cost, 1000)
        model.solve()
        states = rollout(model, seed=10)
        reference = lissajous(np.arange(len(states)))
        self.assertLess(np.linalg.norm(states[:, :2] - reference[:, :2], axis=1).mean(), 0.3)
        start = states[:-1, :2]
        segments = states[1:, :2] - start
        length_squared = np.sum(segments**2, axis=1)
        for cx, cy, radius in OBSTACLES:
            offset = np.array([cx, cy]) - start
            fraction = np.divide(np.sum(offset * segments, axis=1), length_squared,
                                 out=np.zeros_like(length_squared), where=length_squared > 0)
            closest = start + np.clip(fraction, 0, 1)[:, None] * segments
            self.assertTrue(np.all(np.linalg.norm(closest - [cx, cy], axis=1) > radius))

    def test_methods_agree_on_discounted_value_and_periodic_control(self):
        model = GPI(nx=5, ny=5, nth=6, nv=3, nw=3)
        model.solve('value', tol=1e-7)
        value = model.value.copy()
        model.solve('policy', tol=1e-7)
        np.testing.assert_allclose(value, model.value, atol=1e-6, rtol=0)
        state = model.ref[0] + np.array([0.1, -0.2, 0.3])
        np.testing.assert_array_equal(model.control(0, state),
                                      model.control(PERIOD, state + [0, 0, 2*np.pi]))
        # Same tracking error, different world positions: only one is unsafe.
        np.testing.assert_allclose(model.state_cost[0, -1, 2] - model.state_cost[50, -1, 2], 1000)


if __name__ == '__main__':
    unittest.main()
