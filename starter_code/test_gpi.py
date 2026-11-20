import itertools
import unittest

import numpy as np

from GPI_main import DT, PERIOD, SIGMA, GPI, _sweep


class GPITests(unittest.TestCase):
    def test_backup_against_explicit_joint_gaussian_at_phase_wrap(self):
        m = GPI(nx=7, ny=7, nth=12, nv=3, nw=5)
        t, i, j, h = PERIOD - 1, 3, 2, 0
        values = np.random.default_rng(42).normal(size=m.state_cost.shape)
        policy = np.zeros(values.shape, dtype=np.int32)
        # An independently enumerated eight-corner distribution at phase 99 -> 0.
        e = np.array([m.x[i], m.y[j], m.heading[h]])
        theta = e[2] + m.ref[t, 2]
        returns = []
        for v, w in itertools.product(m.v, m.w):
            mean = e + DT * np.array([v*np.cos(theta), v*np.sin(theta), w]) + m.ref[t] - m.ref[0]
            mean[:2] = np.clip(mean[:2], -3, 3)
            mean[2] = (mean[2] + np.pi) % (2*np.pi) - np.pi
            axes = (m.x, m.y, np.r_[m.heading, np.pi])
            choices = []
            for axis, mu in zip(axes, mean):
                lower = np.clip(np.searchsorted(axis, mu, side='right') - 1, 0, len(axis) - 2)
                choices.append([lower, lower + 1])
            indices = list(itertools.product(*choices))
            points = np.array([[axis[idx] for axis, idx in zip(axes, cell)] for cell in indices])
            logp = -0.5 * np.sum(((points - mean) / SIGMA)**2, axis=1)
            probs = np.exp(logp - logp.max())
            expected = np.average([values[0, x, y, z % len(m.heading)] for x, y, z in indices],
                                  weights=probs)
            returns.append(m.state_cost[t, i, j, h] + v*v + w*w + m.gamma*expected)
        _sweep(values, policy, m.state_cost, m.effort, m.xt, m.yt, m.ht, m.gamma, False, True)
        self.assertAlmostEqual(values[t, i, j, h], min(returns))
        self.assertEqual(policy[t, i, j, h], np.argmin(returns))

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
