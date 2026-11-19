import unittest

import numpy as np

from main import (
    CECController,
    car_next_state,
    lissajous,
    obstacle_clearance,
    theta_init,
    time_step,
    x_init,
    y_init,
)


class CECControllerTests(unittest.TestCase):
    def test_prediction_cost_and_clearance_including_terminal_state(self):
        controller = CECController()
        # A noisy measurement can be inside the buffer, but outside the obstacle.
        state = np.array([1.55, 2.0, 0.0])
        controller(state, 0)
        states = controller.opti.value(controller.x)
        controls = controller.opti.value(controller.u)
        np.testing.assert_allclose(states[:, 0], state, atol=1e-7)
        for t in range(controller.horizon):
            expected = car_next_state(time_step, states[:, t], controls[:, t], noise=False)
            np.testing.assert_allclose(states[:, t + 1], expected, atol=1e-7)
        self.assertTrue(np.all(np.abs(states[:2, 1:]) <= 3 + 1e-7))
        for center in ([-2, -2], [1, 2]):
            distances = np.linalg.norm(states[:2, 1:] - np.array(center)[:, None], axis=0)
            self.assertGreaterEqual(distances.min(), 0.5 + obstacle_clearance - 1e-7)

        reference = np.array([lissajous(t) for t in range(controller.horizon + 1)]).T
        error = states - reference
        expected_cost = (
            np.sum(error[:2] ** 2)
            + np.sum(1 - np.cos(error[2]))
            + np.sum(controls ** 2)
        )
        self.assertAlmostEqual(controller.opti.value(controller.opti.f), expected_cost, places=7)

    def test_equivalent_headings_produce_equivalent_controls(self):
        state = np.array(lissajous(25))
        control = CECController()(state, 25)
        state[2] += 2 * np.pi
        wrapped_control = CECController()(state, 25)
        np.testing.assert_allclose(control, wrapped_control, atol=1e-6)

    def test_full_period_and_reference_wrap_with_and_without_noise(self):
        for noise in (False, True):
            with self.subTest(noise=noise):
                np.random.seed(10)
                controller = CECController()
                state = np.array([x_init, y_init, theta_init])
                errors = []
                for iteration in range(240):
                    errors.append(np.linalg.norm(state[:2] - np.array(lissajous(iteration))[:2]))
                    control = controller(state, iteration)
                    self.assertTrue(np.all(control >= np.array([0, -1]) - 1e-7))
                    self.assertTrue(np.all(control <= np.array([1, 1]) + 1e-7))
                    state = car_next_state(time_step, state, control, noise=noise)
                    self.assertTrue(np.all(np.abs(state[:2]) <= 3))
                    for center in ([-2, -2], [1, 2]):
                        self.assertGreater(np.linalg.norm(state[:2] - center), 0.5)
                self.assertLess(np.mean(errors), 0.23 if not noise else 0.25)


if __name__ == '__main__':
    unittest.main()
