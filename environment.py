import numpy as np
import pygame
import pymunk
import gymnasium as gym
from gymnasium import spaces
from math import cos

class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, reward_type="shaped"):
        super(DoublePendulumEnv, self).__init__()

        self.reward_type = reward_type
        self.dt = 1 / 60.0

        # Observation: [cart_x, cart_v, theta1, omega1, theta2, omega2]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action: force applied to cart
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)

        # Rendering
        self.screen = None
        self.clock = None
        self.width = 800
        self.height = 600

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.space = pymunk.Space()
        self.space.gravity = (0, -981)

        # Ground
        static_body = self.space.static_body

        # Cart
        self.cart_body = pymunk.Body(1.0, pymunk.moment_for_box(1.0, (50, 30)))
        self.cart_body.position = (400, 300)
        cart_shape = pymunk.Poly.create_box(self.cart_body, (50, 30))
        self.space.add(self.cart_body, cart_shape)

        # Groove joint (track)
        groove = pymunk.GrooveJoint(
            static_body,
            self.cart_body,
            (100, 300),
            (700, 300),
            (0, 0),
        )
        self.space.add(groove)

        # Pole 1
        self.pole1_body = pymunk.Body(0.5, pymunk.moment_for_segment(0.5, (0, 0), (0, 100), 5))
        self.pole1_body.position = (400, 350)
        pole1_shape = pymunk.Segment(self.pole1_body, (0, 0), (0, 100), 5)
        self.space.add(self.pole1_body, pole1_shape)

        joint1 = pymunk.PinJoint(self.cart_body, self.pole1_body, (0, 15), (0, 0))
        self.space.add(joint1)

        # Pole 2
        self.pole2_body = pymunk.Body(0.5, pymunk.moment_for_segment(0.5, (0, 0), (0, 100), 5))
        self.pole2_body.position = (400, 450)
        pole2_shape = pymunk.Segment(self.pole2_body, (0, 0), (0, 100), 5)
        self.space.add(self.pole2_body, pole2_shape)

        joint2 = pymunk.PinJoint(self.pole1_body, self.pole2_body, (0, 100), (0, 0))
        self.space.add(joint2)

        return self._get_obs(), {}

    def step(self, action):
        force = float(action[0]) * 1000
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))

        self.space.step(self.dt)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._is_done()

        return obs, reward, done, False, {}

    def _get_obs(self):
        cart_x = self.cart_body.position.x
        cart_v = self.cart_body.velocity.x

        theta1 = self.pole1_body.angle
        omega1 = self.pole1_body.angular_velocity

        theta2 = self.pole2_body.angle
        omega2 = self.pole2_body.angular_velocity

        return np.array(
            [cart_x, cart_v, theta1, omega1, theta2, omega2],
            dtype=np.float32,
        )

    def _compute_reward(self, action):
        _, _, theta1, omega1, theta2, omega2 = self._get_obs()
        cart_x = self.cart_body.position.x

        # Baseline reward
        base = cos(theta1) + cos(theta2)

        if self.reward_type == "baseline":
            return base

        # Shaped reward
        reward = base
        reward -= 0.1 * abs(cart_x - 400)
        reward -= 0.01 * (abs(omega1) + abs(omega2))
        reward -= 0.001 * (action[0] ** 2)

        return reward

    def _is_done(self):
        x = self.cart_body.position.x
        theta1 = self.pole1_body.angle
        theta2 = self.pole2_body.angle

        if abs(x - 400) > 300:
            return True

        if abs(theta1) > np.pi / 2 or abs(theta2) > np.pi / 2:
            return True

        return False

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        def draw_body(body, shape):
            if isinstance(shape, pymunk.Poly):
                points = [body.local_to_world(v) for v in shape.get_vertices()]
                points = [(int(p.x), self.height - int(p.y)) for p in points]
                pygame.draw.polygon(self.screen, (0, 0, 0), points)
            elif isinstance(shape, pymunk.Segment):
                a = body.local_to_world(shape.a)
                b = body.local_to_world(shape.b)
                pygame.draw.line(
                    self.screen,
                    (0, 0, 255),
                    (int(a.x), self.height - int(a.y)),
                    (int(b.x), self.height - int(b.y)),
                    5,
                )

        for shape in self.space.shapes:
            draw_body(shape.body, shape)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None