import random
import numpy as np

class RayTracingEnv:
    def __init__(self, grid_size=10, objects=None, light_sources=None):
        self.grid_size = grid_size
        self.objects = set(objects or [])         # set of (x,y) obstacle coords
        self.light_sources = set(light_sources or [])  # set of (x,y) light coords
        self.state_size = grid_size * grid_size
        self.action_list = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
        # convenience?
        self._all_coords = [(x, y) for x in range(grid_size) for y in range(grid_size)]

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_obstacle(self, pos):
        return pos in self.objects

    def is_light(self, pos):
        return pos in self.light_sources

    def coords_to_index(self, pos):
        x, y = pos
        return x * self.grid_size + y

    def index_to_coords(self, idx):
        x = idx // self.grid_size
        y = idx % self.grid_size
        return (x, y)

    def get_possible_actions(self, pos):
        # return indices of actions that lead in-bounds (but we will allow stays for out-of-bounds)
        possible = []
        for i, (dx, dy) in enumerate(self.action_list):
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.in_bounds((nx, ny)):
                possible.append(i)
        return possible

    def random_free_position(self):
        candidates = [p for p in self._all_coords if p not in self.objects and p not in self.light_sources]
        return random.choice(candidates) if candidates else (0, 0)

class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state_idx, possible_action_indices):
        # epsilon-greedy over allowed actions
        if np.random.rand() < self.epsilon:
            return random.choice(possible_action_indices)
        else:
            # pick best among allowed (break ties randomly)
            qvals = self.q_table[state_idx, possible_action_indices]
            max_q = np.max(qvals)
            best_candidates = [a for a, q in zip(possible_action_indices, qvals) if q == max_q]
            return random.choice(best_candidates)

    def update_q_table(self, state_idx, action_idx, reward, next_state_idx, next_possible_actions):
        current_q = self.q_table[state_idx, action_idx]
        if next_possible_actions:
            max_next_q = np.max(self.q_table[next_state_idx, next_possible_actions])
        else:
            max_next_q = 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action_idx] = new_q

    def decay_exploration_rate(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class QLearning:
    def __init__(self, env, agent, num_episodes=2000, max_steps_per_episode=200):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode

    def calculate_reward(self, pos):
        # Terminal: reaching light
        if self.env.is_light(pos):
            return 100.0, True
        if self.env.is_obstacle(pos):
            return -100.0, True
        # Otherwise shaped reward: -1 per step and negative Euclidean distance (encourage moving toward light)
        # Use the closest light source distance (if multiple)
        if self.env.light_sources:
            dists = [np.linalg.norm(np.array(pos) - np.array(ls)) for ls in self.env.light_sources]
            distance = min(dists)
        else:
            distance = 0.0
        return -1.0 - 0.1 * distance, False

    def train(self):
        for ep in range(1, self.num_episodes + 1):
            start = self.env.random_free_position()
            state_idx = self.env.coords_to_index(start)
            total_reward = 0.0

            for step in range(self.max_steps):
                pos = self.env.index_to_coords(state_idx)
                possible_actions = self.env.get_possible_actions(pos)
                if not possible_actions:
                    # stuck (rare) — end episode
                    break
                action_idx = self.agent.choose_action(state_idx, possible_actions)
                dx, dy = self.env.action_list[action_idx]
                next_pos = (pos[0] + dx, pos[1] + dy)

                # safety: if next_pos is out-of-bounds, treat as staying in place with small penalty
                if not self.env.in_bounds(next_pos):
                    next_pos = pos

                next_state_idx = self.env.coords_to_index(next_pos)
                reward, done = self.calculate_reward(next_pos)

                next_possible_actions = self.env.get_possible_actions(next_pos)
                self.agent.update_q_table(state_idx, action_idx, reward, next_state_idx, next_possible_actions)

                total_reward += reward
                state_idx = next_state_idx

                if done:
                    break

            self.agent.decay_exploration_rate()

            if ep % 100 == 0 or ep == 1:
                print(f"Episode {ep:4d} / {self.num_episodes:4d}  epsilon={self.agent.epsilon:.4f}  total_reward={total_reward:.2f}")

    def run_greedy_episode(self, start=None, max_steps=200):
        if start is None:
            start = self.env.random_free_position()
        state_idx = self.env.coords_to_index(start)
        path = [start]
        for _ in range(max_steps):
            pos = self.env.index_to_coords(state_idx)
            possible = self.env.get_possible_actions(pos)
            if not possible:
                break
            action_idx = self.agent.choose_action(state_idx, possible)  # epsilon might be low; for pure greedy set epsilon=0
            dx, dy = self.env.action_list[action_idx]
            next_pos = (pos[0] + dx, pos[1] + dy)
            if not self.env.in_bounds(next_pos):
                break
            path.append(next_pos)
            if self.env.is_light(next_pos) or self.env.is_obstacle(next_pos):
                break
            state_idx = self.env.coords_to_index(next_pos)
        return path

if __name__ == "__main__":
    grid = 10
    obstacles = [(3,3),(3,4),(3,5),(4,5),(5,5),(6,5)]
    lights = [(9,9)]
    env = RayTracingEnv(grid_size=grid, objects=obstacles, light_sources=lights)

    agent = Agent(state_size=env.state_size, action_size=len(env.action_list),
                  learning_rate=0.2, discount_factor=0.95,
                  exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01)

    qlearn = QLearning(env, agent, num_episodes=1200, max_steps_per_episode=200)
    qlearn.train()

    # inspect a greedy trajectory after training
    start = (0,0)
    agent.epsilon = 0.0  # force greedy
    path = qlearn.run_greedy_episode(start=start, max_steps=200)
    print("Greedy path from", start, ":", path)
