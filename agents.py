import numpy as np
import random
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from game import TicTacToe3DEnv

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLAgent:
    """Базовый класс для всех RL агентов."""

    def __init__(self, env: TicTacToe3DEnv, name: str = "Agent"):
        """
        Инициализация агента.

        Args:
            env: Environment
            name: Agent name
        """
        self.env = env
        self.name = name
        self.metrics = {
            "episode_rewards": [],
            "win_rate": [],
            "draw_rate": [],
            "loss_rate": [],
            "avg_steps": [],
            "epsilon": [],
            "loss": [],
            "max_q": [],
            "action_dist": [0] * env.action_space.n,
        }

    def act(self, state: np.ndarray, player: int = 1, training: bool = False) -> int:
        """
        Выбор действия.

        Args:
            state: Current state
            player: Current player (1 or -1)
            training: Whether in training mode

        Returns:
            int: Chosen action
        """
        raise NotImplementedError

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        player: int = 1,
    ) -> float:
        """
        Обновление знаний агента.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            player: Current player (1 or -1)

        Returns:
            float: Loss value
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Сохранение модели.

        Args:
            path: Path to save to
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """
        Загрузка модели.

        Args:
            path: Path to load from
        """
        raise NotImplementedError

    def get_valid_actions(self, state: np.ndarray) -> List[int]:
        """
        Получить верное действие по состоянию среды.

        Args:
            state: Current state

        Returns:
            list: Valid actions
        """
        valid = []
        for i in range(self.env.n):
            for j in range(self.env.n):
                if state[i, j, self.env.n - 1] == 0:
                    valid.append(i * self.env.n + j)
        return valid

    def reset_metrics(self) -> None:
        """Сброс метрик."""
        self.metrics = {
            "episode_rewards": [],
            "win_rate": [],
            "draw_rate": [],
            "loss_rate": [],
            "avg_steps": [],
            "epsilon": [],
            "loss": [],
            "max_q": [],
            "action_dist": [0] * self.env.action_space.n,
        }


class RandomAgent(RLAgent):
    """Агент, который будет всегда выдавать случайные ходы."""

    def act(self, state: np.ndarray, player: int = 1, training: bool = False) -> int:
        """Случайно выбирает один из доступных ходов."""
        valid_actions = self.get_valid_actions(state)
        action = random.choice(valid_actions)

        # Update metrics
        self.metrics["action_dist"][action] += 1

        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        player: int = 1,
    ) -> float:
        """Заглушка, так как этот агент не учится."""
        return 0.0

    def save(self, path: str) -> None:
        """Заглушка, так как нечего сохранять"""
        pass

    def load(self, path: str) -> None:
        """Заглушка, так как нечего загружать"""
        pass


class QLearningAgent(RLAgent):
    """Q-learning agent with epsilon-greedy policy."""

    def __init__(
        self,
        env: TicTacToe3DEnv,
        name: str = "Q-Learning",
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.1,
    ):
        """
        Initialize Q-learning agent.

        Args:
            env: Environment
            name: Agent name
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        super().__init__(env, name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table as defaultdict
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_state_key(self, state: np.ndarray, player: int = 1) -> tuple:
        """
        Convert state to hashable key.

        Args:
            state: Current state
            player: Current player (1 or -1)

        Returns:
            tuple: State key
        """
        # Symmetrize state for player perspective
        return tuple((state * player).flatten())

    def act(self, state: np.ndarray, player: int = 1, training: bool = False) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            player: Current player (1 or -1)
            training: Whether in training mode

        Returns:
            int: Chosen action
        """
        valid_actions = self.get_valid_actions(state)

        # Explore: random action
        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            # Exploit: best action from Q-table
            state_key = self.get_state_key(state, player)
            q_values = self.Q[state_key]

            # Filter for valid actions only
            q_valid = [(a, q_values[a]) for a in valid_actions]
            action, q_value = max(q_valid, key=lambda x: x[1])

            # Update max_q metric
            if training:
                self.metrics["max_q"].append(q_value)

        # Update action distribution
        self.metrics["action_dist"][action] += 1

        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        player: int = 1,
    ) -> float:
        """
        Update Q-values based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            player: Current player (1 or -1)

        Returns:
            float: TD error (loss)
        """
        state_key = self.get_state_key(state, player)
        next_state_key = self.get_state_key(next_state, player)

        # Get current Q-value
        current_q = self.Q[state_key][action]

        if done:
            # Terminal state: target is just the reward
            target_q = reward
        else:
            # Non-terminal: target is reward + discounted future value
            next_q_values = self.Q[next_state_key]
            valid_actions = self.get_valid_actions(next_state)
            next_max_q = (
                max([next_q_values[a] for a in valid_actions]) if valid_actions else 0
            )
            target_q = reward + self.gamma * next_max_q

        # TD update rule
        td_error = target_q - current_q
        self.Q[state_key][action] += self.alpha * td_error

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.metrics["epsilon"].append(self.epsilon)

        # Return TD error as loss
        loss = td_error**2
        self.metrics["loss"].append(loss)

        return loss

    def save(self, path: str) -> None:
        """
        Save Q-table to file.

        Args:
            path: Path to save to
        """
        # Convert defaultdict to dict for saving
        q_dict = {str(k): v.tolist() for k, v in self.Q.items()}
        np.save(path, q_dict)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load Q-table from file.

        Args:
            path: Path to load from
        """
        q_dict = np.load(path, allow_pickle=True).item()
        # Convert back to defaultdict
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for k, v in q_dict.items():
            self.Q[eval(k)] = np.array(v)
        print(f"Model loaded from {path}")


class DQN(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Initialize DQN.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 100_000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch from buffer.

        Args:
            batch_size: Batch size

        Returns:
            tuple: Batch of experiences
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Convert to torch tensors
        state_tensor = torch.FloatTensor(np.array(state)).to(device)
        action_tensor = torch.LongTensor(action).unsqueeze(1).to(device)
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state_tensor = torch.FloatTensor(np.array(next_state)).to(device)
        done_tensor = torch.FloatTensor(done).unsqueeze(1).to(device)

        return (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            done_tensor,
        )

    def __len__(self) -> int:
        """Get buffer length."""
        return len(self.buffer)


class DQNAgent(RLAgent):
    """Deep Q-Network agent."""

    def __init__(
        self,
        env: TicTacToe3DEnv,
        name: str = "DQN",
        alpha: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.1,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        update_target_every: int = 500,
        hidden_dim: int = 128,
    ):
        """
        Initialize DQN agent.

        Args:
            env: Environment
            name: Agent name
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            buffer_size: Replay buffer size
            batch_size: Training batch size
            update_target_every: Update target network frequency
            hidden_dim: Hidden layer dimension
        """
        super().__init__(env, name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0

        # Calculate dimensions
        self.input_dim = (
            env.observation_space.shape[0]
            * env.observation_space.shape[1]
            * env.observation_space.shape[2]
        )
        self.output_dim = env.action_space.n

        # Neural networks
        self.policy_net = DQN(self.input_dim, self.output_dim, hidden_dim).to(device)
        self.target_net = DQN(self.input_dim, self.output_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def symmetrize_state(self, state: np.ndarray, player: int = 1) -> np.ndarray:
        """
        Adjust state for player perspective.

        Args:
            state: Current state
            player: Current player (1 or -1)

        Returns:
            numpy.ndarray: Adjusted state
        """
        return state * player

    def act(self, state: np.ndarray, player: int = 1, training: bool = False) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            player: Current player (1 or -1)
            training: Whether in training mode

        Returns:
            int: Chosen action
        """
        valid_actions = self.get_valid_actions(state)

        # Explore: random action
        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            # Exploit: best action from policy network
            sym_state = self.symmetrize_state(state, player)
            state_tensor = torch.FloatTensor(sym_state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()

            # Set invalid actions to -infinity
            masked_q_values = q_values.copy()
            valid_mask = np.ones(self.output_dim, dtype=bool)
            valid_mask[valid_actions] = False
            masked_q_values[valid_mask] = -np.inf

            action = np.argmax(masked_q_values)

            # Update max_q metric
            if training:
                self.metrics["max_q"].append(float(q_values.max()))

        # Update action distribution
        self.metrics["action_dist"][action] += 1

        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        player: int = 1,
    ) -> float:
        """
        Update neural network based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            player: Current player (1 or -1)

        Returns:
            float: Loss value
        """
        # Symmetrize states for player perspective
        sym_state = self.symmetrize_state(state, player)
        sym_next_state = self.symmetrize_state(next_state, player)

        # Add to replay buffer
        self.buffer.push(sym_state, action, reward, sym_next_state, done)

        loss = 0.0

        # Only train if enough samples
        if len(self.buffer) >= self.batch_size:
            # Sample batch from replay buffer
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
                self.buffer.sample(self.batch_size)
            )

            # Compute current Q values
            current_q = self.policy_net(state_batch).gather(1, action_batch)

            # Compute target Q values
            with torch.no_grad():
                next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]
                target_q = reward_batch + (1 - done_batch) * self.gamma * next_q

            # Compute loss
            loss = self.loss_fn(current_q, target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            self.metrics["loss"].append(float(loss.item()))

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.metrics["epsilon"].append(self.epsilon)

        return float(loss)

    def save(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save to
        """
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"Model loaded from {path}")
