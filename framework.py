import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, Tuple, Optional
import time

from agents import RLAgent, RandomAgent
from game import TicTacToe3DEnv
from config import WIN_REWARD


class Trainer:
    """Фреймворк для обучения RL агентов."""

    def __init__(
        self, env: TicTacToe3DEnv, agent: RLAgent, opponent: Optional[RLAgent] = None
    ):
        """
        Инициализация.

        Args:
            env: Environment
            agent: Agent to train
            opponent: Opponent agent (if None, random opponent will be used)
        """
        self.env = env
        self.agent = agent
        self.opponent = opponent if opponent is not None else RandomAgent(env, "Random")
        self.metrics_history = {}

    def train_vs_random(
        self,
        episodes: int = 10000,
        eval_every: int = 1000,
        eval_episodes: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        Обучение против случайных ходов.

        Args:
            episodes: Number of episodes to train
            eval_every: Evaluate every n episodes
            eval_episodes: Number of episodes for evaluation
            verbose: Whether to print progress

        Returns:
            dict: Training metrics
        """
        # Reset metrics
        self.agent.reset_metrics()

        wins = 0
        draws = 0
        losses = 0
        total_steps = 0

        start_time = time.time()
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            # Randomly decide who goes first
            player_idx = random.choice([0, 1])
            players = [self.agent, self.opponent]
            player_symbols = [1, -1]

            while not done:
                # Current player
                current_player = players[player_idx]
                player_symbol = player_symbols[player_idx]

                # Choose action
                action = current_player.act(state, player_symbol, training=True)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Update agent if it's the one who moved
                if current_player == self.agent:
                    loss = self.agent.update(
                        state, action, reward, next_state, done, player_symbol
                    )
                    episode_reward += reward
                else:
                    # If opponent won, it's a loss for the agent
                    if reward > 1:
                        episode_reward = max(
                            0, episode_reward - self.env.game.steps * 5
                        )

                # Switch player for next turn
                player_idx = (player_idx + 1) % 2

                # Update state
                state = next_state
                steps += 1

                # Check for early termination (invalid move)
                if "error" in info:
                    break

            # Record episode results
            if "error" in info:
                losses += 1
            elif info.get("current_player", 0) == player_symbols[0]:
                wins += 1  # Agent won
            elif info.get("current_player", 0) == player_symbols[1]:
                losses += 1  # Opponent won
            else:
                draws += 1  # Draw

            # Update metrics
            total_steps += steps
            self.agent.metrics["episode_rewards"].append(episode_reward)

            # Evaluate periodically
            if (episode + 1) % eval_every == 0:
                win_rate, draw_rate, loss_rate, avg_steps = self._evaluate(
                    eval_episodes
                )
                self.agent.metrics["win_rate"].append(win_rate)
                self.agent.metrics["draw_rate"].append(draw_rate)
                self.agent.metrics["loss_rate"].append(loss_rate)
                self.agent.metrics["avg_steps"].append(avg_steps)

                if verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"Episode {episode+1}/{episodes} | "
                        + f"Win rate: {win_rate:.2f} | "
                        + f"Draw rate: {draw_rate:.2f} | "
                        + f"Loss rate: {loss_rate:.2f} | "
                        + f"Avg steps: {avg_steps:.2f} | "
                        + f"Epsilon: {self.agent.epsilon:.3f} | "
                        + f"Time: {elapsed:.1f}s"
                    )

        # Final metrics
        training_time = time.time() - start_time

        # Save metrics history
        self.metrics_history = {
            "agent_name": self.agent.name,
            "episodes": episodes,
            "training_time": training_time,
            "final_win_rate": (
                self.agent.metrics["win_rate"][-1]
                if self.agent.metrics["win_rate"]
                else 0
            ),
            "final_epsilon": self.agent.epsilon,
            "metrics": self.agent.metrics,
        }

        if verbose:
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Final win rate: {self.metrics_history['final_win_rate']:.2f}")
            print(f"Final epsilon: {self.agent.epsilon:.3f}")

        return self.metrics_history

    def train_self_play(
        self,
        episodes: int = 10000,
        eval_every: int = 1000,
        eval_episodes: int = 100,
        verbose: bool = True,
        random_proba: float = 0.0,
    ) -> Dict:
        """
        Train agent against itself (self-play).

        Args:
            episodes: Number of episodes to train
            eval_every: Evaluate every n episodes
            eval_episodes: Number of episodes for evaluation
            verbose: Whether to print progress

        Returns:
            dict: Training metrics
        """
        random_agent = RandomAgent(self.env, name="Shuffler")
        # Reset metrics
        self.agent.reset_metrics()

        start_time = time.time()
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            trajectory = []  # (state, action, reward, player)

            # Track current player (1 or -1)
            current_player = 1

            while not done:
                # Choose action (same agent plays both sides)
                if random.random() < random_proba:
                    action = random_agent.act(state, current_player, training=True)
                else:
                    action = self.agent.act(state, current_player, training=True)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Store experience from current player's perspective
                trajectory.append((state, action, reward, current_player))

                # Switch player for next turn
                state = next_state
                current_player *= -1
                state *= -1

                # Check for early termination (invalid move)
                if "error" in info:
                    break

            # Process trajectory to update the agent
            # Start from the end to correctly back-propagate
            loss_sum = 0
            for t in reversed(range(len(trajectory))):
                state, action, reward, player = trajectory[t]

                # Get the next state (if not terminal)
                next_state = trajectory[t + 1][0] if t < len(trajectory) - 1 else None

                # Adjust reward from player's perspective
                adjusted_reward = reward * player

                # Update agent
                if next_state is not None:
                    loss = self.agent.update(
                        state, action, adjusted_reward, next_state, False, player
                    )
                else:
                    loss = self.agent.update(
                        state, action, adjusted_reward, state, True, player
                    )

                loss_sum += loss

            # Calculate average loss for the episode
            avg_loss = loss_sum / len(trajectory) if trajectory else 0
            self.agent.metrics["episode_rewards"].append(avg_loss)

            # Evaluate periodically
            if (episode + 1) % eval_every == 0:
                win_rate, draw_rate, loss_rate, avg_steps = self._evaluate(
                    eval_episodes
                )
                self.agent.metrics["win_rate"].append(win_rate)
                self.agent.metrics["draw_rate"].append(draw_rate)
                self.agent.metrics["loss_rate"].append(loss_rate)
                self.agent.metrics["avg_steps"].append(avg_steps)

                if verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"Episode {episode+1}/{episodes} | "
                        + f"Win rate: {win_rate:.2f} | "
                        + f"Draw rate: {draw_rate:.2f} | "
                        + f"Loss rate: {loss_rate:.2f} | "
                        + f"Avg steps: {avg_steps:.2f} | "
                        + f"Epsilon: {self.agent.epsilon:.3f} | "
                        + f"Time: {elapsed:.1f}s"
                    )

        # Final metrics
        training_time = time.time() - start_time

        # Save metrics history
        self.metrics_history = {
            "agent_name": self.agent.name,
            "episodes": episodes,
            "training_time": training_time,
            "final_win_rate": (
                self.agent.metrics["win_rate"][-1]
                if self.agent.metrics["win_rate"]
                else 0
            ),
            "final_epsilon": self.agent.epsilon,
            "metrics": self.agent.metrics,
        }

        if verbose:
            print(f"\nTraining completed in {training_time:.1f} seconds")
            print(f"Final win rate: {self.metrics_history['final_win_rate']:.2f}")
            print(f"Final epsilon: {self.agent.epsilon:.3f}")

        return self.metrics_history

    def _evaluate(self, episodes: int = 100) -> Tuple[float, float, float, float]:
        """
        Evaluate agent performance.

        Args:
            episodes: Number of episodes for evaluation

        Returns:
            tuple: (win_rate, draw_rate, loss_rate, avg_steps)
        """
        wins = 0
        draws = 0
        losses = 0
        total_steps = 0

        # Save current epsilon and set to evaluation mode
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration during evaluation

        for _ in range(episodes):
            state = self.env.reset()
            done = False
            steps = 0

            # Randomly decide who goes first
            player_idx = random.choice([0, 1])
            players = [self.agent, self.opponent]
            player_symbols = [1, -1]

            while not done:
                # Current player
                current_player = players[player_idx]
                player_symbol = player_symbols[player_idx]

                # Choose action (no training)
                action = current_player.act(state, player_symbol, training=False)

                # Take step in environment
                state, reward, done, info = self.env.step(action)

                # Switch player for next turn
                player_idx = (player_idx + 1) % 2

                steps += 1

                # Check for early termination (invalid move)
                if "error" in info:
                    break

            # Record episode results
            if "error" in info:
                losses += 1
            elif info.get("current_player", 0) == player_symbols[0]:
                wins += 1  # Agent won
            elif info.get("current_player", 0) == player_symbols[1]:
                losses += 1  # Opponent won
            else:
                draws += 1  # Draw

            total_steps += steps

        # Restore epsilon
        self.agent.epsilon = original_epsilon

        win_rate = wins / episodes
        draw_rate = draws / episodes
        loss_rate = losses / episodes
        avg_steps = total_steps / episodes

        return win_rate, draw_rate, loss_rate, avg_steps

    def plot_metrics(self) -> None:
        """Plot training metrics."""
        if not self.metrics_history:
            print("No metrics to plot. Train the agent first.")
            return

        metrics = self.metrics_history["metrics"]
        episodes = self.metrics_history["episodes"]

        plt.figure(figsize=(15, 10))

        # Plot win/draw/loss rates
        plt.subplot(2, 2, 1)
        eval_points = range(0, episodes, episodes // len(metrics["win_rate"]))
        plt.plot(eval_points, metrics["win_rate"], label="Win Rate")
        plt.plot(eval_points, metrics["draw_rate"], label="Draw Rate")
        plt.plot(eval_points, metrics["loss_rate"], label="Loss Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Rate")
        plt.title("Win/Draw/Loss Rates")
        plt.legend()
        plt.grid(True)

        # Plot average steps per game
        plt.subplot(2, 2, 2)
        plt.plot(eval_points, metrics["avg_steps"])
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.title("Average Game Steps")
        plt.grid(True)

        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        if metrics["epsilon"]:
            epsilon_x = np.linspace(0, episodes, len(metrics["epsilon"]))
            plt.plot(epsilon_x, metrics["epsilon"])
            plt.xlabel("Episodes")
            plt.ylabel("Epsilon")
            plt.title("Exploration Rate (Epsilon)")
            plt.grid(True)

        # Plot loss
        plt.subplot(2, 2, 4)
        if metrics["loss"]:
            # Smooth loss curve
            loss_x = np.linspace(0, episodes, len(metrics["loss"]))
            window_size = (
                min(100, len(metrics["loss"]) // 10) if len(metrics["loss"]) > 0 else 1
            )
            loss_smoothed = np.convolve(
                metrics["loss"], np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(loss_x[: len(loss_smoothed)], loss_smoothed)
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)

        plt.tight_layout()
        plt.suptitle(
            f"Training Metrics for {self.metrics_history['agent_name']}",
            y=1.02,
            fontsize=16,
        )
        plt.show()


class Evaluator:
    """Evaluates and compares RL agents."""

    def __init__(self, env: TicTacToe3DEnv):
        """
        Initialize evaluator.

        Args:
            env: Environment
        """
        self.env = env

    def agent_vs_agent(
        self,
        agent1: RLAgent,
        agent2: RLAgent,
        episodes: int = 100,
        render: bool = False,
        render_delay: float = 0.5,
    ) -> Dict:
        """
        Play agent1 vs agent2.

        Args:
            agent1: First agent
            agent2: Second agent
            episodes: Number of episodes
            render: Whether to render games
            render_delay: Delay between moves when rendering

        Returns:
            dict: Results
        """
        random_agent = RandomAgent(self.env, name="InitStep")
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        cum_reward = 0
        total_steps = 0

        for episode in range(episodes):
            if render:
                print(f"\nGame {episode+1}/{episodes}: {agent1.name} vs {agent2.name}")

            state = self.env.reset()
            done = False
            steps = 0

            # Alternate who goes first
            if episode % 2 == 0:
                players = [agent1, agent2]
                player_symbols = [1, -1]
                player_names = [agent1.name, agent2.name]
            else:
                players = [agent2, agent1]
                player_symbols = [1, -1]
                player_names = [agent2.name, agent1.name]

            player_idx = 0
            init_step = True
            while not done:
                # Current player
                current_player = players[player_idx]
                player_symbol = player_symbols[player_idx]
                player_name = player_names[player_idx]

                # Choose action
                if init_step:
                    action = random_agent.act(state, player_symbol, training=False)
                else:
                    action = current_player.act(state, player_symbol, training=False)

                x, y = divmod(action, self.env.n)

                if render:
                    print(f"{player_name} ({player_symbol:+d}) plays: ({x}, {y})")

                # Take step in environment
                state, reward, done, info = self.env.step(action)

                if render:
                    self.env.render()
                    time.sleep(render_delay)

                # Switch player for next turn
                player_idx = (player_idx + 1) % 2

                steps += 1

                # Check for early termination
                if "error" in info:
                    # Current player made invalid move, other player wins
                    reward = -1 * WIN_REWARD
                    done = True
                    if render:
                        print(f"Invalid move by {player_name}!")

            # Record game result
            total_steps += steps

            if "error" in info:
                # Invalid move, penalize current player
                if players[player_idx - 1] == agent1:
                    agent2_wins += 1
                    if render:
                        print(f"{agent2.name} wins (invalid move by {agent1.name})!")
                else:
                    agent1_wins += 1
                    if render:
                        print(f"{agent1.name} wins (invalid move by {agent2.name})!")
            elif reward == WIN_REWARD:
                # Last player who moved won
                last_player = players[(player_idx - 1) % 2]
                if last_player == agent1:
                    agent1_wins += 1
                    if render:
                        print(f"{agent1.name} wins!")
                else:
                    agent2_wins += 1
                    if render:
                        print(f"{agent2.name} wins!")
            else:
                # Draw
                draws += 1
                if render:
                    print("Draw!")

            cum_reward += reward

        # Calculate stats
        agent1_win_rate = agent1_wins / episodes
        agent2_win_rate = agent2_wins / episodes
        draw_rate = draws / episodes
        avg_steps = total_steps / episodes
        avg_reward = cum_reward / episodes

        results = {
            "agent1_name": agent1.name,
            "agent2_name": agent2.name,
            "episodes": episodes,
            "agent1_wins": agent1_wins,
            "agent2_wins": agent2_wins,
            "draws": draws,
            "agent1_win_rate": agent1_win_rate,
            "agent2_win_rate": agent2_win_rate,
            "draw_rate": draw_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
        }

        # Print results
        print(f"\nResults after {episodes} games:")
        print(f"{agent1.name}: {agent1_wins} wins ({agent1_win_rate:.2%})")
        print(f"{agent2.name}: {agent2_wins} wins ({agent2_win_rate:.2%})")
        print(f"Draws: {draws} ({draw_rate:.2%})")
        print(f"Average steps per game: {avg_steps:.2f}")
        print(f"Average reward: {avg_reward:.2f}")

        return results

    def agent_vs_self(
        self,
        agent: RLAgent,
        episodes: int = 100,
        render: bool = False,
        render_delay: float = 0.5,
    ) -> Dict:
        """
        Play agent against itself.

        Args:
            agent: Agent to evaluate
            episodes: Number of episodes
            render: Whether to render games
            render_delay: Delay between moves when rendering

        Returns:
            dict: Results
        """
        return self.agent_vs_agent(agent, agent, episodes, render, render_delay)

    def human_vs_agent(
        self, agent: RLAgent, human_first: bool = True, render_delay: float = 0.5
    ) -> None:
        """
        Play human vs agent.

        Args:
            agent: Agent to play against
            human_first: Whether human plays first
            render_delay: Delay between agent moves
        """
        state = self.env.reset()
        done = False

        print(f"\n====== Human vs {agent.name} ======")
        print("Board coordinates: (x, y) where x and y are 0-2")
        print(
            "Human: X (1), Agent: O (-1)"
            if human_first
            else "Agent: X (1), Human: O (-1)"
        )
        print()

        self.env.render()

        # Set player order
        if human_first:
            human_symbol = 1
            agent_symbol = -1
            current_player = "human"
        else:
            human_symbol = -1
            agent_symbol = 1
            current_player = "agent"

        while not done:
            if current_player == "human":
                # Human's turn
                while True:
                    try:
                        move = input("\nYour move (x y): ")
                        x, y = map(int, move.strip().split())

                        if not (0 <= x < self.env.n and 0 <= y < self.env.n):
                            print(f"Coordinates must be between 0 and {self.env.n-1}")
                            continue

                        action = x * self.env.n + y

                        # Check if valid move
                        if action in self.env.get_valid_actions():
                            break
                        else:
                            print("Invalid move. Try again.")
                    except ValueError:
                        print("Invalid input. Please enter coordinates as 'x y'")
            else:
                # Agent's turn
                print("\nAgent is thinking...")
                time.sleep(render_delay)

                action = agent.act(state, agent_symbol, training=False)
                x, y = divmod(action, self.env.n)
                print(f"Agent plays: ({x}, {y})")

            # Execute move
            state, reward, done, info = self.env.step(action)

            # Render updated board
            print()
            self.env.render()

            # Check for game end
            if done:
                if "error" in info:
                    print(
                        f"\nInvalid move by {'human' if current_player == 'human' else 'agent'}!"
                    )
                    print(f"{'Agent' if current_player == 'human' else 'You'} win!")
                elif reward == WIN_REWARD:
                    print(f"\n{'You' if current_player == 'human' else 'Agent'} win!")
                else:
                    print("\nDraw!")
                break

            # Switch player
            current_player = "agent" if current_player == "human" else "human"
