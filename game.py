import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

from config import WIN_REWARD


class ValidStateWrapper:
    """
    Обёртка для генератора состояний исходных наблюдений.
    В игре есть особенность, что фишки не могут висеть в воздухе.
    Они всегда лежат одна на другой.
    """

    def __init__(self, original_space: spaces.Box):
        assert isinstance(original_space, spaces.Box)
        self.space = original_space
        self.shape = original_space.shape

    def valid_states(self):
        """Генерирует все валидные состояния для куба 2x2x2"""
        shape = self.space.shape
        dtype = self.space.dtype
        low, high = self.space.low.item(), self.space.high.item()

        # Перебираем все возможные состояния
        for flat in np.ndindex(*(high - low + 1,) * np.prod(shape)):
            state = np.array(flat, dtype=dtype).reshape(shape) + low

            if self.is_valid(state):
                yield state

    def is_valid(self, state: np.ndarray) -> bool:
        """Проверяет, что нет 'подвешенных' фишек"""
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                # Если нижняя клетка пуста, верхняя не может быть занята
                if state[x, y, 0] == 0 and state[x, y, 1] != 0:
                    return False
        return True


class TicTacToe3D:
    """Реализация игры N в ряд в трёхмерном случае."""

    def __init__(self, n=3):
        """
        Инициализация игры N в ряд в трёхмерном случае.

        Args:
            n: Size of the board (n x n x n)
        """
        self.n = n
        self.board = np.zeros(
            (n, n, n), dtype=int
        )  # 0 - empty, 1 - player 1, -1 - player 2
        self.current_player = 1  # Player 1 starts
        self.winner = None
        self.done = False
        self.steps = 0

    def reset(self):
        """Сброс состояния поля."""
        self.board.fill(0)
        self.current_player = 1
        self.winner = None
        self.done = False
        self.steps = 0
        return self.board.copy()

    def step(self, x, y):
        """
        Функция совершения хода по координатам.

        Args:
            x: X-coordinate (0 to n-1)
            y: Y-coordinate (0 to n-1)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Game is already finished.")
        if not (0 <= x < self.n and 0 <= y < self.n):
            raise ValueError(f"Coordinates out of bounds: ({x}, {y})")
        if self.board[x, y, self.n - 1] != 0:
            raise ValueError("This column is already full.")

        # Find the lowest empty z-position in the column
        for z in range(self.n):
            if self.board[x, y, z] == 0:
                self.board[x, y, z] = self.current_player
                break

        self.steps += 1
        reward = self.steps
        info = {"current_player": self.current_player, "steps": self.steps}

        if self.check_winner():
            self.winner = self.current_player
            self.done = True
            reward = WIN_REWARD  # Win for current player
        elif np.all(self.board != 0):  # Draw
            self.done = True
            reward = 0
        else:
            self.current_player *= -1  # Switch player

        next_state = self.board.copy()
        return next_state, reward, self.done, info

    def check_winner(self) -> bool:
        """
        Проверка есть ли победитель на поле.

        Returns:
            bool: True if there's a winner, False otherwise
        """
        n = self.n
        board = self.board

        # Check rows, columns and depths
        for i in range(n):
            for j in range(n):
                if abs(sum(board[i, j, :])) == n:  # Check rows (horizontal lines)
                    return True
                if abs(sum(board[i, :, j])) == n:  # Check columns (vertical lines)
                    return True
                if abs(sum(board[:, i, j])) == n:  # Check depths (Z-axis)
                    return True

        # Check diagonals on each level
        for i in range(n):
            if abs(sum(board[i, range(n), range(n)])) == n:
                return True
            if abs(sum(board[i, range(n), range(n - 1, -1, -1)])) == n:
                return True
            if abs(sum(board[range(n), i, range(n)])) == n:
                return True
            if abs(sum(board[range(n), i, range(n - 1, -1, -1)])) == n:
                return True
            if abs(sum(board[range(n), range(n), i])) == n:
                return True
            if abs(sum(board[range(n), range(n - 1, -1, -1), i])) == n:
                return True

        # Check 3D diagonals
        if abs(sum(board[range(n), range(n), range(n)])) == n:
            return True
        if abs(sum(board[range(n), range(n), range(n - 1, -1, -1)])) == n:
            return True
        if abs(sum(board[range(n), range(n - 1, -1, -1), range(n)])) == n:
            return True
        if abs(sum(board[range(n), range(n - 1, -1, -1), range(n - 1, -1, -1)])) == n:
            return True

        return False

    def render(self):
        """Показать текущее поле в ASCII формате."""
        simbol_map = {1: "X", -1: "O"}
        for z in range(self.n):
            print(f"Layer {z}:")
            for y in range(self.n):
                print(
                    " ".join(
                        simbol_map.get(self.board[x, y, z], "-") for x in range(self.n)
                    )
                )
            print()

    def render_3d(self):
        """Показать текущее поле через matplotlib 3d рендер."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        x, y, z = np.where(self.board != 0)
        colors = [
            "red" if self.board[i, j, k] == 1 else "blue" for i, j, k in zip(x, y, z)
        ]

        ax.scatter(x, y, z, c=colors, s=100, marker="o")

        # Set axes limits
        ax.set_xlim(0, self.n - 1)
        ax.set_ylim(0, self.n - 1)
        ax.set_zlim(0, self.n - 1)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Tic-Tac-Toe Board")

        # Add grid lines
        for i in range(self.n):
            ax.plot([0, self.n - 1], [i, i], [0, 0], "k-", alpha=0.2)
            ax.plot([i, i], [0, self.n - 1], [0, 0], "k-", alpha=0.2)

        plt.show()


class TicTacToe3DEnv(gym.Env):
    """Обёртка над игрой для совместимости с Gym."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, n=3):
        """
        Инициализация среды.

        Args:
            n: Size of the board (n x n x n)
        """
        super(TicTacToe3DEnv, self).__init__()
        self.game = TicTacToe3D(n)
        self.n = n

        # Action space: each (x, y) cell where you can place a token
        self.action_space = spaces.Discrete(n * n)

        # Observation space: n x n x n cells with values {0, 1, -1}
        self.observation_space = ValidStateWrapper(
            spaces.Box(low=-1, high=1, shape=(n, n, n), dtype=np.int8)
        )

    def reset(self):
        """Сброс среды."""
        return self.game.reset()

    def step(self, action):
        """
        Функция хода в среде.

        Args:
            action: Action to take (0 to n^2-1)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        x, y = divmod(action, self.n)
        try:
            next_state, reward, done, info = self.game.step(x, y)
        except ValueError:
            # Invalid move: penalize agent, end game
            next_state = self.game.board.copy()
            reward = -1 * WIN_REWARD  # Large negative reward for invalid move
            done = True
            info = {"error": "Invalid move", "steps": self.game.steps}
        return next_state, reward, done, info

    def render(self, mode="ASCII"):
        """Показ поля."""
        if mode == "3d":
            self.game.render_3d()
        else:
            self.game.render()

    def close(self):
        """Заглушка, так как нет промежуточных сохранённых данных."""
        pass

    def get_valid_actions(self):
        """
        Получение списка возможных ходов.

        Returns:
            list: Valid actions
        """
        valid = []
        for i in range(self.n):
            for j in range(self.n):
                if self.game.board[i, j, self.n - 1] == 0:
                    valid.append(i * self.n + j)
        return valid
