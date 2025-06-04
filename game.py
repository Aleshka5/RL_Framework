import gym
import numpy as np
from numpy import ndarray
from gym import spaces
import matplotlib.pyplot as plt
from typing import Optional
from config import WIN_REWARD


class ValidStateWrapper(spaces.Box):
    """
    Обёртка для генератора состояний исходных наблюдений.
    В игре есть особенность, что фишки не могут висеть в воздухе.
    Они всегда лежат одна на другой.
    """

    def __init__(self, original_space: spaces.Box):
        assert isinstance(original_space, spaces.Box)
        self.low = original_space.low
        self.high = original_space.high
        self._shape = original_space.shape
        self.dtype = original_space.dtype
        self.space = original_space

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    # @property
    # def shape(self) -> tuple[int, ...]:
    #     """Has stricter type than gym.Space - never None."""
    #     return self.shape

    def valid_states(self):
        """Генерирует все валидные состояния для куба NxNxN"""
        shape = self.space.shape
        dtype = self.space.dtype
        low = int(self.space.low.min())
        high = int(self.space.high.max())

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
                count_player1 = state[np.where(state == 1)].shape[0]
                count_player2 = state[np.where(state == -1)].shape[0]
                if (
                    state[x, y, 0] == 0
                    and state[x, y, 1] != 0
                    or abs(count_player1 - count_player2) > 1
                ):
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
        self.board = np.zeros((n, n, n), dtype=int)  # 0 - empty, 1 - player 1, -1 - player 2
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

    def reward(self, board, step_x: int, step_y: int, player: int) -> int:
        def opponent_can_win(bd, opponent):
            for x in range(self.n):
                for y in range(self.n):
                    if bd[x, y, self.n - 1] == 0:
                        z = next((z for z in range(self.n) if bd[x, y, z] == 0), None)
                        if z is not None:
                            bd_copy = bd.copy()
                            bd_copy[x, y, z] = opponent
                            if self.check_winner(board=bd_copy):
                                return True
            return False

        def count_traps(bd, player):
            traps = []
            lines = []

            # All possible directions for lines
            n = self.n
            directions = [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),  # Axes
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),  # 2D diagonals
                (1, 1, 1),
                (1, -1, 1),
                (1, 1, -1),
                (1, -1, -1),  # 3D diagonals
            ]

            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        for dx, dy, dz in directions:
                            line = []
                            for i in range(n):
                                xi, yi, zi = x + dx * i, y + dy * i, z + dz * i
                                if 0 <= xi < n and 0 <= yi < n and 0 <= zi < n:
                                    line.append((xi, yi, zi))
                            if len(line) == n:
                                values = [bd[xi, yi, zi] for xi, yi, zi in line]
                                if values.count(player) == n - 1 and values.count(0) == 1:
                                    empty_idx = values.index(0)
                                    ex, ey, ez = line[empty_idx]
                                    if ez != 0 and bd[ex, ey, ez - 1] == 0:
                                        # # Check if opponent can place there next turn
                                        # if opponent_can_win(bd.copy(), -player):
                                        #     continue
                                        traps.append(line[empty_idx])
            # print(traps)
            # print(set(traps))
            return len(traps)

        # Copy of the board before move
        pre_board = board.copy()

        # Place move on a new board for analysis
        post_board = board.copy()
        for z in range(self.n):
            if post_board[step_x, step_y, z] == 0:
                post_board[step_x, step_y, z] = player
                break

        opponent = -player

        reward_value = 0

        # Проверка, что после этого хода у тебя появится ситуация со 100% победой
        sure_win = True
        twice_post_board = post_board.copy()
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    if twice_post_board[x, y, z] == 0:
                        twice_post_board[x, y, z] = -player
                        if not opponent_can_win(twice_post_board, player):
                            sure_win = False
                        twice_post_board[x, y, z] = 0
                        break
        # print(f"Стопроцентная победа: {sure_win}")
        if sure_win:
            reward_value += 100
        # Блокировка
        opponent_win_before = opponent_can_win(pre_board, opponent)
        # print(f"Opponent can win before: {opponent_win_before}")
        opponent_win_after = opponent_can_win(post_board, opponent)
        # print(f"Opponent can win after: {opponent_win_after}")

        if opponent_win_before and not opponent_win_after:
            reward_value += 45
        elif opponent_win_before and opponent_win_after:
            reward_value -= 45
        elif not opponent_win_before and opponent_win_after:
            reward_value -= 60

        # Ловушки
        traps_before = count_traps(pre_board, player)
        # print(f"Count traps before: {traps_before}")
        traps_after = count_traps(post_board, player)
        # print(f"Count traps after: {traps_after}")

        old_traps = min(traps_before, traps_after)
        new_traps = traps_after - traps_before

        reward_value += 30 * new_traps

        return reward_value

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

        reward = self.reward(self.board, x, y, player=self.current_player)
        # Find the lowest empty z-position in the column
        for z in range(self.n):
            if self.board[x, y, z] == 0:
                self.board[x, y, z] = self.current_player
                break

        self.steps += 1
        reward -= self.steps

        info = {"current_player": self.current_player, "steps": self.steps}

        if self.check_winner():
            self.winner = self.current_player
            self.done = True
            reward += WIN_REWARD  # Win for current player
        elif np.all(self.board != 0):  # Draw
            self.done = True
            reward = 0
        else:
            self.current_player *= -1  # Switch player

        next_state = self.board.copy()
        return next_state, reward, self.done, info

    def check_winner(self, board: Optional[ndarray] = None) -> bool:
        """
        Проверка есть ли победитель на поле.

        Returns:
            bool: True if there's a winner, False otherwise
        """
        n = self.n
        if board is None:
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
                print(" ".join(simbol_map.get(self.board[x, y, z], "-") for x in range(self.n)))
            print()

    def render_3d(self):
        """Показать текущее поле через matplotlib 3d рендер."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        x, y, z = np.where(self.board != 0)
        colors = ["red" if self.board[i, j, k] == 1 else "blue" for i, j, k in zip(x, y, z)]

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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n, n, n), dtype=np.int8)

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
