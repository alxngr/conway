from typing import Dict, Tuple, Union, Set, List
from collections import namedtuple
from os.path import join
import numpy as np
import pickle
from tqdm import tqdm
import config

Cell = namedtuple('Cell', 'i j')


class ConwayGame:

    def __init__(self, initial_state: np.array):
        self._shape: Tuple[int, int] = initial_state.shape
        self.initial_state: np.array = self._get_state_from_array(state=initial_state)
        self.current_state: Set[Cell] = self.initial_state
        self.states: Dict[int, Set[Cell]] = {0: self.current_state}
        self.is_stable: bool = False
        self.has_been_simulated: bool = False

    def __repr__(self) -> str:
        return self._get_string_representation(state=self.current_state)

    """Properties"""

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    """Static methods"""

    @staticmethod
    def _get_state_from_array(*, state: np.array) -> Set[Cell]:
        sparse_state = set()

        it = np.nditer(state, flags=['multi_index'])
        while not it.finished:
            if it[0] == 1:
                sparse_state.add(Cell(it.multi_index[0], it.multi_index[1]))
            it.iternext()

        return sparse_state

    @staticmethod
    def get_random_state(*, size: Tuple[int, int], probability_to_be_on=0.2) -> Union[int, np.ndarray]:
        return np.random.choice([0, 1], size[0] * size[1], p=[1 - probability_to_be_on, probability_to_be_on]) \
            .reshape(size[0], size[1])

    @staticmethod
    def load(*, filename: str) -> Set[Cell]:
        path = join(config.MODELS_DIR, filename)
        with open(path, 'rb') as f:
            initial_state = pickle.load(f)
        return initial_state

    """Methods"""

    def _get_next_state(self) -> Set[Cell]:
        """
        Return next state after applying Conway game rules.

        1. If a cell is ON, and has fewer than 2 neighbors ON then it turns OFF
        2. If a cell is ON, and has 2 or 3 neighbors ON then it remains ON
        3. If a cell is ON, and has more than 3 neighbors ON then it turns OFF
        4. If a cell is OFF, and has 3 neighbors ON then it turns ON
        """

        next_state = self.current_state.copy()
        counter = {}

        # Add cells with neighbors to counter
        for cell in next_state:
            if cell not in counter:
                counter[cell] = 0

            neighbors = self.get_neighbors(cell=cell)
            for neighbor in neighbors:
                if neighbor in counter:
                    counter[neighbor] += 1
                else:
                    counter[neighbor] = 1

        # Apply Conway game rules
        for cell in counter:
            if counter[cell] < 2 or counter[cell] > 3:
                next_state.discard(cell)
            if counter[cell] == 3:
                next_state.add(cell)

        return next_state

    def _get_string_representation(self, *, state: Set[Cell]) -> str:
        if self.shape[0] > 50:
            return f'Grid is too large to be displayed ({self.shape}). Use visualization class.'
        else:
            representation = ''
            for row in range(self.shape[0]):
                line = ''
                for col in range(self.shape[1]):
                    if Cell(row, col) in state:
                        line += '+'
                    else:
                        line += '.'
                representation += line
                representation += '\n'
            return representation

    """API"""

    def get_neighbors(self, *, cell: Cell) -> List[Cell]:
        """Return neighbors of cell."""

        neighbors = []
        i, j = cell
        max_i, max_j = self.shape[0] - 1, self.shape[1] - 1

        # West
        if j > 0:
            neighbors.append(Cell(i, j - 1))

        # North West
        if i > 0 and j > 0:
            neighbors.append(Cell(i - 1, j - 1))

        # North
        if i > 0:
            neighbors.append(Cell(i - 1, j))

        # North East
        if i > 0 and j < max_j:
            neighbors.append(Cell(i - 1, j + 1))

        # East
        if j < max_j:
            neighbors.append(Cell(i, j + 1))

        # South East
        if i < max_i and j < max_j:
            neighbors.append(Cell(i + 1, j + 1))

        # South
        if i < max_i:
            neighbors.append(Cell(i + 1, j))

        # South West
        if i < max_i and j > 0:
            neighbors.append(Cell(i + 1, j - 1))

        return neighbors

    def next(self) -> None:
        next_state = self._get_next_state()

        self.is_stable = self.current_state == next_state
        self.current_state = next_state
        self.states[len(self.states)] = self.current_state

    def simulate(self, *, max_iterations: int = config.MAX_ITERATIONS, show_progress: bool = True) -> None:
        """Simulate game until all cells are off or the maximum iterations is reached or the state is stable."""

        it = 0

        if show_progress:
            pbar = tqdm(total=max_iterations)
            while len(self.current_state) > 0 and it < max_iterations and not self.is_stable:
                self.next()
                it += 1
                pbar.update(1)
            pbar.close()
        else:
            while len(self.current_state) > 0 and it < max_iterations and not self.is_stable:
                self.next()
                it += 1

        self.has_been_simulated = True

    def reset(self) -> None:
        self.current_state = self.initial_state
        self.states = {0: self.current_state}
        self.is_stable = False
        self.has_been_simulated = False

    def print(self, *, state: Set[Cell]) -> None:
        print(self._get_string_representation(state=state))

    def save(self, *, model_name: str) -> None:
        path = join(config.MODELS_DIR, model_name)
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self.initial_state, f)
