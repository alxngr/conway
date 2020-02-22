import pytest
import numpy as np

from game import ConwayGame, Cell


@pytest.fixture
def my_game():
    return ConwayGame(initial_state=ConwayGame.get_random_state(size=(20, 20)))


def test_array_to_sparse_state(my_game):
    cell_count = np.sum(my_game.initial_state)

    assert cell_count == len(my_game.current_state)


def test_states_collection(my_game):
    steps = 3

    for i in range(steps):
        my_game.next()

    assert len(my_game.states) == steps + 1


def test_get_neighbors(my_game):
    max_i = my_game.shape[0] - 1
    max_j = my_game.shape[1] - 1
    mid_i = max_i // 2
    mid_j = max_j // 2

    # corners
    assert len(my_game.get_neighbors(cell=Cell(0, 0))) == 3
    assert len(my_game.get_neighbors(cell=Cell(0, max_j))) == 3
    assert len(my_game.get_neighbors(cell=Cell(max_i, max_j))) == 3
    assert len(my_game.get_neighbors(cell=Cell(max_i, 0))) == 3

    # borders
    assert len(my_game.get_neighbors(cell=Cell(0, mid_j))) == 5
    assert len(my_game.get_neighbors(cell=Cell(max_i, mid_j))) == 5
    assert len(my_game.get_neighbors(cell=Cell(mid_i, max_j))) == 5
    assert len(my_game.get_neighbors(cell=Cell(mid_i, 0))) == 5

    # middle
    assert len(my_game.get_neighbors(cell=Cell(mid_i, mid_j))) == 8


def test_next_state(my_game):
    # Given
    force_next_state = set()

    for row in range(my_game.shape[0]):
        for col in range(my_game.shape[1]):
            # Neighbors on
            neighbors = my_game.get_neighbors(cell=Cell(row, col))
            neighbors_on = len([x for x in neighbors if x in my_game.current_state])

            # Cell is on
            if Cell(row, col) in my_game.current_state:
                if neighbors_on == 2 or neighbors_on == 3:
                    force_next_state.add(Cell(row, col))

            # Cell is off
            else:
                if neighbors_on == 3:
                    force_next_state.add(Cell(row, col))

    # When
    my_game.next()

    # Then
    assert force_next_state == my_game.current_state
