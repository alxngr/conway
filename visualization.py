import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from os.path import join, exists
from os import mkdir, listdir
from shutil import rmtree
from tqdm import tqdm

import config
from game import ConwayGame, Cell
from errors import GameHasNotBeenSimulated


class Visualization:

    @staticmethod
    def plot_number_of_cell_evolution(*, game: ConwayGame) -> None:
        """Plot the number of cells per epoch."""

        if not game.has_been_simulated:
            print('Game was not simulated. Simulating now...')
            game.simulate()

        y = [len(game.states[key]) for key in game.states]

        plt.plot(y)
        plt.title('Evolution of population')
        plt.xlabel('Epochs')
        plt.ylabel('Number of cells')

        plt.show()

    @staticmethod
    def plot_state(*, game: ConwayGame, epoch: int, save: bool = False, save_path: str = '') -> None:
        """Plot an epoch of the conway game."""

        fig, ax = plt.subplots(1)

        # Grid
        for row in range(game.shape[1]):
            plt.axvline(x=row, linewidth=0.1, color='b')
        for col in range(game.shape[0]):
            plt.axhline(y=col, linewidth=0.1, color='b')

        # Set ratio of figure
        plt.xlim(0, game.shape[1])
        plt.ylim(0, game.shape[0])
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot cells
        for cell in game.states[epoch]:
            Visualization.plot_square(cell=cell, max_y=game.shape[0], ax=ax)

        # Hide axis
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

        # Title
        plt.title(f'{epoch} epoch')

        if save:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_game(*, game: ConwayGame, model_name: str) -> None:
        if not game.has_been_simulated:
            raise GameHasNotBeenSimulated

        # Make dir
        dir_path = join(config.ANIMATIONS_DIR, model_name)
        if exists(dir_path):
            rmtree(dir_path)
        mkdir(dir_path)

        # Images
        print('Creating images...')
        pbar = tqdm(total=len(game.states))
        for epoch in range(len(game.states)):
            Visualization.plot_state(game=game, epoch=epoch, save=True, save_path=join(dir_path, f'{epoch}.png'))
            pbar.update(1)
        pbar.close()

        # Create video from images
        video_name = join(config.ANIMATIONS_DIR, model_name + '.mp4')

        images = [img for img in listdir(dir_path) if img.endswith('.png')]
        images_dict = {}
        for img in images:
            num = str(img.split(sep='.')[0])
            images_dict[num] = img
        images = []
        for key in range(len(images_dict)):
            images.append(images_dict[str(key)])
        frame = cv2.imread(join(dir_path, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
        for img in images:
            video.write(cv2.imread(join(dir_path, img)))

        video.release()
        cv2.destroyAllWindows()

    @staticmethod
    def plot_square(*, cell: Cell, max_y: int, ax) -> None:
        x, y = cell.j, max_y - cell.i - 1
        square = patches.Rectangle(xy=(x, y), width=1, height=1, edgecolor='none', facecolor='r')
        ax.add_patch(square)
