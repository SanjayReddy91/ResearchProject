from __future__ import print_function

from operator import attrgetter

#from gaps import utils
#from gaps.crossover import Crossover
#from gaps.image_analysis import ImageAnalysis
#from gaps.individual import Individual
#from gaps.plot import Plot
#from gaps.progress_bar import print_progress
#from gaps.selection import roulette_selection

import heapq
import random
from typing import List, Tuple, Dict
import numpy as np
import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
import bisect


def roulette_selection(population, elites=4):
    """Roulette wheel selection.

    Each individual is selected to reproduce, with probability directly
    proportional to its fitness score.

    :params population: Collection of the individuals for selecting.
    :params elite: Number of elite individuals passed to next generation.

    Usage::

        >>> from gaps.selection import roulette_selection
        >>> selected_parents = roulette_selection(population, 10)

    """
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [
        sum(fitness_values[: i + 1]) for i in range(len(fitness_values))
    ]

    def select_individual():
        """Selects random individual from population based on fitess value"""
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected = []
    for i in range(len(population) - elites):
        first, second = select_individual(), select_individual()
        selected.append((first, second))

    return selected


warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


class Plot(object):
    def __init__(self, image, title="Initial problem"):
        aspect_ratio = image.shape[0] / float(image.shape[1])

        width = 8
        height = width * aspect_ratio
        fig = plt.figure(figsize=(width, height), frameon=False)

        # Let image fill the figure
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)

        self._current_image = ax.imshow(image, aspect="auto", animated=True)
        self.show_fittest(image, title)

    def show_fittest(self, image, title):
        plt.suptitle(title, fontsize=20)
        self._current_image.set_data(image)
        plt.draw()

        # Give pyplot 0.05s to draw image
        plt.pause(0.05)



def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=50):
    """Call in a loop to create terminal progress bar"""
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "\033[32m█\033[0m" * filled_length + "\033[31m-\033[0m" * (
        bar_length - filled_length
    )

    sys.stdout.write(
        "\r{0: <16} {1} {2}{3} {4}".format(prefix, bar, percents, "%", suffix)
    )

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces
    in the original image tend to share similar colors along their abutting
    edges, i.e., the sum (over all neighboring pixels) of squared color
    differences (over all three color bands) should be minimal. Let pieces pi ,
    pj be represented in normalized L*a*b* space by corresponding W x W x 3
    matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    """
    rows, columns, _ = first_piece.shape()
    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        color_difference = (
            first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]
        )

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        color_difference = (
            first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]
        )

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)

    return value



class ImageAnalysis(object):
    """Cache for dissimilarity measures of individuals

    Class have static lookup table where keys are Piece's id's.  For each pair
    puzzle pieces there is a map with values representing dissimilarity measure
    between them. Each next generation have greater chance to use cached value
    instead of calculating measure again.

    Attributes:
        dissimilarity_measures: Dictionary with cached dissimilarity measures for pieces
        best_match_table: Dictionary with best matching piece for each edge and piece

    """

    dissimilarity_measures: Dict[Tuple, Dict[str, float]] = {}
    best_match_table: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    @classmethod
    def analyze_image(cls, pieces):
        for piece in pieces:
            # For each edge we keep best matches as a sorted list.
            # Edges with lower dissimilarity_measure have higher priority.
            cls.best_match_table[piece.id] = {"T": [], "R": [], "D": [], "L": []}

        def update_best_match_table(first_piece, second_piece):
            measure = dissimilarity_measure(first_piece, second_piece, orientation)
            cls.put_dissimilarity(
                (first_piece.id, second_piece.id), orientation, measure
            )
            cls.best_match_table[second_piece.id][orientation[0]].append(
                (first_piece.id, measure)
            )
            cls.best_match_table[first_piece.id][orientation[1]].append(
                (second_piece.id, measure)
            )

        # Calculate dissimilarity measures and best matches for each piece.
        iterations = len(pieces) - 1
        for first in range(iterations):
            print_progress(first, iterations - 1, prefix="=== Analyzing image:")
            for second in range(first + 1, len(pieces)):
                for orientation in ["LR", "TD"]:
                    update_best_match_table(pieces[first], pieces[second])
                    update_best_match_table(pieces[second], pieces[first])

        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])

    @classmethod
    def put_dissimilarity(cls, ids, orientation, value):
        """Puts a new value in lookup table for given pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'
        :params value:       Value of dissimilarity measure

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.put_dissimilarity([1, 2], "TD", 42)
        """
        if ids not in cls.dissimilarity_measures:
            cls.dissimilarity_measures[ids] = {}
        cls.dissimilarity_measures[ids][orientation] = value

    @classmethod
    def get_dissimilarity(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.get_dissimilarity([1, 2], "TD")

        """
        return cls.dissimilarity_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece, orientation):
        """ "Returns best match piece for given piece and orientation"""
        return cls.best_match_table[piece][orientation][0][0]


class Piece(object):
    """Represents single jigsaw puzzle piece.

    Each piece has identifier so it can be
    tracked across different individuals

    :param image: ndarray representing piece's RGB values
    :param index: Unique id withing piece's parent image

    Usage::

        >>> from gaps.piece import Piece
        >>> piece = Piece(image[:28, :28, :], 42)

    """

    def __init__(self, image, index):
        self.image = image[:]
        self.id = index

    def __getitem__(self, index):
        return self.image.__getitem__(index)

    def size(self):
        """Returns piece size"""
        return self.image.shape[0]

    def shape(self):
        """Returns shape of piece's image"""
        return self.image.shape


def flatten_image(image, piece_size, indexed=False):
    """Converts image into list of square pieces.

    Input image is divided into square pieces of specified size and than
    flattened into list. Each list element is PIECE_SIZE x PIECE_SIZE x 3

    :params image:      Input image.
    :params piece_size: Size of single square piece.
    :params indexed: If True list of Pieces with IDs will be returned,
        otherwise list of ndarray pieces

    Usage::

        >>> from gaps.image_helpers import flatten_image
        >>> flat_image = flatten_image(image, 32)

    """
    rows, columns = image.shape[0] // piece_size, image.shape[1] // piece_size
    pieces = []

    # Crop pieces from original image
    for y in range(rows):
        for x in range(columns):
            left, top, w, h = (
                x * piece_size,
                y * piece_size,
                (x + 1) * piece_size,
                (y + 1) * piece_size,
            )
            piece = np.empty((piece_size, piece_size, image.shape[2]))
            piece[:piece_size, :piece_size, :] = image[top:h, left:w, :]
            pieces.append(piece)

    if indexed:
        pieces = [Piece(value, index) for index, value in enumerate(pieces)]

    return pieces, rows, columns


def assemble_image(pieces, rows, columns):
    """Assembles image from pieces.

    Given an array of pieces and desired image dimensions, function assembles
    image by stacking pieces.

    :params pieces:  Image pieces as an array.
    :params rows:    Number of rows in resulting image.
    :params columns: Number of columns in resulting image.

    Usage::

        >>> from gaps.image_helpers import assemble_image
        >>> from gaps.image_helpers import flatten_image
        >>> pieces, rows, cols = flatten_image(...)
        >>> original_img = assemble_image(pieces, rows, cols)

    """
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))
    return np.vstack(vertical_stack).astype(np.uint8)


class Individual(object):
    """Class representing possible solution to puzzle.

    Individual object is one of the solutions to the problem
    (possible arrangement of the puzzle's pieces).
    It is created by random shuffling initial puzzle.

    :param pieces:  Array of pieces representing initial puzzle.
    :param rows:    Number of rows in input puzzle
    :param columns: Number of columns in input puzzle

    Usage::

        >>> from gaps.individual import Individual
        >>> from gaps.image_helpers import flatten_image
        >>> pieces, rows, columns = flatten_image(...)
        >>> ind = Individual(pieces, rows, columns)

    """

    FITNESS_FACTOR = 1000

    def __init__(self, pieces, rows, columns, shuffle=True):
        self.pieces = pieces[:]
        self.rows = rows
        self.columns = columns
        self._fitness = None

        if shuffle:
            np.random.shuffle(self.pieces)

        # Map piece ID to index in Individual's list
        self._piece_mapping = {
            piece.id: index for index, piece in enumerate(self.pieces)
        }

    def __getitem__(self, key):
        return self.pieces[key * self.columns : (key + 1) * self.columns]

    @property
    def fitness(self):
        """Evaluates fitness value.

        Fitness value is calculated as sum of dissimilarity measures between
        each adjacent pieces.

        """
        if self._fitness is None:
            fitness_value = 1 / self.FITNESS_FACTOR
            # For each two adjacent pieces in rows
            for i in range(self.rows):
                for j in range(self.columns - 1):
                    ids = (self[i][j].id, self[i][j + 1].id)
                    fitness_value += ImageAnalysis.get_dissimilarity(
                        ids, orientation="LR"
                    )
            # For each two adjacent pieces in columns
            for i in range(self.rows - 1):
                for j in range(self.columns):
                    ids = (self[i][j].id, self[i + 1][j].id)
                    fitness_value += ImageAnalysis.get_dissimilarity(
                        ids, orientation="TD"
                    )

            self._fitness = self.FITNESS_FACTOR / fitness_value

        return self._fitness

    def piece_size(self):
        """Returns single piece size"""
        return self.pieces[0].size

    def piece_by_id(self, identifier):
        """ "Return specific piece from individual"""
        return self.pieces[self._piece_mapping[identifier]]

    def to_image(self):
        """Converts individual to showable image"""
        pieces = [piece.image for piece in self.pieces]
        return assemble_image(pieces, self.rows, self.columns)

    def edge(self, piece_id, orientation):
        edge_index = self._piece_mapping[piece_id]

        if (orientation == "T") and (edge_index >= self.columns):
            return self.pieces[edge_index - self.columns].id

        if (orientation == "R") and (edge_index % self.columns < self.columns - 1):
            return self.pieces[edge_index + 1].id

        if (orientation == "D") and (edge_index < (self.rows - 1) * self.columns):
            return self.pieces[edge_index + self.columns].id

        if (orientation == "L") and (edge_index % self.columns > 0):
            return self.pieces[edge_index - 1].id


def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=50):
    """Call in a loop to create terminal progress bar"""
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "\033[32m█\033[0m" * filled_length + "\033[31m-\033[0m" * (
        bar_length - filled_length
    )

    sys.stdout.write(
        "\r{0: <16} {1} {2}{3} {4}".format(prefix, bar, percents, "%", suffix)
    )

    if iteration == total:
        sys.stdout.write("\n")
    sys.stdout.flush()

def dissimilarity_measure(first_piece, second_piece, orientation="LR"):
    """Calculates color difference over all neighboring pixels over all color channels.

    The dissimilarity measure relies on the premise that adjacent jigsaw pieces
    in the original image tend to share similar colors along their abutting
    edges, i.e., the sum (over all neighboring pixels) of squared color
    differences (over all three color bands) should be minimal. Let pieces pi ,
    pj be represented in normalized L*a*b* space by corresponding W x W x 3
    matrices, where W is the height/width of each piece (in pixels).

    :params first_piece:  First input piece for calculation.
    :params second_piece: Second input piece for calculation.
    :params orientation:  How input pieces are oriented.

                          LR => 'Left - Right'
                          TD => 'Top - Down'

    Usage::

        >>> from gaps.fitness import dissimilarity_measure
        >>> from gaps.piece import Piece
        >>> p1, p2 = Piece(), Piece()
        >>> dissimilarity_measure(p1, p2, orientation="TD")

    """
    rows, columns, _ = first_piece.shape()
    color_difference = None

    # | L | - | R |
    if orientation == "LR":
        color_difference = (
            first_piece[:rows, columns - 1, :] - second_piece[:rows, 0, :]
        )

    # | T |
    #   |
    # | D |
    if orientation == "TD":
        color_difference = (
            first_piece[rows - 1, :columns, :] - second_piece[0, :columns, :]
        )

    squared_color_difference = np.power(color_difference / 255.0, 2)
    color_difference_per_row = np.sum(squared_color_difference, axis=1)
    total_difference = np.sum(color_difference_per_row, axis=0)

    value = np.sqrt(total_difference)

    return value



class ImageAnalysis(object):
    """Cache for dissimilarity measures of individuals

    Class have static lookup table where keys are Piece's id's.  For each pair
    puzzle pieces there is a map with values representing dissimilarity measure
    between them. Each next generation have greater chance to use cached value
    instead of calculating measure again.

    Attributes:
        dissimilarity_measures: Dictionary with cached dissimilarity measures for pieces
        best_match_table: Dictionary with best matching piece for each edge and piece

    """

    dissimilarity_measures: Dict[Tuple, Dict[str, float]] = {}
    best_match_table: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    @classmethod
    def analyze_image(cls, pieces):
        for piece in pieces:
            # For each edge we keep best matches as a sorted list.
            # Edges with lower dissimilarity_measure have higher priority.
            cls.best_match_table[piece.id] = {"T": [], "R": [], "D": [], "L": []}

        def update_best_match_table(first_piece, second_piece):
            measure = dissimilarity_measure(first_piece, second_piece, orientation)
            cls.put_dissimilarity(
                (first_piece.id, second_piece.id), orientation, measure
            )
            cls.best_match_table[second_piece.id][orientation[0]].append(
                (first_piece.id, measure)
            )
            cls.best_match_table[first_piece.id][orientation[1]].append(
                (second_piece.id, measure)
            )

        # Calculate dissimilarity measures and best matches for each piece.
        iterations = len(pieces) - 1
        for first in range(iterations):
            print_progress(first, iterations - 1, prefix="=== Analyzing image:")
            for second in range(first + 1, len(pieces)):
                for orientation in ["LR", "TD"]:
                    update_best_match_table(pieces[first], pieces[second])
                    update_best_match_table(pieces[second], pieces[first])

        for piece in pieces:
            for orientation in ["T", "L", "R", "D"]:
                cls.best_match_table[piece.id][orientation].sort(key=lambda x: x[1])

    @classmethod
    def put_dissimilarity(cls, ids, orientation, value):
        """Puts a new value in lookup table for given pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'
        :params value:       Value of dissimilarity measure

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.put_dissimilarity([1, 2], "TD", 42)
        """
        if ids not in cls.dissimilarity_measures:
            cls.dissimilarity_measures[ids] = {}
        cls.dissimilarity_measures[ids][orientation] = value

    @classmethod
    def get_dissimilarity(cls, ids, orientation):
        """Returns previously cached dissimilarity measure for input pieces

        :params ids:         Identfiers of puzzle pieces
        :params orientation: Orientation of puzzle pieces. Possible values are:
                             'LR' => 'Left-Right'
                             'TD' => 'Top-Down'

        Usage::

            >>> from gaps.image_analysis import ImageAnalysis
            >>> ImageAnalysis.get_dissimilarity([1, 2], "TD")

        """
        return cls.dissimilarity_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece, orientation):
        """ "Returns best match piece for given piece and orientation"""
        return cls.best_match_table[piece][orientation][0][0]



SHARED_PIECE_PRIORITY = -10
BUDDY_PIECE_PRIORITY = -1


class Crossover(object):
    def __init__(self, first_parent, second_parent):
        self._parents = (first_parent, second_parent)
        self._pieces_length = len(first_parent.pieces)
        self._child_rows = first_parent.rows
        self._child_columns = first_parent.columns

        # Borders of growing kernel
        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0

        self._kernel = {}
        self._taken_positions = set()

        # Priority queue
        self._candidate_pieces = []

    def child(self):
        pieces = [None] * self._pieces_length

        for piece, (row, column) in self._kernel.items():
            index = (row - self._min_row) * self._child_columns + (
                column - self._min_column
            )
            pieces[index] = self._parents[0].piece_by_id(piece)

        return Individual(pieces, self._child_rows, self._child_columns, shuffle=False)

    def run(self):
        self._initialize_kernel()

        while len(self._candidate_pieces) > 0:
            _, (position, piece_id), relative_piece = heapq.heappop(
                self._candidate_pieces
            )

            if position in self._taken_positions:
                continue

            # If piece is already placed, find new piece candidate and put it back to
            # priority queue
            if piece_id in self._kernel:
                self.add_piece_candidate(relative_piece[0], relative_piece[1], position)
                continue

            self._put_piece_to_kernel(piece_id, position)

    def _initialize_kernel(self):
        root_piece = self._parents[0].pieces[
            int(random.uniform(0, self._pieces_length))
        ]
        self._put_piece_to_kernel(root_piece.id, (0, 0))

    def _put_piece_to_kernel(self, piece_id, position):
        self._kernel[piece_id] = position
        self._taken_positions.add(position)
        self._update_candidate_pieces(piece_id, position)

    def _update_candidate_pieces(self, piece_id, position):
        available_boundaries = self._available_boundaries(position)

        for orientation, position in available_boundaries:
            self.add_piece_candidate(piece_id, orientation, position)

    def add_piece_candidate(self, piece_id, orientation, position):
        shared_piece = self._get_shared_piece(piece_id, orientation)
        if self._is_valid_piece(shared_piece):
            self._add_shared_piece_candidate(
                shared_piece, position, (piece_id, orientation)
            )
            return

        buddy_piece = self._get_buddy_piece(piece_id, orientation)
        if self._is_valid_piece(buddy_piece):
            self._add_buddy_piece_candidate(
                buddy_piece, position, (piece_id, orientation)
            )
            return

        best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)
        if self._is_valid_piece(best_match_piece):
            self._add_best_match_piece_candidate(
                best_match_piece, position, priority, (piece_id, orientation)
            )
            return

    def _get_shared_piece(self, piece_id, orientation):
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.edge(piece_id, orientation)
        second_parent_edge = second_parent.edge(piece_id, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _get_buddy_piece(self, piece_id, orientation):
        first_buddy = ImageAnalysis.best_match(piece_id, orientation)
        second_buddy = ImageAnalysis.best_match(
            first_buddy, complementary_orientation(orientation)
        )

        if second_buddy == piece_id:
            for edge in [
                parent.edge(piece_id, orientation) for parent in self._parents
            ]:
                if edge == first_buddy:
                    return edge

    def _get_best_match_piece(self, piece_id, orientation):
        for piece, dissimilarity_measure in ImageAnalysis.best_match_table[piece_id][
            orientation
        ]:
            if self._is_valid_piece(piece):
                return piece, dissimilarity_measure

    def _add_shared_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = (SHARED_PIECE_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_buddy_piece_candidate(self, piece_id, position, relative_piece):
        piece_candidate = (BUDDY_PIECE_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_best_match_piece_candidate(
        self, piece_id, position, priority, relative_piece
    ):
        piece_candidate = (priority, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _available_boundaries(self, row_and_column):
        (row, column) = row_and_column
        boundaries = []

        if not self._is_kernel_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1),
            }

            for orientation, position in positions.items():
                if position not in self._taken_positions and self._is_in_range(
                    position
                ):
                    self._update_kernel_boundaries(position)
                    boundaries.append((orientation, position))

        return boundaries

    def _is_kernel_full(self):
        return len(self._kernel) == self._pieces_length

    def _is_in_range(self, row_and_column):
        (row, column) = row_and_column
        return self._is_row_in_range(row) and self._is_column_in_range(column)

    def _is_row_in_range(self, row):
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _is_column_in_range(self, column):
        current_columns = abs(min(self._min_column, column)) + abs(
            max(self._max_column, column)
        )
        return current_columns < self._child_columns

    def _update_kernel_boundaries(self, row_and_column):
        (row, column) = row_and_column
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id):
        return piece_id is not None and piece_id not in self._kernel


def complementary_orientation(orientation):
    return {"T": "D", "R": "L", "D": "T", "L": "R"}.get(orientation, None)


class GeneticAlgorithm(object):
    TERMINATION_THRESHOLD = 10

    def __init__(self, image, piece_size, population_size, generations, elite_size=2):
        self._image = image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size
        pieces, rows, columns = flatten_image(image, piece_size, indexed=True)
        self._population = [
            Individual(pieces, rows, columns) for _ in range(population_size)
        ]
        self._pieces = pieces

    def start_evolution(self, verbose):
        print("=== Pieces:      {}\n".format(len(self._pieces)))

        if verbose:
            plot = Plot(self._image)

        ImageAnalysis.analyze_image(self._pieces)

        fittest = None
        best_fitness_score = float("-inf")
        termination_counter = 0

        for generation in range(self._generations):
            print_progress(
                generation, self._generations - 1, prefix="=== Solving puzzle: "
            )

            new_population = []

            # Elitism
            elite = self._get_elite_individuals(elites=self._elite_size)
            new_population.extend(elite)

            selected_parents = roulette_selection(
                self._population, elites=self._elite_size
            )

            for first_parent, second_parent in selected_parents:
                crossover = Crossover(first_parent, second_parent)
                crossover.run()
                child = crossover.child()
                new_population.append(child)

            fittest = self._best_individual()

            if fittest.fitness <= best_fitness_score:
                termination_counter += 1
            else:
                best_fitness_score = fittest.fitness

            if termination_counter == self.TERMINATION_THRESHOLD:
                print("\n\n=== GA terminated")
                print(
                    "=== There was no improvement for {} generations".format(
                        self.TERMINATION_THRESHOLD
                    )
                )
                return fittest

            self._population = new_population

            if verbose:
                plot.show_fittest(
                    fittest.to_image(),
                    "Generation: {} / {}".format(generation + 1, self._generations),
                )

        return fittest

    def _get_elite_individuals(self, elites):
        """Returns first 'elite_count' fittest individuals from population"""
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        """Returns the fittest individual from population"""
        return max(self._population, key=attrgetter("fitness"))
