import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]


def get_column(level, column):
    """returns a string representation of a column of level"""
    string = ""
    for j in range(height):
        string += level[j][column]
    return string

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        def changePipeHeight(g, prow, pcol, amount):
            """changes the height of a pipe located a prow, pcol by amount. amount can be negative or positive. if pipe
            would go off screen, clips it at the top. if pipe would go underground, deletes pipe. A positive amount will
            make pipe shorter and a negative amount will make pipe taller"""

            # i don't actually use this function, but im keeping it bc i spent too long on it to just delete it

            if amount > 0:
                for row in range(amount):
                    row += prow
                    if row == height-1:
                        break
                    elif row + 1 != amount+prow:
                        g[row][pcol] = '-'
                    else:
                        g[row][pcol] = '-'
                        g[row+1][pcol] = 'T'
            else:
                for row in range(0, amount, -1):
                    row += prow
                    if row == 0:
                        g[row][pcol] = 'T'
                        break
                    elif row - 1 != amount+prow:
                        g[row][pcol] = '|'
                    else:
                        g[row-1][pcol] = 'T'
                        g[row][pcol] = '|'

        def moveBlock(g, row, col, ramt, camt):
            """Moves the block at row, col by ramt in the y direction and camt in the x direction. If the block would be
            moved off screen or into the ground, does nothing. Can move a pipe, but only if pipe is the source block."""
            block = g[row][col]
            rdest = row + ramt
            cdest = col + camt
            if 0 <= rdest < height-1 and 0 <= cdest < width and block not in ['m', 'v', 'f']:
                dest_block = g[rdest][cdest]
                if dest_block not in ['m', 'v', 'f', 'T', '|'] and g[rdest][cdest-1] != '|':
                    if block == 'T':
                        if dest_block != '|' and rdest <= height-5:
                            g[row][col] = dest_block
                        else:
                            return
                        for i in range(height):
                            if g[i][col] == '|':
                                g[i][col] = '-'
                        g[rdest][cdest] = block
                        if cdest == width - 1:
                            g[rdest][cdest] = '-'
                        elif cdest != 0:
                            if 'T' in get_column(g, cdest - 1):
                                g[rdest][cdest] = '-'
                            else:
                                g[rdest][cdest + 1] = '-'
                                for i in range(height - (rdest + 2)):
                                    g[i + rdest + 1][cdest] = '|'
                                    g[i + rdest + 1][cdest + 1] = '-'
                        else:
                            g[rdest][cdest] = '-'
                    else:
                        g[row][col] = dest_block
                        g[rdest][cdest] = block

        left = 1
        right = width - 1

        mutateAtAllChance = 0.1  # a float between 0 and 1. 0 will never mutate and 1 will always mutate
        mut = 0.05   # the likelyhood of any specific block being mutated
        mut_dist_x = 2    # the maximum distance that a block will be moved in the x direction
        mut_dist_y = 2  # the maximum distance that a block will be moved in the y direction
        change_to = [   # list of blocks that a source block might be changed to
            "-",  # an empty space
            "X",  # a solid wall
            "?",  # a question mark block with a coin
            "M",  # a question mark block with a mushroom
            "B",  # a breakable block
            "o",  # a coin
            "E"   # an enemy
        ]

        weights = [
            35,  # an empty space
            1,  # a solid wall
            0.1,  # a question mark block with a coin
            0.1,  # a question mark block with a mushroom
            2,  # a breakable block
            1,  # a coin
            1.5  # an enemy
        ]

        mutable = [   # list of blocks that can be mutated
            "-",  # an empty space
            "X",  # a solid wall
            "?",  # a question mark block with a coin
            "M",  # a question mark block with a mushroom
            "B",  # a breakable block
            "o",  # a coin
            "E",  # an enemy
            "T"   # a pipe
        ]

        if mutateAtAllChance < random.random():
            return genome

        for y in range(height-1):
            for x in range(left, right):
                block = genome[y][x]
                if block in mutable and random.random() < mut and block != 'T' and genome[y][x-1] != '|':
                    genome[y][x] = random.choices(change_to, weights=weights)[0]
                    moveBlock(genome, y, x, math.ceil(random.random()*mut_dist_y), math.ceil(random.random()*mut_dist_x))
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other

        # Basic idea is to loop through 1 block at a time and randomly choose a block from one of the two parents, but
        # have the odds of a specific parent being chosen be based off of that parent's fitness. So if the fitnesses are
        # 60 and 40, then the child genome should have 60% of it's DNA from parent 1 and 40% from parent 2. We run the
        # sanity check afterwards to make sure there aren't any invalid block placements

        # Might want to instead select things by group, i.e. take pipes from parent 1, platforms from parent 2,
        # staircases from parent 1, enemies from parent 2, etc
        blocks = [
            "X",    # a solid wall
            "?",    # a question mark block with a coin
            "M",    # a question mark block with a mushroom
            "B",    # a breakable block
        ]


        sfit = self.fitness()
        ofit = other.fitness()

        pop = [1, 0]
        weights = [sfit, ofit]

        left = 1
        right = width - 1

        enemy_count = 0
        enemy_max = 6
        for y in range(height-1):
            for x in range(left, right):
                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

                #For each tile randomly decide (w/ weights) which parent to set the tile should pull from. 
                # If the y,x coord happens to be a pipe then simply make it an empty space tile  
                if random.choices(pop, weights=weights, k=1)[0]:
                    if new_genome[y][x] == '|':
                        new_genome[y][x] = '-'
                else:
                    new_genome[y][x] = other.genome[y][x]
                    if new_genome[y][x] == '|':
                        new_genome[y][x] = '-'

        # This is the sanity check copy/pasted from random_individual
        spawnable = ['X', 'M', '?', 'B']
        for row in range(height):
            for col in range(width):

                # No floating pipes. Also, all blocks to the right of a pipe should be air
                if new_genome[row][col] == 'T':
                    if col == width - 1:
                        new_genome[row][col] = '-'
                    elif col != 0:
                        if 'T' in get_column(new_genome, col - 1):
                            new_genome[row][col] = '-'
                        else:
                            new_genome[row][col + 1] = '-'
                            for i in range(height - (row + 2)):
                                new_genome[i + row + 1][col] = '|'
                                new_genome[i + row + 1][col + 1] = '-'
                    else:
                        new_genome[row][col] = '-'

                # No floating enemies or enemies more than enemy_max
                if new_genome[row][col] == 'E':
                    enemy_count += 1
                    if new_genome[row + 1][col] not in spawnable or enemy_count > enemy_max:
                        new_genome[row][col] = '-'

                # You need an empty space under a question mark block in order to jump into it
                if new_genome[row][col] == 'M' or new_genome[row][col] == '?' and new_genome[row + 1][col] != '-':
                    new_genome[row][col] = '-'
                # Remove blocks that are 1 tile away from another block since they can't be passed through
                if row > 2 and new_genome[row][col] in blocks and new_genome[row-1][col] == '-' and new_genome[row-2][col] in blocks:
                    new_genome[row][col] = '-'

        # do mutation; note we're returning a one-element tuple here
        self.mutate(new_genome)

        # ensure these essential tiles are placed
        new_genome[15][:] = ["X"] * width
        new_genome[14][0] = "m"
        new_genome[7][-1] = "v"
        for col in range(8, 14):
            new_genome[col][-1] = "f"
        for col in range(14, 16):
            new_genome[col][-1] = "X"

        parents = [self, other]
        best_parent = random.choices(parents, weights=weights, k=1)[0].genome
        for col in range(width-1):
            new_genome[-1][col] = best_parent[-1][col]

        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random

        pop = [
            "-",    # an empty space
            "X",    # a solid wall
            "?",    # a question mark block with a coin
            "M",    # a question mark block with a mushroom
            "B",    # a breakable block
            "o",    # a coin
            "|",    # a pipe segment
            "T",    # a pipe top
            "E"     # an enemy
        ]
        weights = [
            35,  # an empty space
            1,  # a solid wall
            0.1,# a question mark block with a coin
            0.1,# a question mark block with a mushroom
            2,  # a breakable block
            1,  # a coin
            0,  # a pipe segment
            0.6,  # a pipe top
            1.5   # an enemy
        ]
        g = [random.choices(pop, weights=weights, k=width) for row in range(height)]

        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"

        num_pits = math.ceil(random.random() * 5)
        pit_lengths = {}
        pit_start_locations = random.choices(range(1, width-2), k=num_pits)

        for i in pit_start_locations:
            pit_lengths[i] = math.ceil(random.random()*3)

        for loc in pit_start_locations:
            for i in range(0, pit_lengths[loc]):
                g[-1][loc + i] = '-'

        # Now we sanity check to prevent floating pipes, enemies, etc
        spawnable = ['X', 'M', '?', 'B']
        for row in range(height):
            for col in range(width):

                # No floating pipes. Also, all blocks to the right of a pipe should be air also don't make pipes taller
                # than 5
                if g[row][col] == 'T':
                    if col == width-1:
                        g[row][col] = '-'
                    elif row <= height - 5:
                        g[row][col] = '-'
                    elif col != 0:
                        if 'T' in get_column(g, col-1):
                            g[row][col] = '-'
                        else:
                            g[row][col + 1] = '-'
                            for i in range(height - (row+2)):
                                g[i + row + 1][col] = '|'
                                g[i + row + 1][col+1] = '-'
                    else:
                        g[row][col] = '-'

                # No floating enemies
                if g[row][col] == 'E':
                    if g[row + 1][col] not in spawnable:
                        g[row][col] = '-'

                # You need an empty space under a question mark block in order to jump into it
                if g[row][col] == 'M' or g[row][col] == '?' and g[row+1][col] != '-':
                    g[row][col] = '-'

        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        #print(measurements)
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0,
            decorationPercentage=0.0,
            leniency=0.0,
            meaningfulJumps=0.01,
            jumps=0.0,
            jumpVariance=0.0,
            length=0.0
        )

        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        
        # penalize for too many pipes
        if len(list(filter(lambda de: de[1] == "7_pipe", self.genome))) > 3:
            penalties -= 2

        num_tallpipes = 0
        tall_pipes_coefficient = 0.5
        sumDescent = 0
        sumAscent = 0
        lackOfAscent_coefficient = 0.2
        for de in self.genome:
            if de[1] == "6_stairs":
                if de[3] == 1:
                    sumAscent += 1
                else:
                    sumDescent += 1
            if de[1] == "7_pipe":
                if de[2] > 4:
                    num_tallpipes

        # penalize if there's more descending stairs than ascending stairs
        if sumDescent > sumAscent:
            penalties -= lackOfAscent_coefficient
        # penalize for pipes too tall
        penalties -= tall_pipes_coefficient*num_tallpipes

        # Reward levels with good jumps
        jumpVal = 1
        if measurements['meaningfulJumps'] >= 0.0:
            jumpVal += coefficients['meaningfulJumps']*measurements['meaningfulJumps']

        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties + jumpVal
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def get_stairs(self):
        """Returns a list of columns that have stairs in them"""
        # 6_stairs: (x, type, height, ascend/descend)
        columns = []
        for i in self.genome:
            if i[1] == "6_stairs":
                for j in range(i[2] - 1):
                    columns.append(i[0] + 2 + j)
        return columns

    def get_pipes(self):
        """Returns a list in the form of [(pipe1x, pipe1y), (pipe2x, pipe2y)...] for all pipes in genome"""
        pipes = []
        for i in self.genome:
            if i[1] == "7_pipe":
                pipes.append(i[0])
        return pipes

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things

        # these will be used to prevent mutate from mutating to an invalid genome
        pipes = self.get_pipes()
        stairs = self.get_stairs()

        # 10% chance to mutate, 90% chance to do nothing
        if random.random() < 0.1 and len(new_genome) > 0:

            # this part selects a single element from the genome at random
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]

            # choice is how much to mutate. the larger choice is, the more the element will be mutated
            choice = random.random()

            if de_type == "4_block":
                # move blocks in both axes and change breakability
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                # move ? blocks in both axes and change contents
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                # move coins
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                # move pipes
                h = de[2]
                did_succeed = False
                for attempt in range(10):
                    succeeded = True
                    # choose amount to move by
                    if choice < 0.5:
                        x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                    else:
                        h = offset_by_upto(h, 2, min=2, max=height - 4)
                    # if new pipe conflicts with stairs, try again
                    if x in stairs or x+1 in stairs:
                        succeeded = False
                    # if new pipe goes off screen, try again
                    elif x < 2 or x > width - 2:
                        succeeded = False
                    # if new pipe conflicts with other pipes, try again
                    elif x - 1 in pipes or x in pipes or x + 1 in pipes:
                        succeeded = False
                    if succeeded:
                        did_succeed = True
                        break
                    else:
                        # need to choose a new mutate amount
                        choice = random.random()
                if did_succeed:
                    new_de = (x, de_type, h)
                else:
                    new_de = de
            elif de_type == "0_hole":
                # move holes
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                # move stairs, change height, change ascend/descend
                h = de[2]
                dx = de[3]  # -1 or 1

                # get a list of my stair locations
                my_locations = []
                for i in range(de[0] - 1):
                    my_locations.append(i + de[2] + 2)
                # get a list of all other stair locations
                other_locations = []
                for i in stairs:
                    if i not in my_locations:
                        other_locations.append(i)

                # Here we will try to mutate the stairs 10 times. If all 10 attempts fail, then give up
                did_succeed = False
                for attempt in range(10):
                    succeeded = True
                    if choice < 0.33:
                        x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                    elif choice < 0.66:
                        h = offset_by_upto(h, 8, min=2, max=8)
                    else:
                        dx = -dx
                    # sanity check time
                    new_locations = []
                    # get location of new mutated stairs
                    for i in range(h):
                        new_locations.append(i + x + 2)
                    # if new stairs overlap with other stairs or pipes, try again
                    for i in new_locations:
                        if i in pipes or i in other_locations:
                            succeeded = False
                    # if new stairs go off screen, try again
                    if x < 2 or max(new_locations) > width - 2:
                        succeeded = False
                    if succeeded:
                        did_succeed = True
                        break
                    else:
                        # need to choose a new mutate amount
                        choice = random.random()
                if did_succeed:
                    new_de = (x, de_type, h, dx)
                else:
                    new_de = de
            elif de_type == "1_platform":
                # move platforms, change length, and change block type
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                # do not do anything to enemies
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    # (x, type, y, Break? y/n)
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    # (x, type, y, Shroom? y/n)
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    # (x, type, y)
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    # (x, type, y)
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    # (x, type, length)
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    # (x, type, height, ascend/descend)
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    # (x, type, length, y, block_type)
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    # (x, type)
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"]))
        ]) for i in range(elt_count)]
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this

        # 0_hole    (x, type, length)
        # 1_platform(x, type, length, y, block_type)
        # 2_enemy   (x, type)
        # 3_coin    (x, type, y)
        # 4_block   (x, type, y, Break? y/n)
        # 5_qblock  (x, type, y, Shroom? y/n)
        # 6_stairs  (x, type, height, ascend/descend)   NOTE: these are REALLY shittily encoded. the *actual* start
                                                        # location is x + 2 and the actual length is length - 1
        # 7_pipe    (x, type, y)

        weights = [
            0.2,    # hole
            0.5,    # platform
            0.4,    # enemy
            0.7,    # coin
            0.5,    # block
            0.2,    # qblock
            0.7,    # stairs
            0.6     # pipe
        ]

        elt_count = random.randint(8, 160)
        g = []
        stair_locs = [] # this will keep track of the locations of all the stairs as we generate them rather than having
        # to search for them every time

        pipe_locs = []  # same as above but for pipes

        # so elt_count is the number of elements we will *try* to add. Note that if adding an element fails, it will
        # simply not be added. So that means that for particularly picky elements like stairs and pipes, their actual
        # frequency will be significantly less than their weight would suggest as many attempts to add them will fail.
        for dummy in range(elt_count):
            elem = random.choices([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 6)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(1, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(1, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(1, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(1, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(2, 8), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, 4))],
            weights=weights, k=1)[0]

            # Dont let elements overlap with stairs (most of the time)
            # stairs can still be placed on top of other elements (except other stairs and pipes bc i wrote special
            # cases below) but tbh it's not that big of a deal. It's just that if we have too many blocks above the
            # stairs then things can pretty easily get unbeatable, so this will cut down the number of things above the
            # stairs significantly
            if elem[0] in stair_locs:
                continue
            # make sure stairs don't overlap stairs (all the time)
            elif elem[1] == "6_stairs":
                bad = False
                # stairs cannot go off screen
                if elem[0] + elem[2] - 1 > width - 2:
                    bad = True
                # no overlap
                for j in range(elem[2] - 1):
                    if elem[0] + 2 + j in stair_locs or elem[0] + 2 + j in pipe_locs:
                        bad = True
                if not bad:
                    g.append(elem)
                else:
                    continue
            # pipes need space
            elif elem[1] == "7_pipe":
                if elem[0] in pipe_locs or elem[0] + 1 in pipe_locs or elem [0] - 1 in pipe_locs:
                    continue
                else:
                    pipe_locs.append(elem[0])
                    g.append(elem)
            else:
                g.append(elem)

            # special rules for stairs
            if elem[1] == "6_stairs":
                # If the stairs are descending, we should put some ascending stairs opposite them
                if elem[3] == -1:
                    length = elem[2]
                    op_x = elem[0] - length - random.randint(1, 4)
                    new_stair_locs = []
                    abort = False
                    # get location of new stairs
                    for i in range(elem[2] - 1):
                        new_stair_locs.append(i + op_x + 2)
                    # these new stairs must not overlap with pipes or other stairs and must not go off screen
                    for i in new_stair_locs:
                        if i in stair_locs or i < 1 or i > width - 2 or i in pipe_locs:
                            abort = True
                            break
                    if not abort:
                        g.append((op_x, "6_stairs", elem[2] - 1, 1))
                        stair_locs.append(new_stair_locs)
                # update the list of stair locations
                for j in range(elem[2] - 1):
                    stair_locs.append(elem[0] + 2 + j)

        # original code that im too scared to delete just in case
        # g = [random.choice([
        #     (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
        #     (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
        #     (random.randint(1, width - 2), "2_enemy"),
        #     (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
        #     (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
        #     (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
        #     (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
        #     (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        # ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE


def generate_successors(population):
    results = [] # return list of evolved levels
    # STUDENT Design and implement this

    tournament_chosen = tournament_selection(population) # get half the population that are winners 
    # roulette_chosen = roulette_selection(population)
    elitist_chosen = elitist_selection(population)

    # mate the determined successors to get new children
    for index in range(0, len(tournament_chosen)):
        parent1 = tournament_chosen[index]
        parent2 = elitist_chosen[index]
        results.append(parent1.generate_children(parent2)[0])
        results.append(parent2.generate_children(parent1)[0])
    # Hint: Call generate_children() on some individuals and fill up results.
    return results

# still needs work, not sure how to do roulette_selection with negative values
# references: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
# https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)#Methods_of_Selection_(Genetic_Algorithm)
# https://en.wikipedia.org/wiki/Fitness_proportionate_selection
def roulette_selection(population):
    selected  = []
    # get sum of fitnesses
    fitnessSum = 0
    for ind in population:
        fitnessSum += ind.fitness()

    # set up probabilites list and 
    probabilities = []
    sorted_population = sorted(population, key=lambda obj: obj.fitness())[::-1]
    for ind in sorted_population:
        probability = ind.fitness()/fitnessSum
        probabilities.append((probability, ind))

    # iterate 
    for i in range(0, math.floor(len(population)/2)): 
        return fitnessSum

    return selected

def tournament_selection(population):
    # if there's not enough individuals to complete just return itself
    if len(population) < 2:
        return population
    
    winners = []
    randomized = random.shuffle(copy.deepcopy(population))
    for i in range(0, math.floor(len(population)/2)):
        contestant_1 = population[i]
        contestant_2 = population[i+1]
        if contestant_1._fitness > contestant_2._fitness:
            winners.append(contestant_1)
        else:
            winners.append(contestant_2)
    
    # if the population size was an odd number then the last player automatically gets passed through
    if (len(population) % 2) != 0:
        winners.append(populatoin[-1])

    return winners

# returns the top half of the population based on fitness
def elitist_selection(population):
    elitist = []
    sorted_pop = sorted(population, key=lambda obj: obj.fitness())
    for i in range(0, math.floor(len(population)/2)):
        elitist.append(sorted_pop[i])

    # if the population size was an odd number then add one more individual
    if (len(population) % 2) != 0:
        index = math.floor(len(population)/2)
        elitist.append(sorted_pop[index])

    return elitist

def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.empty_individual() if random.random() < 0.9
                      else Individual.random_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        # sets the initialize fitness
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        count = 1
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    # print("Generation:", str(generation))
                    best = max(population, key=Individual.fitness)
                    print("\nGeneration:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                    # print("Best Individual:")
                    # for line in best.genome:
                    #     print(listToString(line))
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = time.time()-start > 60
                if (generation > 15):
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                print("pop length", len(population))
                gentime = time.time()
                next_population = generate_successors(population) # SELECTION STRATEGY
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # CALCULATE FITNESS in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
