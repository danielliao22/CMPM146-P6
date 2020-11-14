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
                    print("lookin at", row)
                    print("amount+prow:", amount+prow)
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
            # print("Block:", block, "rdest:", rdest, "cdest:", cdest)
            if 0 <= rdest < height-1 and 0 <= cdest < width and block not in ['m', 'v', 'f']:
                dest_block = g[rdest][cdest]
                # print("dest_block:", dest_block)
                if dest_block not in ['m', 'v', 'f', 'T', '|'] and g[rdest][cdest-1] != '|':
                    if block == 'T':
                        print("Block:", block, "rdest:", rdest, "cdest:", cdest)
                        for i in range(height):
                            if g[i][col] == '|':
                                g[i][col] = '-'
                        if dest_block != '|':
                            g[row][col] = dest_block
                        else:
                            g[row][col] = '-'
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
            # else:
                # print("out of range or illegal move\n")


        """CODE BETWEEN THESE LINES IS FOR TESTING PURPOSES ONLY, PLEASE COMMENT OUT LATER"""
        # changePipeHeight(genome, 0, 85, -1)
        # moveBlock(genome, 0, 85, 7, 2)
        """CODE BETWEEN THESE LINES IS FOR TESTING PURPOSES ONLY, PLEASE COMMENT OUT LATER"""

        left = 1
        right = width - 1

        mutateAtAllChance = 1  # a float between 0 and 1. 0 will never mutate and 1 will always mutate
        mut = 0.1   # the likelyhood of any specific block being mutated
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
            25,  # an empty space
            1,  # a solid wall
            0.1,  # a question mark block with a coin
            0.1,  # a question mark block with a mushroom
            2,  # a breakable block
            3,  # a coin
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
            # print("Did not mutate")
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

        sfit = self.fitness()
        ofit = other.fitness()

        pop = [1, 0]
        weights = [sfit, ofit]

        left = 1
        right = width - 1
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
                    new_genome[y][x] = other[y][x]
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
                            g[row][col + 1] = '-'
                            for i in range(height - (row + 2)):
                                new_genome[i + row + 1][col] = '|'
                                new_genome[i + row + 1][col + 1] = '-'
                    else:
                        new_genome[row][col] = '-'

                # No floating enemies
                if new_genome[row][col] == 'E':
                    if new_genome[row + 1][col] not in spawnable:
                        new_genome[row][col] = '-'

                # You need an empty space under a question mark block in order to jump into it
                if new_genome[row][col] == 'M' or new_genome[row][col] == '?' and new_genome[row + 1][col] != '-':
                    new_genome[row][col] = '-'

        # do mutation; note we're returning a one-element tuple here
        mutate(new_genome)
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
            25,  # an empty space
            1,  # a solid wall
            0.1,# a question mark block with a coin
            0.1,# a question mark block with a mushroom
            2,  # a breakable block
            3,  # a coin
            0,  # a pipe segment
            0.1,  # a pipe top
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

        # Now we sanity check to prevent floating pipes, enemies, etc
        spawnable = ['X', 'M', '?', 'B']
        for row in range(height):
            for col in range(width):

                # No floating pipes. Also, all blocks to the right of a pipe should be air
                if g[row][col] == 'T':
                    if col == width-1:
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
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
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
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
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
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_Grid


def generate_successors(population):
    results = [] # return list of evolved levels
    # STUDENT Design and implement this

    roulette_chosen = roulette_selection(population)
    tournament_chosen = tournament_selection(population)

    
    # Hint: Call generate_children() on some individuals and fill up results.
    return results

def roulette_selection(population):
    selected  = []

    return selected

def tournament_selection(population):
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
    
    #MAY NOT NEED THE CODE RIGHT BELOW
    # if the population size was an odd number then the last player didn't get to compete
    # make them compete with a random participant
    # contestant_1 = population[-1]
    # contestant_2 = random.choices(population)
    # if (len(population) % 2) != 0:
    #     if contestant_1._fitness > contestant_2._fitness:
    #         winners.append(contestant_1)
    #     else:
    #         winners.append(contestant_2)

    return winners



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
    pop_limit = 16
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
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
        for ind in population:
            print("\nIndividual",count,": ")
            count += 1
            for i in ind.genome:
                print(listToString(i))
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    print("Generation:", str(generation))
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
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
    for ind in final_gen:
        print(ind)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
