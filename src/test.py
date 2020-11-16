import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math
from ga import *

width = 200
height = 16

if __name__ == "__main__":
    # Individual = Individual_Grid


# grid stuff
    # Individual = Individual_Grid
    # random.seed(1)
    # ind = Individual.random_individual()
    # print("\nIndividual:")
    # for i in ind.genome:
    #     print(listToString(i))
    #
    # g = ind.genome
    # for row in range(height):
    #     for col in range(width):
    #         if g[row][col] == 'T':
    #             print("Location of pipe:", row, col)
    #
    # ind.mutate(ind.genome)
    # print("\nMutated:")
    # for i in ind.genome:
    #     print(listToString(i))

    Individual = Individual_DE
    random.seed(16)
    ind = Individual.random_individual()
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIndividual:")
    # for i in ind.genome:
    #     print(i)
    lev = ind.to_level()
    for i in lev:
        print(listToString(i))

    mut = Individual_DE(ind.mutate(ind.genome))
    print("\nmutated:")
    lev = mut.to_level()
    for i in lev:
        print(listToString(i))

    # print("stairs of mut:")
    # print(mut.get_stairs())
    # print("pipes of mut:")
    # print(mut.get_pipes())