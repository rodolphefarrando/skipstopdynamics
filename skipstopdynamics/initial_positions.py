import random
import numpy as np


def initial_position(n, m, I, style, model):
    """
    This function compute the inital positions of the trains. There are three possibilities:
    - Linear: trains are equally distributed along the line
    - Random: the distribution is fully random
    - Full stop: One train stops at all stations

    :return: b_0 and bbar_0
    """

    possibilities = ['linear', 'random']
    b = np.zeros(n, dtype=int)

    if style not in possibilities:
        print('The intial position style does not exist; linear one is affected')
        style = 'linear'

    if style == 'linear':
        b = linear(n, m, b)
    else:
        b = randominit(n, m, b, I, model)

    bbar = 1 - b

    return b, bbar


def linear(n, m, b):

    space = n / m
    for i in range(m):
        b[int(np.floor(i * space))] = 1

    return b


def randominit(n, m, b, I, model):
    while sum(b) != m:
        value = random.randint(0, n - 2)
        if model != 'first':
            b[value] = 1
        else:
            if (value in I or value - 1 in I) and m < 132:
                b[value] = 0
            else:
                b[value] = 1
    return b