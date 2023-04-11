import random
from random import randint
from typing import final

def unique_rand(inicial, limit, total):
        data = []
        i = 0
        while i < total:
            number = randint(inicial, limit)
            if number not in data:
                data.append(number)
                i += 1
        return data

random.seed(698)
RANDOM_SEEDS = unique_rand(1, 999999, 5)
SITE = 'Schiefer'