import random
from random import randint

def unique_rand(inicial:int, limit:int, total:int) -> list:
        data = []
        i = 0
        while i < total:
            number = randint(inicial, limit)
            if number not in data:
                data.append(number)
                i += 1
        return data

random.seed(698)
RANDOM_SEEDS = unique_rand(1, 999999, 5) # 5 = how often the experiment should be performed
SITE = 'Schiefer'
#SITE = 'Bamberg_Stadtwald'
#SITE = 'Tretzendorf'
NUMBER_OF_CLASSES = 10 # schiefer
#NUMBER_OF_CLASSES = 9 # stadtwald 13 gt classes
#NUMBER_OF_CLASSES = 8 # Tretzendorf 9 gt classes

# NUMBER represents the number of appearing classes after the extraction of the single tree images