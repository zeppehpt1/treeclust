import random
import os

from dotenv import load_dotenv
from random import randint
from pathlib import Path

# get env variables
load_dotenv("config.env")
DATASETS_PATH = os.environ.get("DATASETS_PATH")
SITE = os.environ["SITE"]
NUMBER_OF_CLASSES = os.environ["NUMBER_OF_CLASSES"]
NUMBER_OF_CLASSES = int(NUMBER_OF_CLASSES)
EXPERIMENT_RUNS = os.environ.get("EXPERIMENT_RUNS")


def unique_rand(inicial: int, limit: int, total: int) -> list:
    data = []
    i = 0
    while i < total:
        number = randint(inicial, limit)
        if number not in data:
            data.append(number)
            i += 1
    return data


random.seed(698)
RANDOM_SEEDS = unique_rand(1, 999999, int(EXPERIMENT_RUNS))

# if container is used
docker_site_folder = "/treeclust/"
if os.path.exists(docker_site_folder):
    docker_site_folder = "/treeclust/data/" + SITE + "/"
    SITE_FOLDER = docker_site_folder
else:
    SITE_FOLDER = str(Path.home()) + DATASETS_PATH + "/" + SITE + "/"
