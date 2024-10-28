import json
import random


def generate_hard():
    numbers = []

    for _ in range(80):
        num = random.gauss(90.82, 3.5)
        while num < 80 or num > 100:
            num = random.gauss(90.82, 3.5)
        numbers.append(round(num, 2))

    for _ in range(5):
        num = random.gauss(61, 12.73)
        while num < 40 or num > 80:
            num = random.gauss(61, 12.73)
        numbers.append(round(num, 2))

    for _ in range(5):
        num = random.gauss(33.43, 7.82)
        while num < 15 or num > 40:
            num = random.gauss(33.43, 7.82)
        numbers.append(round(num, 2))

    for _ in range(10):
        num = random.gauss(10.78, 2.12)
        while num < 10 or num > 15:
            num = random.gauss(10.78, 2.12)
        numbers.append(round(num, 2))
    random.shuffle(numbers)

    return numbers


def generate_easy():
    return [20] * 8


def generate_medium():
    return [100] * 17


with open("requests/group_6/hard.json", "w") as json_file:
    json.dump({"requests": generate_hard()}, json_file, indent=4)

with open("requests/group_6/easy.json", "w") as json_file:
    json.dump({"requests": generate_easy()}, json_file, indent=4)

with open("requests/group_6/medium.json", "w") as json_file:
    json.dump({"requests": generate_medium()}, json_file, indent=4)
