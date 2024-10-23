import json
import random

def generate_hard():
    numbers = []
    for _ in range(80):
        num = max(80, min(100, random.gauss(90.5, 12.87)))
        numbers.append(round(num, 2))
    
    for _ in range(5):
        num = max(40, min(80, random.gauss(62.22, 7.12)))  
        numbers.append(round(num, 2))
    
    for _ in range(5):
        num = max(15, min(40, random.gauss(30, 5.9))) 
        numbers.append(round(num, 2))
    
    for _ in range(5):
        num = max(5, min(15, random.gauss(10, 2)))
        numbers.append(round(num, 2))
    
    for _ in range(5):
        num = max(1, min(5, random.gauss(3.4, 1)))
        numbers.append(round(num, 2))

    random.shuffle(numbers)

    return numbers


def generate_easy():
     return [20]*8

def generate_medium():
     return [100]*17


with open('requests/Group 6/hard.json', 'w') as json_file:
    json.dump({'requests':generate_hard()}, json_file, indent=4)

with open('requests/Group 6/easy.json', 'w') as json_file:
    json.dump({'requests':generate_easy()}, json_file, indent=4)

with open('requests/Group 6/medium.json', 'w') as json_file:
    json.dump({'requests':generate_medium()}, json_file, indent=4)


