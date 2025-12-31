import os
import json
merges = 0
intersections = 0
for file in os.listdir('configs'):
    path = os.path.join('configs', file)
    with open(path, 'r') as f:
        obj = json.load(f)
    for instruction in obj['instructions']:
        if 'intersection' in instruction:
            intersections += 1
        else:
            merges += 1
print('Number of intersections: ', intersections)
print('Number of merges:', merges)