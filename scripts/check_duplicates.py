import json
import os
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, '..', 'docs', 'gantt', 'bm1Qd3Wa - energyg1.json')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

names = [card['name'] for card in data['cards'] if not card.get('closed')]
counts = Counter(names)

dups = {k:v for k,v in counts.items() if v > 1}

print('Duplicates found:')
if not dups: print('None')
for name, count in dups.items():
    print(f'{count}x {name}')
