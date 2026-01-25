import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, '..', 'docs', 'bm1Qd3Wa - energyg1.json')

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('-' * 40)
print('Inspecting Model PoC:')
for card in data['cards']:
    if 'Model PoC' in card.get('name'):
        print(f'Name: {card.get('name')}')
        print(f'Root Due: {card.get('due')}')
        print(f'Badges Due: {card.get('badges', {}).get('due')}')
        print(f'Start: {card.get('start')}')
        print(f'CustomFieldItems: {card.get('customFieldItems')}')
        break
