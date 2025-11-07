import json

try:
    with open('task2_notebook/task2_improved.ipynb', 'r') as f:
        data = json.load(f)
    print('File is valid JSON')
    print('Number of cells:', len(data['cells']))
except Exception as e:
    print('File is not valid JSON')
    print('Error:', str(e))