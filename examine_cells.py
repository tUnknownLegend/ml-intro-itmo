import json

with open('task2_notebook/task2_improved.ipynb', 'r') as f:
    data = json.load(f)

print(f"Total number of cells: {len(data['cells'])}")

for i, cell in enumerate(data['cells']):
    print(f"\nCell {i}:")
    print(f"Cell type: {cell['cell_type']}")
    if 'execution_count' in cell:
        print(f"Execution count: {cell['execution_count']}")
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'brands_df = pd.read_parquet' in source:
            print(">>> This cell contains the brands data loading code!")
            print("Source:")
            print(source)
            print("="*50)
        else:
            # Print first few lines for context
            lines = source.split('\n')[:3]
            print("First few lines of source:")
            for line in lines:
                print(f"  {line}")
    elif cell['cell_type'] == 'markdown':
        # Print first few lines for context
        source = ''.join(cell['source'])
        lines = source.split('\n')[:3]
        print("First few lines of markdown:")
        for line in lines:
            print(f"  {line}")
