import json

with open("ref.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells in ref.ipynb: {len(cells)}")

for i, cell in enumerate(cells):
    cell_type = cell.get("cell_type", "")
    source = "".join(cell.get("source", []))
    print(f"\n{'='*30} CELL {i} ({cell_type}) {'='*30}")
    print(source)

