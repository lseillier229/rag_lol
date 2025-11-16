import json
from pathlib import Path

base = Path("data/champions")

champ_files = list(base.glob("*.json"))
print("Champions trouvés :", [f.stem for f in champ_files])

for path in champ_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=== {data['name']} ===")
    print("Région :", data["region"])
    print("Rôles  :", ", ".join(data["roles"]))
    print("Nb sorts :", len(data["abilities"]))
