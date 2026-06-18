import json
import hashlib
from pathlib import Path

contract_id = "b0260f5f7048eabd"
pages_dir = Path("logs/pages") / contract_id
print("pages_dir exists:", pages_dir.exists())

checkpoint = Path(f"logs/checkpoints/{contract_id}.json")
if checkpoint.exists():
    state = json.loads(checkpoint.read_text())
    clauses = state.get("clause_extraction", {}).get("clauses", [])
    print(f"Found {len(clauses)} clauses in checkpoint.")
    for c in clauses[:2]:
        raw_text = c.get("raw_text", "")
        clause_hash = hashlib.md5(raw_text.strip().encode("utf-8")).hexdigest()
        crop_path = pages_dir / f"clause_{clause_hash}.png"
        print(f"Clause hash: {clause_hash}")
        print(f"Crop path exists: {crop_path.exists()}")
