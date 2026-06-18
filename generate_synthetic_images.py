import json
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

contract_id = "b0260f5f7048eabd"
pages_dir = Path("logs/pages") / contract_id
pages_dir.mkdir(parents=True, exist_ok=True)

checkpoint = Path(f"logs/checkpoints/{contract_id}.json")
if not checkpoint.exists():
    print("Checkpoint not found!")
    exit(1)

state = json.loads(checkpoint.read_text())
clauses = state.get("clause_extraction", {}).get("clauses", [])

from src import config
from src.helpers.mask import mask_sensitive_text

def create_text_image(text, output_path):
    # Create a simple synthetic image with text
    width, height = 800, 300
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Simple text wrapping
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + " " + word) * 7 < width - 40:
            line += " " + word if line else word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    
    y = 20
    for l in lines:
        d.text((20, y), l, fill=(0, 0, 0))
        y += 20
        
    img.save(output_path)

count = 0
for c in clauses:
    raw_text = c.get("raw_text", "")
    if not raw_text:
        continue
        
    hash_text = raw_text
    if getattr(config, "ENABLE_SENSITIVE_MASKING", False) and getattr(config, "SENSITIVE_KEYWORDS", []):
        hash_text = mask_sensitive_text(raw_text, config.SENSITIVE_KEYWORDS)
        
    clause_hash = hashlib.md5(hash_text.strip().encode("utf-8")).hexdigest()
    crop_path = pages_dir / f"clause_{clause_hash}.png"
    
    create_text_image(raw_text, crop_path)
    count += 1

print(f"Successfully generated {count} synthetic clause images in {pages_dir}!")
