"""Turn model.tflite into esp32_embedder/model_data.h  (run: python make_header.py model.tflite)"""
import sys
from pathlib import Path
src = Path(sys.argv[1] if len(sys.argv) > 1 else "model.tflite")
data = src.read_bytes()
out = Path("esp32_embedder/model_data.h")
out.parent.mkdir(exist_ok=True)
lines = [f"// generated from {src.name}", "alignas(16) const unsigned char model_data[] = {"]
for i in range(0, len(data), 16):
    lines.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",")
lines += ["};", f"const unsigned int model_data_len = {len(data)};"]
out.write_text("\n".join(lines))
print(f"wrote {out} ({len(data)} bytes of model)")
