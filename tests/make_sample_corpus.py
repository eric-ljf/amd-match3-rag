#!/usr/bin/env python3
"""Generate a tiny synthetic corpus exercising every ingest path."""
import csv
import os

from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_sample")
os.makedirs(OUT, exist_ok=True)

# 1. plain text with a distinctive fact
with open(os.path.join(OUT, "notes.txt"), "w") as f:
    f.write(
        "Operations runbook.\n"
        "The backup server hostname is bkup-07.internal.\n"
        "Nightly snapshots run at 02:00 local time.\n"
        "Contact the on-call engineer if a snapshot fails twice in a row.\n"
    )

# 2. csv with a distinctive fact
with open(os.path.join(OUT, "budget.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["quarter", "category", "spend_usd"])
    w.writerow(["Q3", "marketing", "45210"])
    w.writerow(["Q3", "infra", "88120"])

# 3. xlsx with two sheets; the fact lives on sheet 2
wb = Workbook()
ws1 = wb.active
ws1.title = "Summary"
ws1.append(["metric", "value"])
ws1.append(["uptime_pct", "99.95"])
ws2 = wb.create_sheet("Inventory")
ws2.append(["item", "stock_units"])
ws2.append(["Widget X", "1337"])
wb.save(os.path.join(OUT, "report.xlsx"))

# 4. image with printed text (OCR path)
img = Image.new("RGB", (700, 120), "white")
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=28)
d.text((20, 40), "Maintenance window: Sunday 03:00-05:00 UTC", fill="black", font=font)
img.save(os.path.join(OUT, "sign.png"))

# 5. empty file -> must be skipped silently
open(os.path.join(OUT, "empty.txt"), "w").close()

# 6. unknown extension with text -> utf-8 fallback path
with open(os.path.join(OUT, "mystery.zzz"), "w") as f:
    f.write("The fallback parser handles unknown extensions as plain text.\n")

# 7. binary junk with unknown extension -> must be skipped
with open(os.path.join(OUT, "junk.bin"), "wb") as f:
    f.write(b"\x00\x01\x02\xff\xfe binary junk \x00\x00")

print(f"sample corpus written to {OUT}")
