"""
Phase 5: Dataset Acquisition
==============================
Attempts programmatic download of TEDS-D 2022 and prints
manual instructions for CALDATA.

Nyx Dynamics LLC | MIT MicroMasters Capstone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

import io
import zipfile

out_dir = config.PROJECT_ROOT / "data" / "phase5" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# TEDS-D 2022 — direct download (no registration required)
# ══════════════════════════════════════════════════════════════════════

TEDS_URL = (
    "https://www.datafiles.samhsa.gov/sites/default/files/"
    "field-uploads-protected/studies/TEDS-D-2022-DS0001/"
    "TEDS-D-2022-DS0001-info/TEDS-D-2022-DS0001-data/files/"
    "TEDS-D-2022-DS0001-bndl-data-csv_v1.zip"
)

try:
    import requests
    print("Attempting TEDS-D 2022 download...")
    print(f"  URL: {TEDS_URL[:80]}...")
    r = requests.get(TEDS_URL, timeout=120)
    if r.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(out_dir)
            extracted = z.namelist()
            print(f"  TEDS-D extracted to {out_dir}")
            for name in extracted:
                print(f"    {name}")
    else:
        raise Exception(f"HTTP {r.status_code}")
except Exception as e:
    print(f"  Auto-download failed: {e}")
    print()
    print("MANUAL DOWNLOAD INSTRUCTIONS — TEDS-D 2022:")
    print("  1. Go to: https://www.samhsa.gov/data/data-we-collect/"
          "teds-treatment-episode-data-set/datafiles/teds-d-2022")
    print("  2. Click 'Download' next to 'ASCII / Delimited' format")
    print("  3. Unzip and save the .csv to:")
    print(f"     {out_dir}/teds_d_2022.csv")

# ══════════════════════════════════════════════════════════════════════
# CALDATA 1991-1993 — requires free ICPSR registration
# ══════════════════════════════════════════════════════════════════════

print()
print("MANUAL DOWNLOAD INSTRUCTIONS — CALDATA 1991-1993:")
print("  1. Go to: https://www.icpsr.umich.edu/web/NAHDAP/studies/2295")
print("  2. Create a free ICPSR account (takes ~5 minutes)")
print("  3. Click Download → select 'Delimited' format")
print("  4. Unzip and save the .tsv file to:")
print(f"     {out_dir}/caldata_1991_1993.tsv")
print()
print("  CALDATA contains N=1,826 clients with before/after abstinence,")
print("  treatment modality, stimulant/cocaine/meth substance codes,")
print("  income, employment, housing — directly maps to our equity pathways.")

# ══════════════════════════════════════════════════════════════════════
# Report what files are present
# ══════════════════════════════════════════════════════════════════════

print()
print("── Files in data/phase5/raw/ ──")
found_any = False
for f in sorted(out_dir.rglob("*")):
    if f.is_file():
        print(f"  Found: {f.relative_to(out_dir)}  ({f.stat().st_size / 1024:.1f} KB)")
        found_any = True
if not found_any:
    print("  (empty — no datasets downloaded yet)")
