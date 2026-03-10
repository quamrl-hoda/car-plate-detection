"""
fix_env.py  — Run this once to fix all NumPy 2.x compatibility errors.

Usage:
    uv run python fix_env.py

What it fixes:
  1. numpy 2.2.6 → 1.26.4    (fixes pyarrow + torch DLL errors)
  2. pyarrow reinstall         (fixes AttributeError: _ARRAY_API not found)
  3. torch reinstall           (fixes WinError 1114 c10.dll)
"""
import subprocess, sys

def run(args, desc):
    print(f"\n{'='*55}")
    print(f"  {desc}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, "-m", "pip"] + args)
    if result.returncode == 0:
        print(f"  ✓ Done")
    else:
        print(f"  ✗ Failed (exit {result.returncode})")
    return result.returncode

steps = [
    # 1. Pin numpy to last 1.x release that everything was built against
    (["install", "numpy==1.26.4"],
     "Step 1/3 — Downgrade numpy to 1.26.4"),

    # 2. Reinstall pyarrow so it re-links against numpy 1.26.4
    (["install", "--force-reinstall", "--no-deps", "pyarrow"],
     "Step 2/3 — Reinstall pyarrow"),

    # 3. Reinstall torch so c10.dll re-links cleanly
    (["install", "--force-reinstall", "--no-deps", "torch"],
     "Step 3/3 — Reinstall torch"),
]

failed = []
for args, desc in steps:
    code = run(args, desc)
    if code != 0:
        failed.append(desc)

print("\n" + "="*55)
if not failed:
    print("  ALL STEPS PASSED — now run:  uv run app.py")
else:
    print("  Some steps failed:")
    for f in failed:
        print(f"    ✗ {f}")
print("="*55)
