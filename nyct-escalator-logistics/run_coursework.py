"""One offline command: test and run the complete coursework experiment."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
for name in ('.deps', '.runtime'):
    path = ROOT / name
    if path.is_dir():
        sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

if __name__ == '__main__':
    import argparse
    import pytest
    from coursework import run
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Smoke run in a separate output folder')
    args = parser.parse_args()
    code = pytest.main(['-q', '-p', 'no:cacheprovider', str(ROOT / 'tests')])
    if code:
        sys.exit(code)
    run(ROOT, quick=args.quick)
