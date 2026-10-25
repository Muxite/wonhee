"""Compatibility entry point for the direct Moeka/Wonhee/Stagecraft installer.

Dry run by default; --apply snapshots private state before installing. The former
Wonny peer workflow is retired. Main Moeka owns the conversation and tools.
"""
from pathlib import Path
import runpy


def main():
    installer = Path(__file__).resolve().parents[2] / 'stagecraft/deploy/install.py'
    if not installer.is_file():
        raise SystemExit('Stagecraft installer missing; check out sibling stagecraft first.')
    runpy.run_path(str(installer), run_name='__main__')


if __name__ == '__main__':
    main()
