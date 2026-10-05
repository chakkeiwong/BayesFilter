#!/usr/bin/env python3
"""Repaired entry point for the historical, unimplemented 10D draft.

Uses the shared complete model/score implementation and new-scope diagnostic
tuning. Historical warm-start results supplied no oracle or valid score path.
Arguments follow run_sqmc_tuning.py; explicit arguments override these presets.
"""
import sys
from run_sqmc_tuning import main as tuning_main

def main():
    return tuning_main(['--family','diagonal_ar','--dimension','10','--horizon','120',
                        '--particles','1000',*sys.argv[1:]])

if __name__ == '__main__':
    raise SystemExit(main())
