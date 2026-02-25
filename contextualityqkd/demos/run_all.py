"""Run all split demos serially."""

from __future__ import annotations

from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib


DEMO_MODULES = [
    "contextualityqkd.demos.randomness_qubit_z_x_xplusz",
    "contextualityqkd.demos.randomness_qubit_xplusz_xminusz",
    "contextualityqkd.demos.randomness_hexagon_povm",
    "contextualityqkd.demos.randomness_cabello_18ray",
    "contextualityqkd.demos.randomness_peres_24ray",
    "contextualityqkd.demos.qkd_qubit_z_x_xplusz",
    "contextualityqkd.demos.qkd_qubit_xplusz_xminusz",
    "contextualityqkd.demos.qkd_hexagon_povm",
    "contextualityqkd.demos.qkd_cabello_18ray",
    "contextualityqkd.demos.qkd_peres_24ray",
    "contextualityqkd.demos.qkd_porac_3_2",
]


def main() -> None:
    for module_name in DEMO_MODULES:
        module = importlib.import_module(module_name)
        print(f"\nRunning {module_name} ...")
        module.main()


if __name__ == "__main__":
    main()
