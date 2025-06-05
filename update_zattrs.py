#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(
        description="Update .zattrs multiscale levels by upsampling/downsampling from a reference level."
    )
    parser.add_argument(
        "--upsample-factor",
        type=float,
        required=True,
        help="Upsample factor between levels (e.g., 2 for 2x upsampling per level)."
    )
    parser.add_argument(
        "zattrs_path",
        type=Path,
        help="Path to the .zattrs JSON file."
    )
    parser.add_argument(
        "ome_zarr_dir",
        type=Path,
        nargs="?",
        default=None,
        help="(Optional) Path to the ome-zarr directory. If not given, will use the directory containing the .zattrs."
    )
    return parser.parse_args()

def read_zattrs(zattrs_path: Path) -> Dict[str, Any]:
    with open(zattrs_path, 'r') as f:
        return json.load(f)

def write_zattrs(zattrs_path: Path, zattrs: Dict[str, Any]):
    with open(zattrs_path, 'w') as f:
        json.dump(zattrs, f, indent=4)

def get_reference_level(datasets: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """
    Returns the reference level (int) and the dataset entry.
    Assumes 'path' is a string representing an integer.
    """
    assert datasets, "No datasets found in multiscales."
    # Pick the first dataset as reference (by convention of example)
    ref = datasets[0]
    ref_level = int(ref["path"])
    return ref_level, ref

def get_existing_levels(ome_zarr_dir: Path) -> List[int]:
    """
    Returns a sorted list of integers for each group/array in the ome-zarr directory that is an integer name.
    """
    levels = []
    for entry in os.listdir(ome_zarr_dir):
        try:
            lvl = int(entry)
            if (ome_zarr_dir / entry).is_dir():
                levels.append(lvl)
        except ValueError:
            continue
    return sorted(levels)

def update_scale_and_translation(ref_scale: List[float], ref_translation: List[float], ref_level: int, target_level: int, upsample_factor: float) -> Tuple[List[float], List[float]]:
    """
    Calculate new scale and translation for a given target_level using upsample_factor relative to reference.
    """
    factor = upsample_factor ** (ref_level - target_level)
    new_scale = [v / factor for v in ref_scale]
    new_translation = [v / factor for v in ref_translation]
    return new_scale, new_translation

def build_new_datasets(
    levels: List[int],
    ref_level: int,
    ref_dataset: Dict[str, Any],
    upsample_factor: float
) -> List[Dict[str, Any]]:
    """
    Build new datasets list with coordinateTransformations updated for each level.
    """
    ref_scale = None
    ref_translation = None
    for trans in ref_dataset["coordinateTransformations"]:
        if trans["type"] == "scale":
            ref_scale = trans["scale"]
        elif trans["type"] == "translation":
            ref_translation = trans["translation"]
    assert ref_scale is not None, "No scale transformation in reference dataset"
    assert ref_translation is not None, "No translation transformation in reference dataset"
    datasets = []
    for lvl in levels:
        scale, translation = update_scale_and_translation(ref_scale, ref_translation, ref_level, lvl, upsample_factor)
        datasets.append({
            "path": str(lvl),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": scale
                },
                {
                    "type": "translation",
                    "translation": translation
                }
            ]
        })
    return datasets

def main():
    args = parse_args()

    zattrs_path = args.zattrs_path
    ome_zarr_dir = args.ome_zarr_dir or zattrs_path.parent

    zattrs = read_zattrs(zattrs_path)
    multiscale = zattrs["multiscales"][0]
    datasets = multiscale["datasets"]
    ref_level, ref_dataset = get_reference_level(datasets)

    levels = get_existing_levels(ome_zarr_dir)
    if not levels:
        raise RuntimeError(f"No integer-named directories found in {ome_zarr_dir}")

    # Build new datasets
    new_datasets = build_new_datasets(
        levels=levels,
        ref_level=ref_level,
        ref_dataset=ref_dataset,
        upsample_factor=args.upsample_factor
    )

    # Overwrite datasets in zattrs
    zattrs["multiscales"][0]["datasets"] = new_datasets

    # Write back to .zattrs (or print, depending on design)
    write_zattrs(zattrs_path, zattrs)
    print(f"Updated .zattrs written to {zattrs_path}")

if __name__ == "__main__":
    main()
