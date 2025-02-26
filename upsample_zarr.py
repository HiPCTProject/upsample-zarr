from pathlib import Path
import click
import joblib
import zarr
import numpy as np


@joblib.delayed
def _copy(
    arr_in: zarr.Array,
    arr_out: zarr.Array,
    x: int,
    y: int,
    z: int,
    upsample_factor: int,
) -> None:
    source = arr_in[
        x : x + arr_in.chunks[0], y : y + arr_in.chunks[1], z : z + arr_in.chunks[2]
    ]
    if np.all(source == arr_in.fill_value):
        # Check if source is all fill value and don't bother copying if so
        return

    arr_out[
        x * upsample_factor : (x + arr_in.chunks[0]) * upsample_factor,
        y * upsample_factor : (y + arr_in.chunks[1]) * upsample_factor,
        z * upsample_factor : (z + arr_in.chunks[2]) * upsample_factor,
    ] = (
        np.array(source)
        .repeat(upsample_factor, axis=0)
        .repeat(upsample_factor, axis=1)
        .repeat(upsample_factor, axis=2)
    )


@click.command()
@click.argument(
    "input_path",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
)
@click.argument(
    "output_path",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
)
@click.option("--upsample-factor", type=int, required=True)
@click.option("--n-jobs", type=int, required=False, default=-1)
def main(input_path: str, output_path: str, upsample_factor: int, n_jobs: int):
    if upsample_factor <= 1:
        raise ValueError(f"--upsample-factor must be >= 2 (got {upsample_factor})")
    print("👋 Hello from upsample-zarr!")
    print()
    print(f"Upsampling zarr at  {input_path}")
    print(f"Saving to           {output_path}")
    print(f"Upsampling by       {upsample_factor}")
    print()
    print("Loading input Zarr...")
    arr_in = zarr.open_array(input_path, mode="r")
    print(f"shape:         {arr_in.shape}")
    print("🎉 Loaded")
    print()

    if arr_in.ndim != 3:
        raise RuntimeError("Only 3D arrays supported")

    shape_out = tuple(d * upsample_factor for d in arr_in.shape)
    dtype_out = arr_in.dtype
    chunks_out = arr_in.chunks
    compressor_out = arr_in.compressor
    fill_out = arr_in.fill_value

    print("Setting up output Zarr store...")
    print(f"shape:         {shape_out}")
    print(f"chunks:        {chunks_out}")
    print(f"dtype:         {dtype_out}")
    print(f"fill value:    {fill_out}")
    print(f"compressor:    {compressor_out}")
    store_out = zarr.DirectoryStore(output_path, dimension_separator="/")
    arr_out = zarr.open_array(
        store_out,
        shape=shape_out,
        chunks=chunks_out,
        dtype=dtype_out,
        compressor=compressor_out,
        write_empty_chunks=False,
        fill_value=fill_out,
        mode="w",
    )
    print("🎉 Set up Zarr Store")
    print("")
    print("Setting up upsampling jobs")
    jobs = []
    for x in range(0, arr_in.shape[0], arr_in.chunks[0]):
        for y in range(0, arr_in.shape[1], arr_in.chunks[1]):
            for z in range(0, arr_in.shape[2], arr_in.chunks[2]):
                jobs.append(
                    _copy(arr_in, arr_out, x, y, z, upsample_factor=upsample_factor)
                )

    print(f"🎉 Set up {len(jobs)} upsampling jobs")
    print()
    print("Executing jobs...")
    joblib.Parallel(verbose=10, n_jobs=n_jobs)(jobs)
    print("🎉 Finished executing jobs")
    print()
    print("Congratulations, your array has been upsampled.")
    print("We hope you enjoyed using upsample-zarr")
    print("Please consider leavning us a 5⭐️ review!")


if __name__ == "__main__":
    main()
