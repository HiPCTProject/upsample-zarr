# upsample-zarr

A simple command line tool to upsample a Zarr array.

- Supports upsampling by an integer factor >= 2
- Just copies data, no fancy smoothing
- Uses the input Zarr array configuration (chunks, codec, data type) for the output Zarr array
- Always uses a dimension separator of `'/'` for the output array
- Never writes empty chunks
- Writes each chunk of the output Zarr array in parallel

If you want to downsample a lot (e.g., by 32), it's recommended to do this in stages.

## Using

1. Install `uv`
2. Clone this repository
3. In the root directory of this repository, run `uv sync`

example run:

```sh
uv run python upsample_zarr.py --upsample-factor 2 data/30.88um_hippocampus_labels.zarr data/15.44um_hippocampus_labels.zarr
```

This upsamples by a factor of 2, from an output Zarr directory on the left to an output Zarr directory on the right.
