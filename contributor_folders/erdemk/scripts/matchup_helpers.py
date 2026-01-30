import os
import psutil
import logging
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr

def mem_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return None
        
def append_chunk_to_netcdf(ds_chunk, out_nc_path):
    """
    Append along obs. If file doesn't exist, create it.
    Requires ds_chunk has dims: obs, wavelength
    """
    if not os.path.exists(out_nc_path):
        ds_chunk.to_netcdf(out_nc_path, mode="w")
    else:
        ds_chunk.to_netcdf(out_nc_path, mode="a", append_dim="obs")

def load_done_obs(progress_csv):
    """
    Returns a set of obs_keys already completed.
    obs_key is string: f"{pace_filename}||{row_id}"
    """
    if not os.path.exists(progress_csv):
        return set()
    prev = pd.read_csv(progress_csv, usecols=["obs_key", "status"])
    return set(prev.loc[prev["status"] == "OBS_DONE", "obs_key"].dropna().unique())

def log_progress(rows, progress_csv):
    """
    rows: list[dict] appended to CSV
    """
    df = pd.DataFrame(rows)
    header = not os.path.exists(progress_csv)
    df.to_csv(progress_csv, mode="a", header=header, index=False)

def get_granule_groups(
    diagnostics_csv: str,
    *,
    status_ok: str = "Swath Covers Station",
    short_name: str = "PACE_OCI_L2_AOP",
    version_tag: str = "V3_1",
    s3_bucket: str = "ob-cumulus-prod-public",
    require_columns=("row_id", "time", "lat", "lon", "granule_id"),
    keep_columns=("row_id", "time", "lat", "lon", "granule_id", "station"),
):
    """
    Build (pace_filename, gdf) groups from a diagnostics CSV.

    Returns:
      granule_groups: list of (pace_filename, gdf)
      m: cleaned matchup table (rows with status == status_ok)
    """

    diag = pd.read_csv(diagnostics_csv)

    # Validate required columns
    missing_cols = [c for c in require_columns if c not in diag.columns]
    if missing_cols:
        raise ValueError(f"Diagnostics file missing required columns: {missing_cols}")

    # Filter to valid coverage rows
    m = diag[diag["status"] == status_ok].copy()

    # Keep only columns that exist
    cols = [c for c in keep_columns if c in m.columns]
    m = m[cols].copy()

    # Parse time (keep UTC)
    m["time"] = pd.to_datetime(m["time"], utc=True, errors="coerce")

    # Extract filename whether granule_id is filename OR str(DataGranule) blob
    # Example filename: PACE_OCI.20250117T194803.L2.OC_AOP.V3_1.nc
    pat = rf"(PACE_OCI\.\d{{8}}T\d{{6}}\.L2\.OC_AOP\.{re.escape(version_tag)}\.nc)"
    m["pace_filename"] = m["granule_id"].astype(str).str.extract(pat, expand=False)

    # Fail-fast if we can't extract names
    bad = m[m["pace_filename"].isna()]
    if len(bad) > 0:
        sample = bad[["row_id", "granule_id"]].head(3).to_dict("records")
        raise ValueError(
            f"Could not extract pace_filename for {len(bad)} rows. "
            f"Sample: {sample}"
        )

    # Build S3 URI (we open directly; no extra CMR search needed)
    m["s3_uri"] = "s3://" + s3_bucket + "/" + m["pace_filename"]

    # Group by swath file
    granule_groups = list(m.groupby("pace_filename", sort=True))

    return granule_groups, m



def get_file_logger(log_path: str, name: str = "rrs_extract"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't duplicate in notebook

    # avoid double handlers if you re-run cells
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_path).resolve())
               for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def run_rrs_extraction_resumable(
    granule_groups,
    out_nc,
    progress_csv,
    *,
    buffer_n=2,                 # write NetCDF every N valid spectra
    window=5,                   # single 5x5 window
    provider="OB_CLOUD",
    log_path="extract.log",
    clear_file_cache_each_granule=True,
    progress_flush_every=50,    # NEW: write progress CSV every N status rows
):
    """
    Resumable extraction with file logging and cache control.
    Assumes earthaccess.login() already called.

    Key change vs previous: progress CSV is written independently of NetCDF flush,
    so you can resume even if you crash before getting buffer_n valid spectra.
    """

    logger = get_file_logger(log_path)

    # xarray cache control (works with your xarray version)
    xr.set_options(file_cache_maxsize=1)

    # Ensure progress file exists immediately (so you can see it on disk)
    log_progress([{"obs_key": "RUN_START", "status": "RUN_START"}], progress_csv)

    done_obs = load_done_obs(progress_csv)
    logger.info(
        f"START | out_nc={out_nc} progress={progress_csv} done_obs={len(done_obs)} "
        f"buffer_n={buffer_n} window={window} progress_flush_every={progress_flush_every}"
    )

    mm = mem_mb()
    if mm is not None:
        logger.info(f"MEM_MB start: {mm:.1f}")

    wavelength_ref = None
    half = window // 2

    # buffers
    buf_times, buf_lats, buf_lons = [], [], []
    buf_rowid, buf_gran, buf_obskey = [], [], []
    buf_rrs_mean, buf_rrs_std = [], []

    # progress buffer (can include OBS_DONE + failure statuses)
    progress_buffer = []

    def flush_netcdf_if_needed(force=False):
        """Write buffered spectra to NetCDF if buffer full or forced."""
        if len(buf_times) == 0:
            return
        if (not force) and (len(buf_times) < buffer_n):
            return

        ds_chunk = xr.Dataset(
            data_vars={
                "Rrs_mean": (("obs", "wavelength"), np.stack(buf_rrs_mean, axis=0)),
                "Rrs_std":  (("obs", "wavelength"), np.stack(buf_rrs_std,  axis=0)),
            },
            coords={
                "time": ("obs", buf_times),
                "lat": ("obs", np.array(buf_lats, dtype=np.float32)),
                "lon": ("obs", np.array(buf_lons, dtype=np.float32)),
                "row_id": ("obs", np.array(buf_rowid, dtype=np.int64)),
                "granule_id": ("obs", np.array(buf_gran, dtype=object)),
                "obs_key": ("obs", np.array(buf_obskey, dtype=object)),
                "wavelength": ("wavelength", wavelength_ref),
            },
            attrs={"rrs_aggregation": f"{window}x{window}_mean_std"},
        )

        append_chunk_to_netcdf(ds_chunk, out_nc)

        logger.info(f"NETCDF_FLUSH | wrote_obs={len(buf_times)} total_done_obs={len(done_obs)}")
        mm2 = mem_mb()
        if mm2 is not None:
            logger.info(f"MEM_MB after netcdf flush: {mm2:.1f}")

        buf_times.clear(); buf_lats.clear(); buf_lons.clear()
        buf_rowid.clear(); buf_gran.clear(); buf_obskey.clear()
        buf_rrs_mean.clear(); buf_rrs_std.clear()

    def flush_progress_if_needed(force=False):
        """Write progress_buffer to CSV if big enough or forced."""
        nonlocal progress_buffer
        if len(progress_buffer) == 0:
            return
        if (not force) and (len(progress_buffer) < progress_flush_every):
            return
        log_progress(progress_buffer, progress_csv)
        progress_buffer.clear()

    # loop granules
    for pace_filename, gdf in tqdm(granule_groups, desc="granules"):
        s3_uri = gdf["s3_uri"].iloc[0]
        logger.info(f"GRANULE start | {pace_filename} | rows={len(gdf)} | mem_mb={mem_mb()}")

        remote = None
        dt = None

        # counters per granule
        n_skip_done = n_no_pix = n_no_spec = n_written = 0

        try:
            remote = earthaccess.open([s3_uri], provider=provider)[0]
            dt = xr.open_datatree(remote)

            lat2d = dt["navigation_data"]["latitude"]
            lon2d = dt["navigation_data"]["longitude"]
            Rrs   = dt["geophysical_data"]["Rrs"]
            wv    = dt["sensor_band_parameters"]["wavelength_3d"].values.astype(np.float32)

            if wavelength_ref is None:
                wavelength_ref = wv
                logger.info(f"WAVELENGTH set | n={len(wavelength_ref)}")
            else:
                if len(wv) != len(wavelength_ref) or np.nanmax(np.abs(wv - wavelength_ref)) > 1e-6:
                    logger.info(f"GRANULE skip wavelength mismatch | {pace_filename}")
                    progress_buffer.append({"obs_key": f"{pace_filename}||ALL", "status": "SKIP_WAVELENGTH_MISMATCH"})
                    flush_progress_if_needed(force=True)  # ensure it hits disk
                    continue

            for _, row in gdf.iterrows():
                row_id = int(row["row_id"])
                obs_key = f"{pace_filename}||{row_id}"

                if obs_key in done_obs:
                    n_skip_done += 1
                    continue

                lat0 = float(row["lat"])
                lon0 = float(row["lon"])
                t0   = pd.to_datetime(row["time"], utc=True)

                # nearest pixel (creates a big temporary when .values is called)
                dist2 = (lat2d - lat0)**2 + (lon2d - lon0)**2
                dist2 = dist2.where(lat2d.notnull() & lon2d.notnull())
                iy, ix = np.unravel_index(np.nanargmin(dist2.values), dist2.shape)

                patch = Rrs[:, iy-half:iy+half+1, ix-half:ix+half+1]
                if not np.isfinite(patch.values).any():
                    progress_buffer.append({"obs_key": obs_key, "status": f"NO_VALID_PIXELS_{window}x{window}"})
                    n_no_pix += 1
                    flush_progress_if_needed()  # NEW: progress can be persisted even without netcdf flush
                    continue

                patch = patch.where(patch.notnull())
                rrs_mean = patch.mean(dim=("number_of_lines", "pixels_per_line"), skipna=True).values
                rrs_std  = patch.std(dim=("number_of_lines", "pixels_per_line"), skipna=True).values

                if not np.isfinite(rrs_mean).any():
                    progress_buffer.append({"obs_key": obs_key, "status": "NO_VALID_SPECTRUM"})
                    n_no_spec += 1
                    flush_progress_if_needed()
                    continue

                # buffer valid spectra for NetCDF
                buf_times.append(t0.to_datetime64())
                buf_lats.append(lat0); buf_lons.append(lon0)
                buf_rowid.append(row_id)
                buf_gran.append(pace_filename)
                buf_obskey.append(obs_key)
                buf_rrs_mean.append(rrs_mean.astype(np.float32))
                buf_rrs_std.append(rrs_std.astype(np.float32))

                progress_buffer.append({"obs_key": obs_key, "status": "OBS_DONE"})
                done_obs.add(obs_key)
                n_written += 1

                # NEW: progress writes are independent of netcdf flush
                flush_progress_if_needed()

                # NetCDF flush only when enough valid spectra accumulated
                flush_netcdf_if_needed(force=False)

            # end-of-granule: force progress write even if no valid spectra
            flush_progress_if_needed(force=True)

            logger.info(
                f"GRANULE done | {pace_filename} | written={n_written} "
                f"skip_done={n_skip_done} no_pix={n_no_pix} no_spec={n_no_spec}"
            )

        except Exception as e:
            logger.exception(f"GRANULE error | {pace_filename} | {type(e).__name__}: {e}")

            # Force-write progress + any buffered spectra before raising
            flush_progress_if_needed(force=True)
            flush_netcdf_if_needed(force=True)
            raise

        finally:
            try:
                if dt is not None:
                    dt.close()
            except Exception:
                pass
            dt = None
            remote = None

            # Force-write any buffered spectra at granule boundary (optional but safer)
            flush_netcdf_if_needed(force=True)

            if clear_file_cache_each_granule:
                try:
                    xr.backends.file_manager.FILE_CACHE.clear()
                except Exception:
                    pass

    # final flushes
    flush_progress_if_needed(force=True)
    flush_netcdf_if_needed(force=True)

    logger.info("DONE")
    mm_end = mem_mb()
    if mm_end is not None:
        logger.info(f"MEM_MB end: {mm_end:.1f}")
