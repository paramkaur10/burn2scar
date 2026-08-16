"""Sentinel-2 acquisition, masking, and export — no sensor simulation."""
import datetime as dt
import logging
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from eolearn.core import EOTask, FeatureType
from sentinelhub import (
    BBox, DataCollection, SentinelHubCatalog, SentinelHubInputTask, SHConfig,
)
from sentinelhub.geo_utils import get_utm_crs, wgs84_to_utm

log = logging.getLogger(__name__)

MASK_CLEAR, MASK_FRESH, MASK_OLD = 0, 2, 3
MASK_CLOUD, MASK_SHADOW, MASK_WATER, MASK_NODATA = 4, 5, 6, 255
MASK_COLORMAP = {
    0:   (200, 200, 200, 255),
    2:   (220,  60,  20, 255),
    3:   (150,  40,  20, 255),
    4:   (255, 255, 255, 255),
    5:   ( 60,  60,  90, 255),
    6:   ( 30,  90, 200, 255),
    255: (  0,   0,   0,   0),
}

EXPORT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B10", "B11", "B12"]
EXPORT_AS_UINT16 = True
IDX_B03 = EXPORT_BANDS.index("B03")
IDX_B04 = EXPORT_BANDS.index("B04")
IDX_B08 = EXPORT_BANDS.index("B08")

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                  "/protocol/openid-connect/token")

S2L1C_COLLECTION = DataCollection.SENTINEL2_L1C.define_from(
    "SENTINEL2_L1C_CDSE", service_url=CDSE_BASE_URL)
S2L2A_COLLECTION = DataCollection.SENTINEL2_L2A.define_from(
    "SENTINEL2_L2A_CDSE", service_url=CDSE_BASE_URL)


def make_sh_config(sh_client_id, sh_client_secret):
    sh = SHConfig()
    sh.sh_client_id, sh.sh_client_secret = sh_client_id, sh_client_secret
    sh.sh_base_url = CDSE_BASE_URL
    from sentinelhub.download import session as shs
    def _collect_new_token(self):
        return self._fetch_token(shs.DownloadRequest(url=CDSE_TOKEN_URL))
    shs.SentinelHubSession._collect_new_token = _collect_new_token
    return sh


def get_utm_bbox(lat_centre, lon_centre, bbox_size_m):
    east, north = wgs84_to_utm(lon_centre, lat_centre)
    east, north = 10 * int(east / 10), 10 * int(north / 10)
    return BBox(((east - bbox_size_m // 2, north - bbox_size_m // 2),
                 (east + bbox_size_m // 2, north + bbox_size_m // 2)),
                crs=get_utm_crs(lon_centre, lat_centre))


class WaterMaskTask(EOTask):
    def __init__(self, ndwi_threshold=0.05, use_scl_water=True):
        self.thr, self.use_scl = ndwi_threshold, use_scl_water

    def execute(self, eopatch):
        bands = eopatch[(FeatureType.DATA, "BANDS")]
        green, nir = bands[..., IDX_B03], bands[..., IDX_B08]
        denom = np.maximum(green + nir, 1e-6)
        water = ((green - nir) / denom) > self.thr
        if self.use_scl and "SCL" in eopatch.mask:
            water = water | (eopatch.mask["SCL"][..., 0] == 6)
        eopatch[(FeatureType.MASK, "WATER")] = water[..., np.newaxis].astype(np.uint8)
        return eopatch


class SCLCloudTask(EOTask):
    def __init__(self, scl_feature=(FeatureType.MASK, "SCL")):
        self.scl_feature = scl_feature

    def execute(self, eopatch):
        scl = eopatch[self.scl_feature][..., 0]
        eopatch[(FeatureType.MASK, "SCL_CLOUD")] = (
            (scl == 8) | (scl == 9))[..., np.newaxis].astype(np.uint8)
        eopatch[(FeatureType.MASK, "SCL_CLOUD_SHADOW")] = (
            scl == 3)[..., np.newaxis].astype(np.uint8)
        eopatch[(FeatureType.MASK, "SCL_CIRRUS")] = (
            scl == 10)[..., np.newaxis].astype(np.uint8)
        return eopatch


class OmniCloudMaskTask(EOTask):
    def __init__(self, inference_device=None, batch_size=1):
        self.inference_device, self.batch_size = inference_device, batch_size

    def execute(self, eopatch):
        from omnicloudmask import predict_from_array
        bands = eopatch[(FeatureType.DATA, "BANDS")]
        t, h, w, _ = bands.shape
        cloud = np.zeros((t, h, w, 1), np.uint8)
        shadow = np.zeros((t, h, w, 1), np.uint8)
        for ti in range(t):
            arr = np.stack([bands[ti, ..., IDX_B04], bands[ti, ..., IDX_B03],
                            bands[ti, ..., IDX_B08]], axis=0).astype(np.float32) * 10000.0
            kwargs = dict(batch_size=self.batch_size, apply_no_data_mask=True, no_data_value=0)
            if self.inference_device:
                kwargs["inference_device"] = self.inference_device
            pred = np.asarray(predict_from_array(arr, **kwargs))[0]
            cloud[ti, ..., 0] = ((pred == 1) | (pred == 2)).astype(np.uint8)
            shadow[ti, ..., 0] = (pred == 3).astype(np.uint8)
        eopatch[(FeatureType.MASK, "OCM_CLOUD")] = cloud
        eopatch[(FeatureType.MASK, "OCM_SHADOW")] = shadow
        return eopatch


class BurnScarRasterTask(EOTask):
    def __init__(self, fires_gdf, reference_feature, output_feature=(FeatureType.MASK, "BURN"),
                 min_age_days=-1, max_age_days=365):
        self.fires, self.ref, self.out = fires_gdf, reference_feature, output_feature
        self.min_age, self.max_age = min_age_days, max_age_days

    def execute(self, eopatch):
        t, h, w = eopatch[self.ref].shape[:3]
        bbox = eopatch.bbox
        transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, w, h)
        fires_utm = self.fires.to_crs(bbox.crs.pyproj_crs())
        burn = np.zeros((t, h, w, 1), np.uint8)
        for ti, ts in enumerate(eopatch.timestamp):
            ts64 = np.datetime64(ts)
            age_days = (ts64 - fires_utm["fire_date"].values) / np.timedelta64(1, "D")
            sel = fires_utm[(age_days >= 0) & (age_days > self.min_age) & (age_days <= self.max_age)]
            if len(sel):
                burn[ti, ..., 0] = rasterize(((g, 1) for g in sel.geometry),
                                             out_shape=(h, w), transform=transform,
                                             fill=0, dtype=np.uint8)
        eopatch[self.out] = burn
        return eopatch


class CombineMaskTask(EOTask):
    def __init__(self, include_cirrus=False):
        self.include_cirrus = include_cirrus

    def execute(self, eopatch):
        m = eopatch.mask
        cloud = m["OCM_CLOUD"][..., 0].astype(bool)
        if self.include_cirrus and "SCL_CIRRUS" in m:
            cloud |= m["SCL_CIRRUS"][..., 0].astype(bool)
        shadow = m["OCM_SHADOW"][..., 0].astype(bool)
        water  = m["WATER"][..., 0].astype(bool)
        fresh  = m["BURN_FRESH"][..., 0].astype(bool)
        old    = m["BURN_OLD"][..., 0].astype(bool)
        valid  = m["dataMask"][..., 0].astype(bool)

        labels = np.full(cloud.shape, MASK_CLEAR, np.uint8)
        labels[water]  = MASK_WATER
        labels[old]    = MASK_OLD
        labels[fresh]  = MASK_FRESH
        labels[shadow] = MASK_SHADOW
        labels[cloud]  = MASK_CLOUD
        labels[~valid] = MASK_NODATA
        eopatch[(FeatureType.MASK, "LABELS")] = labels[..., np.newaxis]
        return eopatch


def _profile(bbox, h, w, count, dtype, nodata=None):
    return dict(driver="GTiff", height=h, width=w, count=count, dtype=dtype,
                crs=bbox.crs.pyproj_crs(), compress="deflate", nodata=nodata,
                transform=from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, w, h))


def _write_bands(path, data_hwc, bbox):
    h, w, c = data_hwc.shape
    if EXPORT_AS_UINT16:
        arr = np.clip(np.round(data_hwc * 10000.0), 0, 65535).astype(np.uint16)
        prof = _profile(bbox, h, w, c, "uint16", nodata=0)
    else:
        arr = data_hwc.astype(np.float32)
        prof = _profile(bbox, h, w, c, "float32")
    prof.update(predictor=2, tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **prof) as dst:
        for b in range(c):
            dst.write(arr[..., b], b + 1)
            dst.set_band_description(b + 1, EXPORT_BANDS[b])


def _write_mask(path, mask_hw, bbox):
    h, w = mask_hw.shape
    prof = _profile(bbox, h, w, 1, "uint8", nodata=MASK_NODATA)
    prof.update(tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write_colormap(1, MASK_COLORMAP)
        dst.write(mask_hw.astype(np.uint8), 1)


def export_products(eopatch, out_dir, fire_id):
    import os
    written, bbox = [], eopatch.bbox
    bands, labels = eopatch.data["BANDS"], eopatch.mask["LABELS"]
    for ti, ts in enumerate(eopatch.timestamp):
        ts_str = ts.strftime("%Y-%m-%dT%H-%M-%S")
        scene_dir = os.path.join(out_dir, "acquisitions", fire_id, ts_str)
        os.makedirs(scene_dir, exist_ok=True)
        ip = os.path.join(scene_dir, f"{fire_id}_{ts_str}_s2.tif")
        mp = os.path.join(scene_dir, f"{fire_id}_{ts_str}_mask.tif")
        _write_bands(ip, bands[ti], bbox)
        _write_mask(mp, labels[ti, ..., 0], bbox)
        written += [ip, mp]
    return written


def get_candidate_dates(bbox, fire_date, cfg, sh, max_retries=3, retry_delay=15):
    t0 = (fire_date + dt.timedelta(days=cfg["post_fire_min_days"])).date().isoformat()
    t1 = (fire_date + dt.timedelta(days=cfg["post_fire_max_days"])).date().isoformat()
    for attempt in range(1, max_retries + 1):
        try:
            results = list(SentinelHubCatalog(sh).search(
                collection=S2L2A_COLLECTION, bbox=bbox, time=(t0, t1),
                filter=f"eo:cloud_cover < {cfg['maxcc'] * 100:.0f}",
                fields={"include": ["properties.datetime", "properties.eo:cloud_cover"],
                        "exclude": []}))
            results.sort(key=lambda r: r["properties"]["eo:cloud_cover"])
            return [(r["properties"]["datetime"].split("T")[0], r["properties"]["eo:cloud_cover"])
                   for r in results[:cfg["max_candidate_dates"]]]
        except Exception as exc:
            msg = str(exc)
            transient = any(s in msg for s in
                            ("NameResolutionError", "ConnectionError", "Max retries exceeded"))
            if transient and attempt < max_retries:
                log.warning("Catalog search failed (attempt %d/%d, retrying in %ds): %s",
                           attempt, max_retries, retry_delay, msg.splitlines()[0])
                time.sleep(retry_delay)
                continue
            log.warning("Catalog search failed: %s", msg.splitlines()[0])
            return []
    return []


def process_fire(fire_row, fires_gdf, cfg, sh):
    fire_id = fire_row["fire_id"]
    centroid = fire_row.geometry.centroid
    bbox = get_utm_bbox(centroid.y, centroid.x, cfg["bbox_size_m"])

    candidates = get_candidate_dates(bbox, fire_row["fire_date"], cfg, sh)
    if not candidates:
        log.info("[%s] no S2 acquisitions found - skipped", fire_id); return []

    aux = {"processing": {"upsampling": "BICUBIC"}}
    accepted_eop = accepted_date = None
    fallback_eop = fallback_date = None
    fallback_frac = None

    for date, cc in candidates:
        eop = SentinelHubInputTask(
            data_collection=S2L2A_COLLECTION, resolution=cfg["s2_resolution_m"],
            additional_data=[(FeatureType.MASK, "SCL")], maxcc=cfg["maxcc"],
            aux_request_args=aux, config=sh, cache_folder=cfg["cache_folder"],
            time_difference=dt.timedelta(minutes=180),
        )(bbox=bbox, time_interval=(date, date))
        if len(eop.timestamp) == 0:
            continue
        eop = SentinelHubInputTask(
            data_collection=S2L1C_COLLECTION, resolution=cfg["s2_resolution_m"],
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=[(FeatureType.MASK, "dataMask")],
            bands=EXPORT_BANDS, aux_request_args=aux, config=sh,
            cache_folder=cfg["cache_folder"], time_difference=dt.timedelta(minutes=180),
        )(eopatch=eop)

        eop = WaterMaskTask(cfg["ndwi_threshold"], cfg["use_scl_water"])(eop)
        eop = SCLCloudTask(scl_feature=(FeatureType.MASK, "SCL"))(eop)
        eop = OmniCloudMaskTask(inference_device=cfg.get("ocm_device"))(eop)
        eop = BurnScarRasterTask(fires_gdf, (FeatureType.DATA, "BANDS"),
                                 max_age_days=cfg["burn_max_age_days"])(eop)

        burn = eop.mask["BURN"][0, ..., 0].astype(bool)
        obscured = (eop.mask["OCM_CLOUD"][0, ..., 0] | eop.mask["OCM_SHADOW"][0, ..., 0]).astype(bool)
        frac = (burn & obscured).sum() / max(burn.sum(), 1)
        log.info("[%s] candidate %s (scene cc=%.0f%%) burn-scar obscured=%.1f%%",
                 fire_id, date, cc, frac * 100)

        if fallback_eop is None or frac < fallback_frac:
            fallback_eop, fallback_date, fallback_frac = eop, date, frac
        if frac <= cfg["burn_cloud_max_fraction"]:
            accepted_eop, accepted_date = eop, date
            break

    if accepted_eop is None:
        if fallback_eop is None:
            log.info("[%s] no usable acquisition - skipped", fire_id); return []
        log.warning("[%s] no candidate met %.0f%% threshold - using best available "
                    "%s (%.1f%% obscured)", fire_id, cfg["burn_cloud_max_fraction"] * 100,
                    fallback_date, fallback_frac * 100)
        accepted_eop, accepted_date = fallback_eop, fallback_date

    eop, date = accepted_eop, accepted_date
    log.info("[%s] fire %s (%.0f ha) -> S2 date %s accepted", fire_id,
             fire_row["fire_date"].date(), fire_row.get("area_ha", np.nan), date)

    eop = BurnScarRasterTask(fires_gdf, (FeatureType.DATA, "BANDS"),
                             output_feature=(FeatureType.MASK, "BURN_FRESH"),
                             min_age_days=-1, max_age_days=cfg["fresh_burn_max_days"])(eop)
    eop = BurnScarRasterTask(fires_gdf, (FeatureType.DATA, "BANDS"),
                             output_feature=(FeatureType.MASK, "BURN_OLD"),
                             min_age_days=cfg["fresh_burn_max_days"],
                             max_age_days=cfg["burn_max_age_days"])(eop)
    eop = CombineMaskTask(include_cirrus=cfg["include_scl_cirrus_as_cloud"])(eop)

    files = export_products(eop, cfg["out_dir"], fire_id)
    log.info("[%s] wrote %d files", fire_id, len(files))
    return files
