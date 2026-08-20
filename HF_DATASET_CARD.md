---
license: other
license_name: copernicus-sentinel-data-legal-notice
license_link: https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice
task_categories:
- image-segmentation
tags:
- remote-sensing
- earth-observation
- sentinel-2
- burn-scar
- wildfire
- segmentation
- effis
- geospatial
size_categories:
- 10K<n<100K
---

# burn2scar

Sentinel-2 imagery paired with EFFIS-derived burn-scar segmentation masks — real,
unsimulated satellite data, no sensor simulation applied.

Extraction code: [github.com/paramkaur10/burn2scar](https://github.com/paramkaur10/burn2scar)

---

## Dataset summary

For each wildfire recorded in EFFIS (European Forest Fire Information System), this
dataset provides two Sentinel-2 acquisitions — one taken shortly after the fire and
one taken several months later — each paired with a 7-class segmentation mask
identifying clear land, fresh burn, old burn, cloud, cloud shadow, water, and nodata.

- **13,903 fires** with at least one usable acquisition (19,443 queried; the
 remainder had no candidate scene passing the cloud/obscuration threshold in
 either window), **26,854 scene/mask pairs** total
- **Countries**: Italy, France (south of 46°N), Spain, Greece
- **Date range**: 2020–2026
- **30.57 GB** total

---

## Classes

| Mask value | Class | Definition |
|---|---|---|
| 0 | clear | No burn, cloud, shadow, or water detected |
| 2 | fresh burn | Burned area, ≤ 90 days since ignition |
| 3 | old burn | Burned area, > 90 days since ignition (up to 365 days) |
| 4 | cloud | Detected via OmniCloudMask |
| 5 | cloud shadow | Detected via OmniCloudMask |
| 6 | water | NDWI > 0.05, or Sentinel-2 SCL value 6 |
| 255 | nodata | Outside the valid data mask |

---

## Class distribution

Computed across all 26,854 masks (1.76 billion pixels total):

| Class | % of valid pixels |
|---|---|
| clear | 82.62% |
| old_burn | 7.21% |
| fresh_burn | 7.12% |
| water | 2.32% |
| cloud | 0.54% |
| shadow | 0.19% |

0.95% of all pixels are nodata (tile-edge artifacts from UTM reprojection),
excluded from the percentages above. Fresh and old burn are closely balanced —
a result of acquiring two independent time windows per fire rather than relying
on incidental overlap with other fires' perimeters. See **Limitations** below
for what `clear` does and doesn't represent.

---

## Dataset structure

```
acquisitions/
  <fire_id>/
    <timestamp>/
      <fire_id>_<timestamp>_s2.tif     # 13-band image, uint16
      <fire_id>_<timestamp>_mask.tif   # 1-band label mask, uint8
```

Each `fire_id` (e.g. `effis_52171`) may contain **one or two timestamp
subfolders** — one from the fresh-burn acquisition window (5–60 days post-fire)
and, when a usable scene was found, one from the old-burn window (120–300 days
post-fire). These are independent acquisitions of the same location, not a single
paired before/after image — see **Limitations** below.

---

## Image format

- **Bands** (13, in order): `B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12`
- **Resolution**: 10 m (coarser native bands resampled up to this common grid)
- **Tile size**: 256 × 256 pixels (2.56 km × 2.56 km), centered on the fire
- **Encoding**: `uint16`, scale factor 10,000 → divide by 10,000 for TOA reflectance
- **Processing level**: Sentinel-2 L1C (top-of-atmosphere) — no atmospheric
 correction, no sensor simulation, no synthetic degradation

---

## How to load

This dataset is distributed as georeferenced GeoTIFF pairs rather than a flat
image-classification layout, so the standard `datasets.load_dataset()` image
loader does not apply directly. Load with `rasterio`:

```python
import rasterio
import numpy as np
from huggingface_hub import hf_hub_download

repo_id = "neet1797/burn2scar"

s2_path = hf_hub_download(repo_id, repo_type="dataset",
                          filename="acquisitions/effis_52171/2021-09-19T.../effis_52171_..._s2.tif")
mask_path = hf_hub_download(repo_id, repo_type="dataset",
                            filename="acquisitions/effis_52171/2021-09-19T.../effis_52171_..._mask.tif")

with rasterio.open(s2_path) as src:
    bands = src.read()          # shape: (13, 256, 256), uint16
    reflectance = bands.astype(np.float32) / 10000.0

with rasterio.open(mask_path) as src:
    mask = src.read(1)          # shape: (256, 256), uint8, values in {0,2,3,4,5,6,255}
```

To browse all files programmatically, use `huggingface_hub.HfApi().list_repo_files()`
or `snapshot_download()` for a full local mirror.

---

## Fire selection criteria

| Filter | Value |
|---|---|
| Countries | IT, FR (south of 46°N), ES, EL |
| Date range | 2020-01-01 to 2026-08-04 |
| Area | 5–2,000 hectares |

Area bounds exclude both marginal micro-detections and mega-fires large enough to
make a single tile 100% burn scar with no useful class boundary.

---

## Source data

- **Imagery**: Copernicus Sentinel-2, via the Copernicus Data Space Ecosystem
- **Burn labels**: EFFIS (European Forest Fire Information System), Copernicus
 Emergency Management Service
- **Cloud / shadow**: [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
- **Water**: NDWI + Sentinel-2 Scene Classification Layer (SCL)

---

## Limitations

- **No pre-fire baseline.** Both acquisition windows are strictly post-fire; this
 dataset does not provide a true "before" image for change-detection-style pairing.
- **Cloud cover permitted elsewhere in-frame.** Scene selection requires the burn
 scar itself to be minimally obscured (≤ 50%), but up to 70% cloud cover is
 permitted in the wider scene outside the scar.
- **`clear` is a residual class, not a confirmed vegetation label.** It covers
 anything not detected as burn, cloud, shadow, or water. Mediterranean terrain
 in particular can put bare soil, exposed rock, and volcanic terrain (present
 in both Italy and Greece) into `clear`, since there is no separate class for
 them. Users needing a purer vegetation mask should combine this dataset with
 a land-cover product (e.g. ESA WorldCover) and filter further based on their
 own criteria.
- **Burn labels are only as complete as EFFIS's own detection.** EFFIS's
 documented minimum mapping unit for burnt-area products is 10 hectares —
 fires smaller than that are not labeled by EFFIS, and so appear as `clear`
 in this dataset rather than `fresh_burn`/`old_burn`, even where a real
 (sub-threshold) fire occurred.

---

## Licensing

- **Sentinel-2 imagery**: governed by the Copernicus Sentinel Data Legal Notice —
 free, full, and open, permitting reproduction, distribution, and adaptation,
 with attribution required ("Contains modified Copernicus Sentinel data").
- **EFFIS burn-perimeter labels**: derived from Copernicus Emergency Management
 Service data. Redistribution terms for this specific product have not been
 independently verified beyond the general Copernicus data policy — consult
 EFFIS/CEMS directly before relying on this for redistribution outside your
 organization.

This is not legal advice; verify licensing terms independently for your use case.

## Citation

If you use this dataset in your work, please cite:

```bibtex
@misc{thind2026burn2scar,
  author    = {Thind, Parampuneet Kaur},
  title     = {burn2scar: A Sentinel-2 Burn-Scar Segmentation Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/neet1797/burn2scar}
}
```

Please also retain attribution to the underlying source data, as required by its
own license terms:

```
Contains modified Copernicus Sentinel data.
Burn labels derived from EFFIS (European Forest Fire Information System).
```

---

**Author:** Parampuneet Kaur Thind (Param)