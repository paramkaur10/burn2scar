import argparse
import copy
import gc
import itertools
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.effis_loader import load_effis
from src.s2_pipeline import make_sh_config, process_fire
from src.hf_upload import init_hf, load_quota_state, save_quota_state, upload_and_cleanup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sh = make_sh_config(os.environ["SH_CLIENT_ID"], os.environ["SH_CLIENT_SECRET"])
    api = init_hf(cfg["hf_repo_id"], os.environ["HF_TOKEN"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    fires_all = load_effis(cfg["effis_path"], countries=cfg["countries"],
                           start_date=cfg["start_date"], end_date=cfg["end_date"],
                           min_area_ha=cfg["min_area_ha"])
    if cfg.get("max_area_ha"):
        before = len(fires_all)
        fires_all = fires_all[fires_all["area_ha"] <= cfg["max_area_ha"]]
        log.info("max_area_ha filter: %d -> %d fires", before, len(fires_all))
    is_fr = fires_all["country"] == "FR"
    southern_ok = ~is_fr | (fires_all.geometry.centroid.y <= cfg["southern_france_max_lat"])
    fires = fires_all[southern_ok].reset_index(drop=True)
    log.info("%d fires queued after filtering", len(fires))

    if args.smoke_test:
        row = fires.iloc[0]
        files = process_fire(row, fires, cfg, sh)
        if not files:
            log.error("smoke test produced no files"); return
        fire_dir = os.path.dirname(os.path.dirname(files[0]))
        size = upload_and_cleanup(api, cfg["hf_repo_id"], fire_dir, row["fire_id"])
        log.info("smoke test: uploaded %s bytes for fire %s -- check the HF repo now",
                 size, row["fire_id"])
        return

    quota_path = os.path.join(cfg["out_dir"], "quota_state.json")
    state = load_quota_state(quota_path)
    running_by_country = state["running_by_country"]
    done_fire_ids = set(state["done_fire_ids"])
    log.info("Resumed: %d fires already done", len(done_fire_ids))

    country_budget_bytes = {c: frac * cfg["total_budget_gb"] * 1e9
                            for c, frac in cfg["country_quota"].items()}
    device_cycle = itertools.cycle(["cuda:0", "cuda:1"])
    lock = threading.Lock()
    stop_flags = {}

    def free_gb():
        return shutil.disk_usage(cfg["out_dir"]).free / 1e9

    def worker(fire_row):
        country, fid = fire_row["country"], fire_row["fire_id"]
        if fid in done_fire_ids:
            return "already done", 0
        if stop_flags.get("total") or stop_flags.get(country):
            return f"skipped ({country}/total budget reached)", 0
        if free_gb() < cfg["min_free_gb"]:
            stop_flags["total"] = True
            log.error("Free disk below %d GB - stopping", cfg["min_free_gb"])
            return "skipped (low disk)", 0

        with lock:
            if running_by_country.get(country, 0) >= country_budget_bytes.get(country, 0):
                stop_flags[country] = True
                return f"skipped ({country} quota reached)", 0
            if sum(running_by_country.values()) >= cfg["total_budget_gb"] * 1e9:
                stop_flags["total"] = True
                return "skipped (total budget reached)", 0

        run_cfg = copy.deepcopy(cfg)
        run_cfg["ocm_device"] = next(device_cycle)
        try:
            files = process_fire(fire_row, fires, run_cfg, sh)
            if not files:
                return "skipped", 0
            fire_dir = os.path.dirname(os.path.dirname(files[0]))
            size = upload_and_cleanup(api, cfg["hf_repo_id"], fire_dir, fid)
            if size is None:
                return "upload failed - will retry", 0
            with lock:
                running_by_country[country] = running_by_country.get(country, 0) + size
                done_fire_ids.add(fid)
                save_quota_state(quota_path, {"running_by_country": running_by_country,
                                              "done_fire_ids": list(done_fire_ids)})
            return "ok (uploaded)", size
        finally:
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            shutil.rmtree(run_cfg["cache_folder"], ignore_errors=True)

    index_rows = []
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(worker, fire): fire for _, fire in fires.iterrows()}
        for fut in as_completed(futures):
            fire = futures[fut]
            try:
                status, size = fut.result()
            except Exception as exc:
                log.exception("Fire %s failed", fire["fire_id"])
                status, size = f"error: {exc}", 0
            index_rows.append(dict(fire_id=fire["fire_id"], country=fire["country"],
                                   status=status, mb_uploaded=round(size / 1e6, 1)))
            log.info("progress %d/%d | total uploaded %.1f/%d GB | free disk %.0f GB",
                     len(index_rows), len(fires), sum(running_by_country.values()) / 1e9,
                     cfg["total_budget_gb"], free_gb())

    pd.DataFrame(index_rows).to_csv(os.path.join(cfg["out_dir"], "processing_index.csv"), index=False)
    log.info("Done. Total uploaded: %.2f GB", sum(running_by_country.values()) / 1e9)


if __name__ == "__main__":
    main()
