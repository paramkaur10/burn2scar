"""Streaming upload to Hugging Face — download one fire, upload, delete locally."""
import json
import logging
import os
import shutil

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def init_hf(repo_id, token):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    return api


def load_quota_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"running_by_country": {}, "done_fire_ids": []}


def save_quota_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


def upload_and_cleanup(api, repo_id, fire_dir, fire_id):
    if not os.path.isdir(fire_dir):
        return None
    size = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(fire_dir) for f in fs)
    if size == 0:
        return 0
    try:
        api.upload_folder(folder_path=fire_dir, path_in_repo=f"acquisitions/{fire_id}",
                          repo_id=repo_id, repo_type="dataset")
        shutil.rmtree(fire_dir, ignore_errors=True)
        return size
    except Exception as exc:
        log.error("[%s] upload FAILED, keeping local files for retry: %s", fire_id, exc)
        return None
