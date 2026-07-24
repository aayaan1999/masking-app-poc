"""
engine/jobs.py
Extraction and masking happen in two separate HTTP requests (extract
first so the UI can show found fields, mask second once the user picks
which ones). This module persists the page images + detected instances
to disk in between, keyed by job_id, and cleans up stale jobs so nothing
lingers on disk longer than necessary.
"""

import json
import os
import time
import shutil

JOB_TTL_SECONDS = 30 * 60  # abandon a job (no /mask call) after 30 minutes


def job_dir(base_dir, job_id):
    return os.path.join(base_dir, "jobs", job_id)


def create_job(base_dir, job_id):
    d = job_dir(base_dir, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_page_image(base_dir, job_id, page_idx, image):
    d = job_dir(base_dir, job_id)
    image.save(os.path.join(d, f"page_{page_idx}.png"))


def load_page_image(base_dir, job_id, page_idx):
    from PIL import Image
    return Image.open(os.path.join(job_dir(base_dir, job_id), f"page_{page_idx}.png"))


def save_ocr_data(base_dir, job_id, pages_words_lines):
    """pages_words_lines: list of (words, lines, img_w, img_h) per page."""
    d = job_dir(base_dir, job_id)
    serializable = [
        {"words": words, "lines": lines, "img_w": img_w, "img_h": img_h}
        for words, lines, img_w, img_h in pages_words_lines
    ]
    with open(os.path.join(d, "ocr_data.json"), "w") as f:
        json.dump(serializable, f)


def load_ocr_data(base_dir, job_id):
    path = os.path.join(job_dir(base_dir, job_id), "ocr_data.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return [(p["words"], p["lines"], p["img_w"], p["img_h"]) for p in raw]


def set_status(base_dir, job_id, status, error=None, extra=None):
    """status: 'processing' | 'done' | 'error'."""
    d = job_dir(base_dir, job_id)
    payload = {"status": status, "updated_at": time.time()}
    if error is not None:
        payload["error"] = error
    if extra:
        payload.update(extra)
    tmp_path = os.path.join(d, "status.json.tmp")
    final_path = os.path.join(d, "status.json")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)  # atomic — a poll never sees a half-written file


def get_status(base_dir, job_id):
    path = os.path.join(job_dir(base_dir, job_id), "status.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_instances(base_dir, job_id, instances, num_pages):
    d = job_dir(base_dir, job_id)
    with open(os.path.join(d, "instances.json"), "w") as f:
        json.dump({
            "created_at": time.time(),
            "num_pages": num_pages,
            "instances": instances,
        }, f)


def load_job_data(base_dir, job_id):
    path = os.path.join(job_dir(base_dir, job_id), "instances.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cleanup_job(base_dir, job_id):
    d = job_dir(base_dir, job_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


def cleanup_stale_jobs(base_dir):
    jobs_root = os.path.join(base_dir, "jobs")
    if not os.path.isdir(jobs_root):
        return
    now = time.time()
    for job_id in os.listdir(jobs_root):
        d = os.path.join(jobs_root, job_id)
        meta_path = os.path.join(d, "instances.json")
        try:
            mtime = os.path.getmtime(meta_path) if os.path.exists(meta_path) else os.path.getmtime(d)
            if now - mtime > JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue
