import os
import uuid
import threading
import traceback

from flask import Flask, request, render_template, send_file, jsonify, after_this_request

from engine import pipeline, jobs, ner, ocr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "jobs"), exist_ok=True)

MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    return render_template("index.html", ner_active=ner.ner_available(),
                            ocr_languages=ocr.active_ocr_langs())


def _run_extraction_job(job_id, input_path):
    """
    Runs on a background thread. OCR (especially multilingual, at 300
    DPI, over a multi-page PDF) can easily take well past what a
    reverse proxy or load balancer will hold a single HTTP request open
    for — that mismatch is what silently drops the connection and shows
    up client-side as an opaque "network error" with no real
    explanation. Running it off-request and having the client poll for
    the result means the only requests ever left open are near-instant
    ones, so nothing has a mid-length window to be killed in.
    """
    try:
        page_images, instances, ocr_cache = pipeline.extract_fields(input_path)
    except Exception as exc:
        traceback.print_exc()
        jobs.set_status(BASE_DIR, job_id, "error", error=f"Could not read this PDF: {exc}")
        return
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)  # never keep the original once it's been OCR'd

    try:
        for idx, img in enumerate(page_images):
            jobs.save_page_image(BASE_DIR, job_id, idx, img)
        jobs.save_ocr_data(BASE_DIR, job_id, ocr_cache)
        jobs.save_instances(BASE_DIR, job_id, instances, len(page_images))

        groups = pipeline.group_for_ui(instances)
        result = {
            "num_pages": len(page_images),
            "groups": groups,
            "ner_active": ner.ner_available(),
            "ocr_languages": ocr.active_ocr_langs(),
        }
        if not groups:
            result["message"] = ("No standard fields were detected automatically. "
                                  "You can still describe what to mask in plain text below.")
        jobs.set_status(BASE_DIR, job_id, "done", extra=result)
    except Exception as exc:
        traceback.print_exc()
        jobs.set_status(BASE_DIR, job_id, "error", error=f"Could not process this PDF: {exc}")


@app.route("/extract", methods=["POST"])
def extract():
    jobs.cleanup_stale_jobs(BASE_DIR)

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    file.save(input_path)

    jobs.create_job(BASE_DIR, job_id)
    jobs.set_status(BASE_DIR, job_id, "processing")

    thread = threading.Thread(target=_run_extraction_job, args=(job_id, input_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "processing"}), 202


@app.route("/extract/status/<job_id>")
def extract_status(job_id):
    status = jobs.get_status(BASE_DIR, job_id)
    if status is None:
        return jsonify({"status": "error", "error": "Unknown or expired job — please re-upload"}), 404
    # Always 200 here — the poller distinguishes states via the "status"
    # field, not the HTTP status, so a slow-but-healthy "processing"
    # response is never confused with an actual transport-level failure.
    return jsonify({"job_id": job_id, **status})


@app.route("/mask", methods=["POST"])
def mask():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")
    selected_group_ids = set(body.get("group_ids", []))
    instructions = (body.get("instructions") or "").strip()

    if not job_id:
        return jsonify({"error": "Missing job_id — please re-upload the document"}), 400

    job_data = jobs.load_job_data(BASE_DIR, job_id)
    if job_data is None:
        return jsonify({"error": "This session has expired — please re-upload the document"}), 400

    all_instances = job_data["instances"]
    num_pages = job_data["num_pages"]

    # Re-derive each instance's group_id the same way group_for_ui does,
    # so a selected checkbox maps back to every matching instance.
    selected_instances = [
        inst for inst in all_instances
        if f"{inst['category']}::{inst['field_type']}::{inst['display_label']}" in selected_group_ids
    ]

    if instructions:
        ocr_cache = jobs.load_ocr_data(BASE_DIR, job_id)
        if ocr_cache:
            selected_instances += pipeline.run_custom_search(ocr_cache, instructions)

    if not selected_instances:
        return jsonify({"error": "Select at least one field, or describe what to mask"}), 400

    try:
        page_images = [jobs.load_page_image(BASE_DIR, job_id, i) for i in range(num_pages)]
    except Exception as exc:
        return jsonify({"error": f"Session data missing: {exc}"}), 400

    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_masked.pdf")
    try:
        pipeline.render_masked_pdf(page_images, selected_instances, output_path)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Masking failed: {exc}"}), 500
    finally:
        jobs.cleanup_job(BASE_DIR, job_id)

    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        return response

    return send_file(
        output_path, as_attachment=True,
        download_name="masked_output.pdf", mimetype="application/pdf",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ner_active": ner.ner_available(),
                     "ocr_languages": ocr.active_ocr_langs()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
