import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.tracker_service import VehicleCounter, generate_job_id


# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "best.pt")
)

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="Vehicle Detection & Counting API",
    description=(
        "YOLO + DeepSORT vehicle detection, "
        "tracking, dan counting dari rekaman CCTV."
    ),
    version="1.0.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# GLOBAL MODEL
# ==========================================================

_counter_instance: VehicleCounter | None = None


# ==========================================================
# JOB STATUS
# ==========================================================

stream_jobs = {}


# ==========================================================
# STARTUP
# ==========================================================

@app.on_event("startup")
def load_model():
    global _counter_instance

    try:
        _counter_instance = VehicleCounter(
            weights_path=MODEL_PATH
        )

        print(
            f"[startup] Model dimuat dari {MODEL_PATH} "
            f"(device={_counter_instance.device})"
        )

    except FileNotFoundError as e:

        print(f"[startup][WARNING] {e}")

        _counter_instance = None


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "ok" if _counter_instance else "model_not_loaded",
        "model_path": MODEL_PATH,
        "device": (
            _counter_instance.device
            if _counter_instance
            else None
        ),
    }


# ==========================================================
# NORMAL VIDEO PROCESSING
# ==========================================================

@app.post("/count-vehicles")
async def count_vehicles(
    video: UploadFile = File(
        ...,
        description="File video CCTV (mp4/avi)"
    ),

    conf_threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0
    ),

    line_pos_ratio: float = Query(
        0.85,
        ge=0.0,
        le=1.0
    ),

    line_orientation: str = Query(
        "horizontal",
        pattern="^(horizontal|vertical)$"
    ),
):

    if _counter_instance is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum tersedia. "
                f"Pastikan file weights ada di {MODEL_PATH}."
            ),
        )

    if not video.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        raise HTTPException(
            status_code=400,
            detail="Format video tidak didukung."
        )

    job_id = generate_job_id()

    input_path = (
        UPLOAD_DIR /
        f"{job_id}_{video.filename}"
    )

    output_path = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    # ------------------------------------------------------
    # SIMPAN VIDEO
    # ------------------------------------------------------

    with open(input_path, "wb") as f:

        shutil.copyfileobj(
            video.file,
            f
        )

    # ------------------------------------------------------
    # SET PARAMETER
    # ------------------------------------------------------

    _counter_instance.conf_threshold = conf_threshold
    _counter_instance.line_pos_ratio = line_pos_ratio
    _counter_instance.line_orientation = line_orientation

    # ------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------

    try:

        result = _counter_instance.process_video(
            str(input_path),
            str(output_path)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses video: {e}"
        )

    finally:

        input_path.unlink(
            missing_ok=True
        )

    result["job_id"] = job_id

    result["download_url"] = (
        f"/download/{job_id}"
    )

    result.pop(
        "output_video",
        None
    )

    return result


# ==========================================================
# PROCESS STREAM
# ==========================================================

@app.post("/process-stream")
async def process_stream(
    video: UploadFile = File(
        ...,
        description="File video CCTV"
    ),

    conf_threshold: float = Query(
        0.5,
        ge=0.0,
        le=1.0
    ),

    line_pos_ratio: float = Query(
        0.85,
        ge=0.0,
        le=1.0
    ),

    line_orientation: str = Query(
        "horizontal",
        pattern="^(horizontal|vertical)$"
    ),
):

    if _counter_instance is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum tersedia. "
                f"Pastikan file weights ada di {MODEL_PATH}."
            ),
        )

    if not video.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):

        raise HTTPException(
            status_code=400,
            detail="Format video tidak didukung."
        )

    # ------------------------------------------------------
    # JOB ID
    # ------------------------------------------------------

    job_id = generate_job_id()

    input_path = (
        UPLOAD_DIR /
        f"stream_{job_id}_{video.filename}"
    )

    # ------------------------------------------------------
    # SIMPAN VIDEO
    # ------------------------------------------------------

    with open(input_path, "wb") as f:

        shutil.copyfileobj(
            video.file,
            f
        )

    # ------------------------------------------------------
    # SIMPAN STATUS JOB
    # ------------------------------------------------------

    stream_jobs[job_id] = {
        "status": "ready",
        "counts": {},
        "total_frames": 0,
        "avg_fps": 0,
        "min_fps": 0,
        "max_fps": 0,
        "runtime_seconds": 0,
        "device": _counter_instance.device,
        "error": None,
    }

    # ------------------------------------------------------
    # SIMPAN PARAMETER PER JOB
    # ------------------------------------------------------

    stream_jobs[job_id]["conf_threshold"] = conf_threshold
    stream_jobs[job_id]["line_pos_ratio"] = line_pos_ratio
    stream_jobs[job_id]["line_orientation"] = line_orientation

    return {
        "job_id": job_id,
        "stream_url": f"/stream/{job_id}",
        "message": (
            "Video berhasil di-upload "
            "dan siap diproses."
        ),
    }


# ==========================================================
# STREAM VIDEO
# ==========================================================

@app.get("/stream/{job_id}")
def stream_video(job_id: str):

    # ------------------------------------------------------
    # CEK JOB
    # ------------------------------------------------------

    if job_id not in stream_jobs:

        raise HTTPException(
            status_code=404,
            detail="Job tidak ditemukan."
        )

    # ------------------------------------------------------
    # CARI VIDEO
    # ------------------------------------------------------

    input_files = list(
        UPLOAD_DIR.glob(
            f"stream_{job_id}_*"
        )
    )

    if not input_files:

        raise HTTPException(
            status_code=404,
            detail=(
                "Video untuk job tersebut "
                "tidak ditemukan."
            )
        )

    input_path = input_files[0]

    # ------------------------------------------------------
    # AMBIL PARAMETER
    # ------------------------------------------------------

    job = stream_jobs[job_id]

    _counter_instance.conf_threshold = (
        job["conf_threshold"]
    )

    _counter_instance.line_pos_ratio = (
        job["line_pos_ratio"]
    )

    _counter_instance.line_orientation = (
        job["line_orientation"]
    )

    # ------------------------------------------------------
    # GENERATOR FRAME
    # ------------------------------------------------------

    def generate():

        stream_jobs[job_id]["status"] = "processing"

        try:

            for frame_bytes in (
                _counter_instance.process_video_stream(
                    str(input_path), job_id
                )
            ):

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )

            # ------------------------------------------------
            # STREAM SELESAI
            # ------------------------------------------------

            stream_jobs[job_id]["status"] = "completed"

        except Exception as e:

            stream_jobs[job_id]["status"] = "error"

            stream_jobs[job_id]["error"] = str(e)

            print(
                f"[stream][ERROR] job={job_id}: {e}"
            )

        finally:

            input_path.unlink(
                missing_ok=True
            )

    return StreamingResponse(
        generate(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ==========================================================
# STREAM RESULT
# ==========================================================

@app.get("/stream-result/{job_id}")
def stream_result(job_id: str):
    if _counter_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum tersedia."
        )

    result = _counter_instance.stream_results.get(job_id)

    if result is None:
        return {
            "status": "processing"
        }

    return result
    
    if job_id not in stream_jobs:

        raise HTTPException(
            status_code=404,
            detail="Job tidak ditemukan."
        )

    job = stream_jobs[job_id]

    return {
        "job_id": job_id,
        "status": job["status"],
        "counts": job["counts"],
        "total_frames": job["total_frames"],
        "avg_fps": job["avg_fps"],
        "min_fps": job["min_fps"],
        "max_fps": job["max_fps"],
        "runtime_seconds": job["runtime_seconds"],
        "device": job["device"],
        "error": job["error"],
    }


# ==========================================================
# DOWNLOAD RESULT
# ==========================================================

@app.get("/download/{job_id}")
def download_result(job_id: str):

    output_path = (
        OUTPUT_DIR /
        f"{job_id}.mp4"
    )

    if not output_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Hasil video tidak ditemukan "
                "atau sudah kedaluwarsa."
            )
        )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=(
            f"vehicle_count_{job_id}.mp4"
        ),
    )