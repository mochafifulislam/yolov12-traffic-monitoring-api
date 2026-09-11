import os
import time
import uuid
from collections import defaultdict
from typing import Optional

import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


CLASS_MAPPING = {
    "LV": "LIGHT VEHICLE",
    "MC": "MOTORCYCLE",
    "HV": "HEAVY VEHICLE"
}


class VehicleCounter:
    """
    Pipeline YOLO + DeepSORT + line-crossing counter.
    """

    def __init__(
        self,
        weights_path: str,
        allowed_classes: Optional[list] = None,
        conf_threshold: float = 0.5,
        line_pos_ratio: float = 0.85,
        line_orientation: str = "horizontal",
    ):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Model weights tidak ditemukan di '{weights_path}'. "
                "Pastikan file best.pt tersedia."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights_path)

        self.allowed_classes = allowed_classes or ["HV", "LV", "MC"]
        self.conf_threshold = conf_threshold
        self.line_pos_ratio = line_pos_ratio
        self.line_orientation = line_orientation
        self.stream_results = {}
    # ==========================================================
    # YOLO DETECTION
    # ==========================================================

    def _detections_from_results(self, results):
        dets = []

        if results is None or results.boxes is None:
            return dets

        if len(results.boxes) == 0:
            return dets

        boxes = results.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        names = self.model.names

        for i in range(len(confs)):
            conf = float(confs[i])

            if conf < self.conf_threshold:
                continue

            cls_name = names.get(
                int(cls_ids[i]),
                str(cls_ids[i])
            )

            if cls_name not in self.allowed_classes:
                continue

            # Map abbreviations to standard long names for frontend consistency
            mapped_cls_name = CLASS_MAPPING.get(cls_name, cls_name)

            x1, y1, x2, y2 = xyxy[i]

            dets.append(
                (
                    [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    conf,
                    mapped_cls_name,
                )
            )

        return dets

    # ==========================================================
    # PROCESS VIDEO NORMAL
    # ==========================================================

    def process_video(self, input_path: str, output_path: str) -> dict:

        tracker = DeepSort(max_age=30)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise IOError(
                f"Tidak dapat membuka video: {input_path}"
            )

        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise IOError("Video kosong atau rusak.")

        orig_h, orig_w = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out_fps = cap.get(cv2.CAP_PROP_FPS)

        if not out_fps or out_fps <= 0:
            out_fps = 30

        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            out_fps,
            (orig_w, orig_h),
        )

        if self.line_orientation == "horizontal":

            line_y = int(
                orig_h * self.line_pos_ratio
            )

            line_p1 = (0, line_y)
            line_p2 = (orig_w, line_y)

        else:

            line_x = int(
                orig_w * self.line_pos_ratio
            )

            line_p1 = (line_x, 0)
            line_p2 = (line_x, orig_h)

        previous_centroid = {}
        counted_ids = set()

        counter = defaultdict(int)

        fps_list = []

        total_frames = 0

        start_global = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            start_frame = time.time()

            total_frames += 1

            # ----------------------------------------------
            # YOLO
            # ----------------------------------------------

            results = self.model(
                frame,
                verbose=False
            )[0]

            dets = self._detections_from_results(
                results
            )

            # ----------------------------------------------
            # DEEPSORT
            # ----------------------------------------------

            tracks = tracker.update_tracks(
                dets,
                frame=frame
            )

            # ----------------------------------------------
            # TRACKING + COUNTING
            # ----------------------------------------------

            for track in tracks:

                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                left, top, right, bottom = [
                    int(x)
                    for x in track.to_ltrb()
                ]

                cx = int(
                    (left + right) / 2
                )

                cy = int(
                    (top + bottom) / 2
                )

                cls_name = None

                try:

                    if (
                        hasattr(track, "last_detection")
                        and track.last_detection
                    ):

                        det_tuple = track.last_detection

                        if (
                            isinstance(det_tuple, tuple)
                            and len(det_tuple) >= 3
                        ):

                            cls_name = det_tuple[2]

                    if cls_name is None:

                        cls_name = getattr(
                            track,
                            "det_class",
                            None
                        )

                except Exception:

                    cls_name = None

                cls_name = cls_name or "unknown"

                # ------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 255),
                    2,
                )

                label = (
                    f"ID:{track_id} {cls_name}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        left,
                        max(top - 10, 15)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 255),
                    -1,
                )

                # ------------------------------------------
                # LINE CROSSING
                # ------------------------------------------

                prev = previous_centroid.get(
                    track_id
                )

                previous_centroid[
                    track_id
                ] = (cx, cy)

                crossed = False

                if prev is not None:

                    px, py = prev

                    if self.line_orientation == "horizontal":

                        crossed = (
                            py < line_y <= cy
                        )

                    else:

                        crossed = (
                            px < line_x <= cx
                        )

                if (
                    crossed
                    and track_id not in counted_ids
                ):

                    counter[cls_name] += 1

                    counted_ids.add(track_id)

            # ----------------------------------------------
            # DRAW COUNTING LINE
            # ----------------------------------------------

            cv2.line(
                frame,
                line_p1,
                line_p2,
                (0, 255, 255),
                3,
            )

            # ----------------------------------------------
            # DRAW COUNTER
            # ----------------------------------------------

            y0 = 80

            for i, (cls_name, cnt) in enumerate(
                counter.items()
            ):

                cv2.putText(
                    frame,
                    f"{cls_name}: {cnt}",
                    (20, y0 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3,
                )

            # ----------------------------------------------
            # FPS
            # ----------------------------------------------

            fps = 1.0 / max(
                time.time() - start_frame,
                1e-6
            )

            fps_list.append(fps)

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3,
            )

            video_writer.write(frame)

        cap.release()
        video_writer.release()

        return {
            "counts": dict(counter),
            "total_frames": total_frames,
            "avg_fps": (
                round(
                    sum(fps_list) / len(fps_list),
                    2
                )
                if fps_list
                else 0
            ),
            "min_fps": (
                round(min(fps_list), 2)
                if fps_list
                else 0
            ),
            "max_fps": (
                round(max(fps_list), 2)
                if fps_list
                else 0
            ),
            "runtime_seconds": round(
                time.time() - start_global,
                2
            ),
            "device": self.device,
            "output_video": output_path,
        }

    # ==========================================================
    # PROCESS VIDEO STREAM
    # ==========================================================

    def process_video_stream(self, input_path: str, job_id: str):
        """
        Memproses video frame-by-frame.

        Setiap frame akan:
        1. dideteksi menggunakan YOLO
        2. dilacak menggunakan DeepSORT
        3. dihitung berdasarkan line crossing
        4. diberi anotasi
        5. dikonversi menjadi JPEG

        Generator ini digunakan oleh endpoint
        /stream/{job_id}.
        """

        tracker = DeepSort(max_age=30)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise IOError(
                f"Tidak dapat membuka video: {input_path}"
            )

        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise IOError(
                "Video kosong atau rusak."
            )

        orig_h, orig_w = frame.shape[:2]

        # ----------------------------------------------
        # COUNTING LINE
        # ----------------------------------------------

        if self.line_orientation == "horizontal":

            line_y = int(
                orig_h * self.line_pos_ratio
            )

            line_p1 = (0, line_y)
            line_p2 = (orig_w, line_y)

        else:

            line_x = int(
                orig_w * self.line_pos_ratio
            )

            line_p1 = (line_x, 0)
            line_p2 = (line_x, orig_h)

        # ----------------------------------------------
        # STATE
        # ----------------------------------------------

        previous_centroid = {}
        counted_ids = set()
        counter = defaultdict(int)

        fps_list = []
        total_frames = 0
        start_global = time.time()

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                total_frames += 1
                start_frame = time.time()

                # ------------------------------------------
                # YOLO
                # ------------------------------------------

                results = self.model(
                    frame,
                    verbose=False
                )[0]

                dets = self._detections_from_results(
                    results
                )

                # ------------------------------------------
                # DEEPSORT
                # ------------------------------------------

                tracks = tracker.update_tracks(
                    dets,
                    frame=frame
                )

                # ------------------------------------------
                # TRACKING
                # ------------------------------------------

                for track in tracks:

                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id

                    left, top, right, bottom = [
                        int(x)
                        for x in track.to_ltrb()
                    ]

                    cx = int(
                        (left + right) / 2
                    )

                    cy = int(
                        (top + bottom) / 2
                    )

                    cls_name = None

                    try:

                        if (
                            hasattr(
                                track,
                                "last_detection"
                            )
                            and track.last_detection
                        ):

                            det_tuple = (
                                track.last_detection
                            )

                            if (
                                isinstance(
                                    det_tuple,
                                    tuple
                                )
                                and len(det_tuple) >= 3
                            ):

                                cls_name = (
                                    det_tuple[2]
                                )

                        if cls_name is None:

                            cls_name = getattr(
                                track,
                                "det_class",
                                None
                            )

                    except Exception:

                        cls_name = None

                    cls_name = (
                        cls_name or "unknown"
                    )

                    # --------------------------------------
                    # BOUNDING BOX
                    # --------------------------------------

                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"ID:{track_id} {cls_name}",
                        (
                            left,
                            max(top - 10, 15)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (0, 255, 255),
                        -1,
                    )

                    # --------------------------------------
                    # LINE CROSSING
                    # --------------------------------------

                    prev = previous_centroid.get(
                        track_id
                    )

                    previous_centroid[
                        track_id
                    ] = (cx, cy)

                    crossed = False

                    if prev is not None:

                        px, py = prev

                        if (
                            self.line_orientation
                            == "horizontal"
                        ):

                            crossed = (
                                py < line_y <= cy
                            )

                        else:

                            crossed = (
                                px < line_x <= cx
                            )

                    if (
                        crossed
                        and track_id
                        not in counted_ids
                    ):

                        counter[
                            cls_name
                        ] += 1

                        counted_ids.add(
                            track_id
                        )

                # ------------------------------------------
                # COUNTING LINE
                # ------------------------------------------

                cv2.line(
                    frame,
                    line_p1,
                    line_p2,
                    (0, 255, 255),
                    3,
                )

                # ------------------------------------------
                # COUNTER PANEL
                # ------------------------------------------

                y0 = 80

                for i, (
                    cls_name,
                    cnt
                ) in enumerate(counter.items()):

                    cv2.putText(
                        frame,
                        f"{cls_name}: {cnt}",
                        (
                            20,
                            y0 + i * 30
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3,
                    )

                # ------------------------------------------
                # FPS
                # ------------------------------------------

                fps = 1.0 / max(
                    time.time() - start_frame,
                    1e-6
                )

                fps_list.append(fps)

                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3,
                )

                # ------------------------------------------
                # JPEG
                # ------------------------------------------

                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),
                        85,
                    ],
                )

                if not success:
                    continue

                yield buffer.tobytes()

                # ------------------------------------------
                # LIVE UPDATE (for frontend polling)
                # ------------------------------------------

                self.stream_results[job_id] = {
                    "status": "processing",
                    "counts": dict(counter),
                    "total_frames": total_frames,
                    "avg_fps": (
                        round(
                            sum(fps_list) / len(fps_list),
                            2
                        )
                        if fps_list
                        else 0
                    ),
                    "runtime_seconds": round(
                        time.time() - start_global,
                        2
                    ),
                    "device": self.device,
                }

        finally:

            cap.release()

            runtime = time.time() - start_global

            self.stream_results[job_id] = {
                "status": "completed",
                "counts": dict(counter),
                "total_frames": total_frames,
                "avg_fps": (
                    round(sum(fps_list) / len(fps_list), 2)
                    if fps_list
                    else 0
                ),
                "min_fps": (
                    round(min(fps_list), 2)
                    if fps_list
                    else 0
                ),
                "max_fps": (
                    round(max(fps_list), 2)
                    if fps_list
                    else 0
                ),
                "runtime_seconds": round(runtime, 2),
                "device": self.device,
            }


# ==========================================================
# JOB ID
# ==========================================================

def generate_job_id() -> str:
    return uuid.uuid4().hex[:12]