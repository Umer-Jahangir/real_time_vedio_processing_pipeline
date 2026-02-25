import cv2
import numpy as np


def show_frame(
    window_name,
    frame,
    model_latency,
    e2e_latency,
    cpu,
    memory,
    fps=None,
):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    box_height = 110
    box_width = int(w * 0.45)

    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    y0 = 25
    dy = 22

    cv2.putText(
        frame,
        f"Model: {model_latency:.1f} ms",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"E2E: {e2e_latency:.1f} ms",
        (10, y0 + dy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )

    cv2.putText(
        frame,
        f"CPU: {cpu:.1f}% | RAM: {memory:.1f}%",
        (10, y0 + 2 * dy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, y0 + 3 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow(window_name, frame)


def show_finished(window_name, width=320, height=256):
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        placeholder,
        "Finished",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.imshow(window_name, placeholder)