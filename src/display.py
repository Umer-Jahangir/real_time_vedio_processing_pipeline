import cv2

def render_streams(streams, cpu, memory):
    frames = []

    for stream in streams:
        if stream.frame is None:
            continue

        frame = stream.frame.copy()

        cv2.putText(frame,
                    f"Stream {stream.stream_id} | Lat: {stream.latency:.1f} ms | FPS: {stream.fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    f"CPU: {cpu}% | RAM: {memory}%",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

        frames.append(frame)

    if len(frames) == 3:
        top = cv2.hconcat([frames[0], frames[1]])
        bottom = cv2.resize(frames[2], (top.shape[1], frames[2].shape[0]))
        tiled = cv2.vconcat([top, bottom])
        cv2.imshow("Multi-Stream AI System", tiled)
