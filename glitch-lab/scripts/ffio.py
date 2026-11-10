"""Lectura/escritura de video fotograma a fotograma mediante pipes de ffmpeg.

Nunca se carga el video entero en RAM: se lee un fotograma RGB24 cada vez.
"""
import json
import subprocess

import numpy as np


def probe(path):
    """Devuelve (ancho, alto, fps_str, n_fotogramas_estimado) del primer stream de video."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    n = st.get("nb_frames")
    if not n or n == "N/A":
        n = round(float(info["format"]["duration"]) * num / den)
    return st["width"], st["height"], st["r_frame_rate"], int(n)


def read_frames(path, width=None, start=None, duration=None):
    """Generador de fotogramas RGB (H, W, 3) uint8. width reescala manteniendo aspecto."""
    w, h, fps, _ = probe(path)
    if width and width != w:
        h = int(round(h * width / w / 2) * 2)
        w = width
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-an",
            "-vf", f"scale={w}:{h}:in_color_matrix=bt709:flags=bicubic,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if proc.poll() is None:  # lectura cortada antes del final: cerrar ffmpeg sin ruido
            proc.kill()
        proc.stdout.close()
        proc.wait()


class VideoWriter:
    """Escribe fotogramas RGB a MP4 H.264 yuv420p BT.709 (rango TV)."""

    def __init__(self, path, width, height, fps="30000/1001", crf=16, preset="medium"):
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", fps, "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
               "-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg falló al escribir el video")
