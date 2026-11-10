#!/usr/bin/env python3
"""Slit-scan en video (time displacement): cada zona del cuadro muestra un
instante distinto del pasado.

Se guarda un buffer circular con los últimos (max-delay + 1) fotogramas. Para
cada píxel, un "mapa de retraso" dice cuántos fotogramas atrás hay que mirar.
Memoria: (max-delay+1) * ancho * alto * 3 bytes (60 retrasos a 960x540 ~ 95 MB).

Formas del mapa de retraso:
  horizontal  bandas horizontales: arriba retraso 0, abajo el máximo
  vertical    bandas verticales: izquierda 0, derecha el máximo
  radial      círculos: centro 0, esquinas el máximo (use --invert para lo contrario)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffio import VideoWriter, probe, read_frames  # noqa: E402


def delay_map(shape, h, w, max_delay, invert=False, steps=0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if shape == "horizontal":
        m = yy / (h - 1)
    elif shape == "vertical":
        m = xx / (w - 1)
    else:  # radial
        r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
        m = r / r.max()
    if invert:
        m = 1.0 - m
    if steps > 1:  # escalonado: bandas duras en vez de un degradado suave
        m = np.floor(m * steps).clip(0, steps - 1) / (steps - 1)
    return np.rint(m * max_delay).astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", help="MP4 H.264 yuv420p")
    ap.add_argument("--shape", choices=["horizontal", "vertical", "radial"], default="horizontal")
    ap.add_argument("--max-delay", type=int, default=30, help="retraso máximo en fotogramas. Def: 30")
    ap.add_argument("--invert", action="store_true", help="invierte el mapa (el máximo pasa al otro extremo)")
    ap.add_argument("--steps", type=int, default=0,
                    help="cuantiza el mapa en N bandas duras (0 = degradado continuo)")
    ap.add_argument("--width", type=int, default=None, help="reescalar al leer")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--save-map", default=None, help="guarda el mapa de retraso como PNG para inspeccionarlo")
    a = ap.parse_args()

    w, h, fps, _ = probe(a.input)
    if a.width:
        h = int(round(h * a.width / w / 2) * 2)
        w = a.width
    D = a.max_delay
    dmap = delay_map(a.shape, h, w, D, a.invert, a.steps)
    if a.save_map:
        from PIL import Image
        Image.fromarray((dmap * 255 // max(D, 1)).astype(np.uint8)).save(a.save_map)

    size = D + 1
    buf = None                       # buffer circular (size, h, w, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    out = VideoWriter(a.output, w, h, fps, crf=a.crf)
    t = -1
    for t, f in enumerate(read_frames(a.input, a.width)):
        if buf is None:              # al inicio se rellena con el primer fotograma
            buf = np.repeat(f[None], size, axis=0)
        head = t % size
        buf[head] = f
        idx = (head - dmap) % size   # qué ranura del buffer corresponde a cada píxel
        out.write(buf[idx, yy, xx])
    out.close()
    print(f"OK -> {a.output}  {w}x{h}  {t + 1} fotogramas, forma={a.shape}, max_delay={D}")


if __name__ == "__main__":
    main()
