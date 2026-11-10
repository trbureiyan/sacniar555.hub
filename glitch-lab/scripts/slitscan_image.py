#!/usr/bin/env python3
"""Slit-scan a imagen fija (PNG): convierte el TIEMPO de un video en un eje ESPACIAL.

Modos:
  center   franja vertical fija en el centro; cada fotograma aporta una columna
           que se coloca a la derecha de la anterior (eje X = tiempo).
  scan     modo escáner: la franja recorre el cuadro de izquierda a derecha
           mientras pasa el tiempo; la imagen final tiene el tamaño del cuadro,
           pero cada columna pertenece a un instante distinto.
  hcenter  franja horizontal fija en el centro; los fotogramas se apilan hacia
           abajo (eje Y = tiempo).
  hscan    escáner vertical: la franja horizontal baja de arriba a abajo.

Solo se guarda en memoria la imagen de salida, nunca el video completo.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffio import probe, read_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video de entrada")
    ap.add_argument("output", help="PNG de salida")
    ap.add_argument("--mode", choices=["center", "scan", "hcenter", "hscan"], default="center")
    ap.add_argument("--slit", type=int, default=1,
                    help="grosor de la franja en px tomada de cada fotograma (center/hcenter). Def: 1")
    ap.add_argument("--stretch", type=int, default=3,
                    help="px que ocupa cada fotograma en la salida (alargado). Def: 3")
    ap.add_argument("--pos", type=float, default=0.5,
                    help="posición de la franja fija, 0..1 del ancho/alto. Def: 0.5 (centro)")
    ap.add_argument("--width", type=int, default=None, help="reescalar el video a este ancho al leer")
    ap.add_argument("--start", type=float, default=None, help="segundo inicial")
    ap.add_argument("--duration", type=float, default=None, help="duración en segundos")
    a = ap.parse_args()

    w, h, _, n = probe(a.input)
    if a.width:
        h = int(round(h * a.width / w / 2) * 2)
        w = a.width
    if a.duration:
        num, den = (int(x) for x in probe(a.input)[2].split("/"))
        n = min(n, int(a.duration * num / den))

    if a.mode == "center":
        canvas = np.zeros((h, n * a.stretch, 3), np.uint8)
    elif a.mode == "hcenter":
        canvas = np.zeros((n * a.stretch, w, 3), np.uint8)
    else:
        canvas = np.zeros((h, w, 3), np.uint8)

    used = 0
    for t, f in enumerate(read_frames(a.input, a.width, a.start, a.duration)):
        if t >= n:
            break
        if a.mode == "center":
            x0 = int(a.pos * (w - a.slit))
            strip = Image.fromarray(f[:, x0:x0 + a.slit]).resize((a.stretch, h), Image.BICUBIC)
            canvas[:, t * a.stretch:(t + 1) * a.stretch] = np.asarray(strip)
        elif a.mode == "hcenter":
            y0 = int(a.pos * (h - a.slit))
            strip = Image.fromarray(f[y0:y0 + a.slit]).resize((w, a.stretch), Image.BICUBIC)
            canvas[t * a.stretch:(t + 1) * a.stretch] = np.asarray(strip)
        elif a.mode == "scan":
            x0, x1 = (t * w) // n, ((t + 1) * w) // n
            canvas[:, x0:max(x1, x0 + 1)] = f[:, x0:max(x1, x0 + 1)]
        else:  # hscan
            y0, y1 = (t * h) // n, ((t + 1) * h) // n
            canvas[y0:max(y1, y0 + 1)] = f[y0:max(y1, y0 + 1)]
        used = t + 1

    if a.mode == "center":
        canvas = canvas[:, :used * a.stretch]
    elif a.mode == "hcenter":
        canvas = canvas[:used * a.stretch]
    Image.fromarray(canvas).save(a.output)
    print(f"OK -> {a.output}  {canvas.shape[1]}x{canvas.shape[0]}  ({used} fotogramas, modo {a.mode})")


if __name__ == "__main__":
    main()
