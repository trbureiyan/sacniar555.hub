#!/usr/bin/env python3
"""Pixel stretch con sujeto protegido: el sujeto (el ganso) queda intacto y el
resto del cuadro se sustituye por píxeles de una sola columna/fila estirados.

Modos:
  hband   franja vertical intacta alrededor del sujeto; a izquierda y derecha
          cada fila repite el color de la columna del borde de la franja
          (rayas horizontales que "salen" del sujeto).
  cutout  todo el fondo se convierte en rayas horizontales tomadas de unas pocas
          columnas de referencia (textura de "peine"); el sujeto se pega encima
          con una máscara suave, como recortado.
  drip    goteo vertical: en cada columna, desde una fila de inicio, el píxel se
          repite hacia abajo (o arriba) una longitud variable; el sujeto queda
          por delante.

Entrada: imagen (PNG/JPG) o video. Con video, el sujeto se detecta en cada
fotograma y la caja se suaviza en el tiempo para que no tiemble.

Detección automática (--subject white): píxeles claros y poco saturados,
erosión para borrar reflejos finos, y se queda la mancha más grande.
Con --box x,y,w,h (en px de la imagen de trabajo) se fija la caja a mano.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffio import VideoWriter, probe, read_frames  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi")


# ---------------------------------------------------------------- detección --
def _components(m):
    """Etiquetado 4-conexo por BFS (sin scipy). m es bool pequeño. Devuelve listas de (y,x)."""
    h, w = m.shape
    seen = np.zeros_like(m)
    comps = []
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        stack, comp = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        comps.append(comp)
    return comps


def detect_white_subject(f, thresh=165, sat=0.22, erode=5, scale=None, min_px=12, prev=None,
                         grow=5, bg_dist=60):
    """Busca el sujeto blanco. Devuelve (máscara suave 0..1, caja x0,y0,x1,y1) o (None, None).

    Puntuación de cada mancha = área * compacidad^2, penalizada por la distancia a
    'prev' (centro del fotograma anterior). Los destellos del agua son ralos (poca
    compacidad) y saltan de sitio; el ganso es sólido y se mueve poco.
    """
    rgb = f.astype(np.float32)
    if scale is None:            # rejilla proporcional a la resolución (4 px a 960, 8 px a 1920)
        scale = max(2, round(f.shape[1] / 240))
    mx, mn = rgb.max(2), rgb.min(2)
    white = (mn > thresh) & ((mx - mn) / (mx + 1) < sat)
    h, w = white.shape
    # densidad de blancos por celda de scale x scale (área media)
    dens = white[:h // scale * scale, :w // scale * scale].reshape(
        h // scale, scale, w // scale, scale).mean((1, 3))
    small = dens > 0.5
    im = Image.fromarray((small * 255).astype(np.uint8))
    small = np.asarray(im.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0
    best, best_score = None, 0.0
    diag = np.hypot(*small.shape)
    for comp in _components(small):
        if len(comp) < min_px:
            continue
        ys, xs = np.array(comp).T
        bw, bh = np.ptp(xs) + 1, np.ptp(ys) + 1
        fill = len(comp) / (bw * bh)
        score = len(comp) * fill ** 2
        if prev is not None:
            d = np.hypot(xs.mean() - prev[0] / scale, ys.mean() - prev[1] / scale) / diag
            score *= np.exp(-d / 0.08)
        if score > best_score:
            best, best_score = (ys, xs), score
    if best is None:
        return None, None
    ys, xs = best
    sm = np.zeros_like(small)
    sm[ys, xs] = True
    # zona de búsqueda: la mancha crecida ~grow celdas (recupera cuello, cabeza, pico)
    grown = Image.fromarray((sm * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * grow + 1))
    region = np.asarray(grown.resize((w // scale * scale, h // scale * scale), Image.NEAREST)) > 0
    region = np.pad(region, ((0, h - region.shape[0]), (0, w - region.shape[1])))
    # dentro de la zona: todo lo que se aleja del color típico del fondo
    bg = np.median(rgb[::8, ::8].reshape(-1, 3), axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(2))
    m = Image.fromarray(((region & (dist > bg_dist)) * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))   # quita motas
    m = m.filter(ImageFilter.GaussianBlur(1.2))
    mm = np.asarray(m) > 128
    if mm.any():
        yy, xx = np.nonzero(mm)
        box = (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1)
        return np.asarray(m, np.float32) / 255.0, box
    box = (xs.min() * scale, ys.min() * scale, (xs.max() + 1) * scale, (ys.max() + 1) * scale)
    return np.asarray(m, np.float32) / 255.0, box


# ----------------------------------------------------------------- efectos --
def band_rows(col, block):
    """Promedia una columna (h,3) en bloques de 'block' filas -> rayas más gruesas."""
    if block <= 1:
        return col
    h = col.shape[0]
    pad = (-h) % block
    c = np.concatenate([col, np.repeat(col[-1:], pad, 0)]).astype(np.float32)
    c = c.reshape(-1, block, 3).mean(1)
    return np.repeat(c, block, 0)[:h].astype(np.uint8)


def fx_hband(f, box, pad, block, inset=0):
    h, w, _ = f.shape
    x0 = max(0, box[0] - pad)
    x1 = min(w - 1, box[2] + pad)
    out = f.copy()
    left = band_rows(f[:, min(x0 + inset, w - 1)], block)
    right = band_rows(f[:, max(x1 - inset, 0)], block)
    out[:, :x0] = left[:, None]
    out[:, x1 + 1:] = right[:, None]
    return out


def fx_cutout(f, mask, ref, comb, block):
    h, w, _ = f.shape
    xr = int(ref * (w - comb))
    cols = np.stack([band_rows(f[:, xr + k], block) for k in range(comb)], 1)  # (h,comb,3)
    bg = np.tile(cols, (1, w // comb + 1, 1))[:, :w]
    a = mask[..., None]
    return (f * a + bg * (1 - a)).astype(np.uint8)


class DripField:
    """Inicio y largo de cada goteo por columna. Se genera una vez (semilla fija)
    para que en video los goteos sean estables y no parpadeen."""

    def __init__(self, w, h, density, seed, min_len, max_len, max_group=8):
        rng = np.random.default_rng(seed)
        # columnas agrupadas en "cintas" de 1..max_group px que comparten parámetros
        gid, x = np.zeros(w, int), 0
        while x < w:
            g = int(rng.integers(1, max_group + 1))
            gid[x:x + g] = gid[x - 1] + 1 if x else 0
            x += g
        n = gid.max() + 1
        coarse = np.interp(np.arange(n), np.linspace(0, n, 16), rng.random(16))
        act = (coarse * 0.55 + rng.random(n) * 0.45) > (1 - density)
        jitter = rng.random(n)
        length = (min_len + rng.random(n) ** 1.5 * (max_len - min_len)) * h
        self.active, self.jitter, self.length = act[gid], jitter[gid], length[gid]
        self.gid = gid


def fx_drip(f, field, mask, up=False, from_subject=True, zone=0.75):
    h, w, _ = f.shape
    out = f.copy()
    rows = np.arange(h)[:, None]
    # inicio = fila más brillante de cada cinta dentro de la zona superior
    lum = f[: int(h * zone)].astype(np.float32).mean(2)
    lum = lum + field.jitter[None] * 20          # desempata con algo de azar estable
    s = lum.argmax(0)
    # promedio del inicio dentro de la cinta para que sea un bloque recto
    s = (np.bincount(field.gid, s) / np.bincount(field.gid))[field.gid].astype(int)
    active = field.active.copy()
    if from_subject and mask is not None and mask.max() > 0:
        cols = mask.max(0) > 0.5                 # columnas donde está el sujeto
        low = h - 1 - np.argmax(mask[::-1] > 0.5, axis=0)  # fila inferior del sujeto
        s = np.where(cols, low - 2, s)
        active = active | cols
    if up:
        e = (s - field.length).astype(int).clip(0, h - 1)
        sel = (rows <= s) & (rows >= e) & active
    else:
        e = (s + field.length).astype(int).clip(0, h - 1)
        sel = (rows >= s) & (rows <= e) & active
    src = f[s, np.arange(w)]                      # color del píxel de inicio por columna
    out = np.where(sel[..., None], src[None], out)
    if mask is not None:
        a = mask[..., None]
        out = f * a + out * (1 - a)
    return out.astype(np.uint8)


def grade_cyber(img, strength=1.0):
    """Contraste + saturación + virado: sombras verde-azuladas, luces magenta."""
    x = img.astype(np.float32) / 255.0
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    x = lum + (x - lum) * (1 + 0.8 * strength)            # saturación
    x = 0.5 + (x - 0.5) * (1 + 0.5 * strength)            # contraste
    shadow = np.array([-0.02, 0.10, 0.14], np.float32)    # teal
    light = np.array([0.16, -0.04, 0.12], np.float32)     # magenta
    x = x + strength * ((1 - lum) * shadow + lum * light)
    return (x.clip(0, 1) * 255).astype(np.uint8)


def subject_alpha(mask, box, shape, mode, pad):
    """Zona que NO se colorea: la franja en hband, la máscara en el resto."""
    h, w = shape
    if mode == "hband":
        a = np.zeros((h, w), np.float32)
        a[:, max(0, box[0] - pad):min(w, box[2] + pad + 1)] = 1
        return a
    return mask


# -------------------------------------------------------------------- main --
def parse_box(s):
    x, y, w, h = (int(v) for v in s.split(","))
    return (x, y, x + w, y + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="imagen o video")
    ap.add_argument("output", help="PNG (si la entrada es imagen o se usa --frame-at) o MP4")
    ap.add_argument("--mode", choices=["hband", "cutout", "drip"], default="hband")
    ap.add_argument("--width", type=int, default=960, help="ancho de trabajo. Def: 960")
    ap.add_argument("--frame-at", type=float, default=None,
                    help="con video: extrae solo el fotograma de este segundo y guarda PNG")
    ap.add_argument("--start", type=float, default=None, help="video: segundo inicial")
    ap.add_argument("--duration", type=float, default=None, help="video: duración")
    ap.add_argument("--box", type=parse_box, default=None, help="caja manual x,y,w,h")
    ap.add_argument("--thresh", type=int, default=165, help="umbral de 'blanco' 0-255. Def: 165")
    ap.add_argument("--pad", type=int, default=6, help="hband: margen extra de la franja (px)")
    ap.add_argument("--block", type=int, default=1,
                    help="promedia filas en bloques de N px -> rayas más gruesas. Def: 1")
    ap.add_argument("--ref", type=float, default=0.15,
                    help="cutout: posición 0..1 de las columnas de referencia. Def: 0.15")
    ap.add_argument("--comb", type=int, default=3,
                    help="cutout: nº de columnas que se repiten (textura peine). Def: 3")
    ap.add_argument("--density", type=float, default=0.55, help="drip: fracción de columnas que gotean")
    ap.add_argument("--min-len", type=float, default=0.05, help="drip: largo mínimo (fracción del alto)")
    ap.add_argument("--max-len", type=float, default=0.9, help="drip: largo máximo (fracción del alto)")
    ap.add_argument("--up", action="store_true", help="drip: gotea hacia arriba")
    ap.add_argument("--seed", type=int, default=7, help="drip: semilla del patrón")
    ap.add_argument("--inset", type=int, default=0,
                    help="hband: toma la columna N px DENTRO de la franja (rayos que salen del sujeto)")
    ap.add_argument("--no-subject-drip", action="store_true",
                    help="drip: no hacer gotear al sujeto (por defecto gotea desde su borde inferior)")
    ap.add_argument("--grade", type=float, default=0.0,
                    help="virado cyberpunk del fondo, 0 (nada) .. 1.5. El sujeto no se toca")
    ap.add_argument("--seed-xy", type=lambda v: tuple(float(x) for x in v.split(",")), default=None,
                    help="x,y aproximados del sujeto en el primer fotograma (ayuda al seguimiento)")
    ap.add_argument("--smooth", type=float, default=0.25,
                    help="video: suavizado de la caja (0..1, menor = más suave). Def: 0.25")
    ap.add_argument("--debug", action="store_true", help="dibuja la caja/máscara detectada")
    a = ap.parse_args()

    is_video = a.input.lower().endswith(VIDEO_EXT)
    if not is_video:
        img = Image.open(a.input).convert("RGB")
        if a.width and img.width != a.width:
            img = img.resize((a.width, round(img.height * a.width / img.width / 2) * 2), Image.LANCZOS)
        frames = iter([np.asarray(img)])
        single = True
    elif a.frame_at is not None:
        frames = read_frames(a.input, a.width, a.frame_at, 0.05)
        single = True
    else:
        frames = read_frames(a.input, a.width, a.start, a.duration)
        single = False

    writer, field, box_s, n = None, None, None, 0
    for f in frames:
        h, w, _ = f.shape
        prev = None
        if box_s is not None:
            prev = ((box_s[0] + box_s[2]) / 2, (box_s[1] + box_s[3]) / 2)
        elif a.seed_xy:
            prev = a.seed_xy
        mask, box = detect_white_subject(f, a.thresh, prev=prev)
        if a.box:
            box = a.box
            if mask is None:
                mask = np.zeros((h, w), np.float32)
                mask[box[1]:box[3], box[0]:box[2]] = 1
        if box is None:
            box = box_s or (w // 2 - 40, h // 2 - 40, w // 2 + 40, h // 2 + 40)
            mask = np.zeros((h, w), np.float32)
        # suavizado temporal de la caja (media exponencial)
        box_s = box if box_s is None else tuple(
            p + a.smooth * (q - p) for p, q in zip(box_s, box))
        bi = tuple(int(round(v)) for v in box_s)

        if a.mode == "hband":
            out = fx_hband(f, bi, a.pad, a.block, a.inset)
        elif a.mode == "cutout":
            out = fx_cutout(f, mask, a.ref, a.comb, a.block)
        else:
            if field is None:
                field = DripField(w, h, a.density, a.seed, a.min_len, a.max_len)
            out = fx_drip(f, field, mask, a.up, not a.no_subject_drip)

        if a.grade:
            al = subject_alpha(mask, bi, (h, w), a.mode, a.pad)[..., None]
            out = (out * al + grade_cyber(out, a.grade) * (1 - al)).astype(np.uint8)

        if a.debug:
            out = out.copy()
            out[..., 0] = np.maximum(out[..., 0], (mask * 255).astype(np.uint8))
            x0, y0, x1, y1 = bi
            out[y0:y1, [x0, min(x1, w - 1)]] = (0, 255, 0)
            out[[y0, min(y1, h - 1)], x0:x1] = (0, 255, 0)

        if single:
            Image.fromarray(out).save(a.output)
            print(f"OK -> {a.output}  {w}x{h}  modo={a.mode} caja={bi}")
            return
        if writer is None:
            writer = VideoWriter(a.output, w, h, probe(a.input)[2])
        writer.write(out)
        n += 1
    if writer:
        writer.close()
        print(f"OK -> {a.output}  {n} fotogramas, modo={a.mode}")


if __name__ == "__main__":
    main()
