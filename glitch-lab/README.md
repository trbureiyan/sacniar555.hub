# glitch-lab

Slit-scan, time displacement y pixel stretch sobre un clip de Canon EOS Rebel SL3.
Solo ffmpeg + Python (numpy, Pillow). El video se lee y escribe fotograma a fotograma
por pipes de ffmpeg: nunca se carga entero en RAM.

Clip verificado con ffprobe: H.264 High, 1920x1080, 29.97 fps, **yuvj420p (rango completo)**,
BT.709, AAC estéreo 48 kHz, 16,95 s, 508 fotogramas. Las salidas se convierten a yuv420p
rango TV con BT.709 explícito.

Carpetas: `input/` copia de solo lectura del original · `work/` intermedios ·
`out/` resultados · `sheets/` hojas de contactos · `test/` clip sintético.
Los medios están en `.gitignore` y no se suben a ningún sitio.

## Scripts (todos aceptan `--help`)

| Script | Qué hace |
|---|---|
| `make_test_clip.sh [salida] [ancho]` | Clip sintético testsrc2 + mandelbrot con corte en 5 s |
| `contact_sheet.sh in.mp4 out.png [cols] [filas] [ancho]` | Hoja de contactos con número de fotograma y tiempo |
| `slitscan_image.py` | Slit-scan a PNG |
| `time_displace.py` | Slit-scan en video (cada zona con distinto retraso) |
| `pixel_stretch.py` | Pixel stretch con el ganso protegido (imagen o video) |
| `ffio.py` | Módulo común de lectura/escritura por pipes |

## Comandos usados

```bash
# Proxy de trabajo: tramo 6.5–16.5 s a 960 px
ffmpeg -ss 6.5 -t 10 -i input/MVI_9903.MP4 -an -vf "scale=960:-2:in_color_matrix=bt709:out_color_matrix=bt709:out_range=tv,format=yuv420p" -c:v libx264 -crf 14 work/tramo_960.mp4

# B) slit-scan imagen
python3 scripts/slitscan_image.py work/tramo_960.mp4 out/B1_center_stretch4.png --mode center --stretch 4
python3 scripts/slitscan_image.py work/tramo_960.mp4 out/B2_scan.png --mode scan
python3 scripts/slitscan_image.py work/tramo_960.mp4 out/B3_hcenter_stretch3.png --mode hcenter --stretch 3

# C) time displacement
python3 scripts/time_displace.py work/tramo_960.mp4 out/C_radial_d60.mp4 --shape radial --max-delay 60

# E/G) pixel stretch (fotograma 4.5 s)
python3 scripts/pixel_stretch.py input/MVI_9903.MP4 out/G1.png --frame-at 4.5 --mode cutout --ref 0.08 --comb 4 --block 3 --width 1920
python3 scripts/pixel_stretch.py input/MVI_9903.MP4 out/E3.png --frame-at 8.7 --mode hband --pad 2 --grade 1.0

# F/G) pixel stretch en video con seguimiento del ganso
python3 scripts/pixel_stretch.py input/MVI_9903.MP4 out/G2.mp4 --start 3.5 --duration 6.5 --seed-xy 360,135 --mode cutout --ref 0.08 --comb 4 --block 3
```

## Parámetros

**slitscan_image.py** — `--mode` center (franja vertical fija, X = tiempo) · scan (la franja
barre el cuadro) · hcenter / hscan (lo mismo en horizontal). `--slit` grosor de la franja
leída. `--stretch` px que ocupa cada fotograma en la salida (alargado). `--pos` posición
0..1 de la franja fija. `--width/--start/--duration` recorte al leer.

**time_displace.py** — `--shape` horizontal | vertical | radial (forma del mapa de retraso).
`--max-delay` retraso máximo en fotogramas (memoria ≈ (max+1)·ancho·alto·3 bytes).
`--invert` invierte el mapa. `--steps N` bandas duras en vez de degradado.
`--save-map` guarda el mapa como PNG. `--crf` calidad x264.

**pixel_stretch.py** — `--mode` hband (franja intacta, rayas desde sus bordes) · cutout
(fondo entero en rayas, ganso recortado encima) · drip (goteo vertical).
`--frame-at S` saca un PNG de ese segundo. `--width` ancho de trabajo (960 iterar, 1920 final).
`--pad` margen de la franja. `--inset N` lee la columna N px dentro de la franja (rayos que
salen del ganso). `--block N` promedia filas en bloques (rayas más gruesas). `--ref` y
`--comb` posición y nº de columnas que se repiten en cutout (textura de peine).
`--density/--min-len/--max-len/--up/--seed` controlan el goteo. `--grade 0..1.5` virado
cyberpunk solo del fondo. `--seed-xy x,y` pista de dónde está el ganso al empezar (evita que
el seguimiento se vaya a los destellos del agua). `--smooth` suavizado de la caja en video.
`--box x,y,w,h` caja manual. `--thresh` umbral de blanco. `--debug` dibuja máscara y caja.

## Limitaciones conocidas

- La detección busca "blanco poco saturado": sirve para este ganso, no para cualquier sujeto.
- Máscara: semilla blanca a nivel de píxel, pasos cortos por grises claros para cruzar la
  cara gris hasta la cabeza, crecimiento solo por tonos naranja/rosado para el pico, y una
  "cerca" del 5 % del ancho para que no se derrame. Funciona bien con el ganso solo
  (fotograma 4,5 s, primeros ~2 s del video).
- En video, cuando los gansos se tocan la máscara se contagia al ganso gris, y con agua muy
  rizada o espuma (último tramo) se cuelan manchas de agua.
- El datamosh se probó y se descartó: el codificador MPEG-4 refrescaba la imagen en ~10
  fotogramas tras el corte, y el estilo buscado es otro.
