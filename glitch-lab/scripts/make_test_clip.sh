#!/usr/bin/env bash
# Genera un clip sintético para validar los scripts: 5 s de testsrc2 + 5 s de
# mandelbrot (fractal con zoom), con un corte de escena duro en t=5 s.
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Uso: $0 [salida.mp4] [ancho]   (por defecto test/synthetic.mp4, 960)"
  echo "Crea 10 s a 29.97 fps: testsrc2 (0-5 s) y mandelbrot (5-10 s) con corte duro."
  exit 0
fi
OUT="${1:-test/synthetic.mp4}"
W="${2:-960}"; H=$(( W * 9 / 16 / 2 * 2 ))
mkdir -p "$(dirname "$OUT")"
ffmpeg -v error -y \
  -f lavfi -i "testsrc2=s=${W}x${H}:r=30000/1001:d=5" \
  -f lavfi -i "mandelbrot=s=${W}x${H}:r=30000/1001,trim=duration=5" \
  -filter_complex "[0:v][1:v]concat=n=2:v=1,format=yuv420p" \
  -c:v libx264 -crf 16 -preset fast -colorspace bt709 -color_primaries bt709 \
  -color_trc bt709 -color_range tv "$OUT"
echo "OK -> $OUT"
