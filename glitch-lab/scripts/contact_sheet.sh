#!/usr/bin/env bash
# Hoja de contactos: N fotogramas repartidos uniformemente, con su tiempo impreso.
set -euo pipefail
if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Uso: $0 entrada.mp4 salida.png [columnas=6] [filas=3] [ancho_miniatura=320]"
  echo "Toma columnas*filas fotogramas equiespaciados y los monta con el filtro tile."
  exit 0
fi
IN="$1"; OUT="$2"; C="${3:-6}"; R="${4:-3}"; TW="${5:-320}"
N=$(ffprobe -v error -count_frames -select_streams v:0 \
      -show_entries stream=nb_read_frames -of csv=p=0 "$IN")
STEP=$(( N / (C * R) )); (( STEP < 1 )) && STEP=1
ffmpeg -v error -y -i "$IN" -vf \
  "scale=${TW}:-2,drawtext=text='f%{n} %{pts\:hms}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6,select='not(mod(n\,${STEP}))',tile=${C}x${R}" \
  -frames:v 1 -fps_mode passthrough "$OUT"
echo "OK -> $OUT  ($N fotogramas, 1 de cada $STEP)"
