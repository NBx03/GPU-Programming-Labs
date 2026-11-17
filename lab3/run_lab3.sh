#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results examples
ARCH="${1:-}"

make ${ARCH:+ARCH=$ARCH}

INPUT="../lab2/examples/input.png"
if [ ! -f "$INPUT" ]; then
    echo "Создаем тестовую картинку..."
    ../lab2/filters --gen 2048 "$INPUT"
fi

echo "=== 1. Обычный запуск на 2 GPU ==="
./filters_2gpu "$INPUT" examples/out_blur_2gpu.png blur --radius 3 | tee results/blur_2gpu.txt
./filters_2gpu "$INPUT" examples/out_sobel_2gpu.png sobel | tee results/sobel_2gpu.txt
./filters_2gpu "$INPUT" examples/out_median_2gpu.png median | tee results/median_2gpu.txt

echo -e "\n=== 2. Профилирование Nsight Systems (nsys) ==="
if command -v nsys &>/dev/null; then
    nsys profile --stats=true --force-overwrite=true -o results/nsys_2gpu_blur \
        ./filters_2gpu "$INPUT" examples/out_blur_nsys.png blur --radius 3 | tee results/nsys_summary.txt
    echo "Отчет профилирования сохранен в results/nsys_2gpu_blur.nsys-rep"
else
    echo "nsys не найден в PATH, пропускаем профилирование"
fi
