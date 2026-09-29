#!/bin/bash
# Запуск ЛР2 на кластере. Пример: ./run_lab2.sh sm_80 [путь/к/картинке.png]
# Планировщик: уточнить по инструкции кластера. Если Slurm -- раскомментировать/поправить строки ниже
# и отправить:  sbatch run_lab2.sh sm_80
##SBATCH -p compclass          # TODO: проверить имя очереди по инструкции
##SBATCH --gres=gpu:1          # TODO: проверить синтаксис запроса GPU
##SBATCH -c 8
##SBATCH -t 01:00:00
##SBATCH -o results/slurm_%j.out
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results examples
ARCH="${1:-}"
INPUT="${2:-}"

make ${ARCH:+ARCH=$ARCH}

{ echo "== nvidia-smi"; nvidia-smi; echo; echo "== nvcc"; nvcc --version; echo; echo "== cpu"; lscpu | head -20; } \
  > results/env.txt 2>&1 || true

# входное изображение: свой файл или сгенерированное
if [ -z "$INPUT" ]; then
  INPUT=examples/input.png
  [ -f "$INPUT" ] || ./filters --gen 1024 "$INPUT"
fi

# примеры "до/после" и проверка корректности GPU против CPU
./filters "$INPUT" examples/out_blur.png   blur   --radius 3 | tee results/single_blur.txt
./filters "$INPUT" examples/out_sobel.png  sobel             | tee results/single_sobel.txt
./filters "$INPUT" examples/out_median.png median            | tee results/single_median.txt

# бенчмарк накладных расходов (размеры 512..4096; добавить 8192 при наличии памяти и времени)
./filters --bench --sizes 512,2048,4096 --radii 1,2,4,7 --reps 5 --csv results/bench.csv | tee results/bench.txt
