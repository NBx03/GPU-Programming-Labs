#!/bin/bash
# Запуск ЛР1 на кластере. Пример: ./run_lab1.sh sm_80
# Планировщик: уточнить по инструкции кластера. Если Slurm -- раскомментировать/поправить строки ниже
# и отправить:  sbatch run_lab1.sh sm_80
##SBATCH -p compclass          # TODO: проверить имя очереди по инструкции
##SBATCH --gres=gpu:1          # TODO: проверить синтаксис запроса GPU
##SBATCH -t 00:30:00
##SBATCH -o results/slurm_%j.out
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
ARCH="${1:-}"

make ${ARCH:+ARCH=$ARCH}

{ echo "== nvidia-smi"; nvidia-smi; echo; echo "== nvcc"; nvcc --version; } > results/env.txt 2>&1 || true

# основной запуск: разные размеры блока
for B in 128 256 512 1024; do
  ./sin_precision --block "$B" --csv "results/main_block${B}.csv" | tee "results/main_block${B}.txt"
done
# весь массив целиком (если хватает памяти GPU)
./sin_precision --full --block 256 --csv results/full.csv | tee results/full.txt
# -use_fast_math: sinf превращается в __sinf
./sin_precision_fastmath --block 256 --no-micro --csv results/fastmath.csv | tee results/fastmath.txt
