#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
ARCH="${1:-}"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== 1. Сборка программы ЛР4 ==="
make clean
make ${ARCH:+ARCH=$ARCH}

echo -e "\n=== 2. Сравнение производительности (512x512, 1024x1024, 2048x2048) ==="
for S in 512 1024 2048; do
    echo "--- Размер матрицы: ${S}x${S} ---"
    ./matmul --size "$S" --reps 5 | tee -a results/benchmarks.txt
done

echo -e "\n=== 3. Профилирование Nsight Systems (nsys) ==="
if command -v nsys &>/dev/null; then
    nsys profile --stats=true --force-overwrite=true -o results/nsys_matmul \
        ./matmul --size 1024 --reps 1 | tee results/nsys_summary.txt
    echo "Отчет nsys сохранен в results/nsys_matmul.nsys-rep"
else
    echo "nsys не найден в PATH"
fi

echo -e "\n=== 4. Профилирование Nsight Compute (ncu) ==="
if command -v ncu &>/dev/null; then
    echo "Запуск ncu для сравнения Naive и Shared версий (размер 512x512)..."
    # Запускаем сбор базовых метрик SOL (Speed of Light) и пропускной способности памяти
    set +e
    ncu --set basic --force-overwrite -o results/ncu_matmul \
        ./matmul --size 512 --reps 1 > results/ncu_summary.txt 2>&1
    NCU_STATUS=$?
    set -e

    if [ $NCU_STATUS -eq 0 ]; then
        echo "Отчет ncu успешно сформирован: results/ncu_matmul.ncu-rep"
    else
        echo "Внимание: ncu завершился с предупреждением/ошибкой прав (ERR_NVGPUCTRPERM)."
        echo "Текст ошибки сохранен в results/ncu_summary.txt (см. ЧАСТЬ 3 инструкции)."
    fi
else
    echo "ncu не найден в PATH"
fi

echo -e "\nВсе тесты завершены! Результаты сохранены в папке lab4/results/"
