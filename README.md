# Программирование графических процессоров (CUDA) — лабораторные

| ЛР | Тема | Каталог | Срок |
|----|------|---------|------|
| 1 | Точность и скорость `sin` / `sinf` / `__sinf`, float и double | [lab1](lab1) | 13.10.2026 |
| 2 | Фильтры изображений на CUDA (размытие, Собель, медиана), накладные расходы GPU | [lab2](lab2) | 13.10.2026 |

Общий код: [common/cuda_check.h](common/cuda_check.h) (макрос `CUDA_CHECK`, таймер на `cudaEvent`).

## Сборка и запуск на кластере

Требуется `nvcc`, для ЛР2 ещё `libpng` (заголовок `png.h`) и OpenMP. Архитектуру GPU узнать так:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv    # compute_cap 8.0 -> ARCH=sm_80
```

```bash
git clone https://github.com/NBx03/GPU-Programming-Labs.git && cd GPU-Programming-Labs
./lab1/run_lab1.sh sm_80        # собирает, запускает, кладёт логи в lab1/results/
./lab2/run_lab2.sh sm_80        # то же для ЛР2, примеры картинок в lab2/examples/
```

Скрипты запускаются напрямую или через планировщик кластера (очереди `compclass`, `compclass_unstable`);
директивы планировщика в шапке скриптов закомментированы — поправить по инструкции кластера.

Локально (Windows, без GPU) проверяется только CPU-часть ЛР2 (`lab2/cpu_filters.cpp`, `lab2/filter_common.h`).
