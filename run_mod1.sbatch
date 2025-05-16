#!/bin/bash -l

# Nazwa zlecenia
#SBATCH -J cmaes_base
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks=32
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=2GB
## Maksymalny czas trwania zlecenia (format DD-HH:MM:SS)
#SBATCH --time=01:00:00 
## Parametr wyznaczający indeksy zadania tablicowego

# Grant name
#SBATCH -A plgcmaes

# Partition (plgrid for 72 hours)
#SBATCH -p plgrid

# Output files
#SBATCH --output="out/out_%j.log"
#SBATCH --error="out/err_%j.log"


set -e
eval "$(conda shell.bash hook)"
conda activate cma

echo "Running experiment"
for seed in $(seq 32 50); do
    python3 main.py --seed ${seed} --dim 30 --method mod &
done
wait
echo "All runs finished"