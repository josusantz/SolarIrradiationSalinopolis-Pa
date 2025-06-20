#!/bin/bash

# Obtém o número máximo de processadores disponíveis
NUM_CORES=$(nproc)

# Executa os scripts Python em paralelo utilizando os processadores disponíveis
# taskset -c define quais CPUs podem ser usadas

echo "Iniciando os scripts Python em paralelo..."

taskset -c 0-$((NUM_CORES-1)) python3 -m SimpleRNN.py &
taskset -c 0-$((NUM_CORES-1)) python3 -m GRU_RNN.py   &
taskset -c 0-$((NUM_CORES-1)) python3 -m LSTM_RNN.py  &

# Aguarda todos os processos terminarem
wait

echo "Todos os scripts foram executados."
