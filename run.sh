#!/usr/bin/env bash
# Script para executar a pipeline do QuantAI usando o ambiente virtual (.venv) correto

# Obter o diretório onde o script está localizado
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verificar se existe o python do virtual env
if [ -f "$DIR/.venv/bin/python" ]; then
    # Executa usando o python do ambiente virtual
    "$DIR/.venv/bin/python" "$DIR/main.py" "$@"
else
    # Fallback para o python3 do sistema
    python3 "$DIR/main.py" "$@"
fi
