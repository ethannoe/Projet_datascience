#!/bin/bash
# Lance les services du projet avec le bon environnement Python (Anaconda 3.13).
# Usage : bash start.sh [dashboard|api|train|all]

PYTHON="/opt/anaconda3/bin/python"

case "${1:-dashboard}" in
  dashboard)
    echo "Lancement du dashboard → http://localhost:8501"
    $PYTHON -m streamlit run dashboard/app.py --server.port 8501
    ;;
  api)
    echo "Lancement de l'API REST → http://localhost:8000/docs"
    $PYTHON -m uvicorn api.main:app --reload --port 8000
    ;;
  train)
    echo "Entraînement des modèles..."
    $PYTHON train.py
    ;;
  all)
    echo "Lancement de l'API en arrière-plan puis du dashboard..."
    $PYTHON -m uvicorn api.main:app --reload --port 8000 &
    sleep 2
    $PYTHON -m streamlit run dashboard/app.py --server.port 8501
    ;;
  *)
    echo "Usage : bash start.sh [dashboard|api|train|all]"
    exit 1
    ;;
esac
