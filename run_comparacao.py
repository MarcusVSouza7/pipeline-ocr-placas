"""
Compara os resultados dos dois pipelines de OCR.

Pre-requisito: rodar antes `python run_tesseract.py` e `python run_easyocr.py`.

Uso:
    python run_comparacao.py

Saida:
    resultados/comparacao/comparacao.txt
    resultados/comparacao/comparacao.csv
    resultados/comparacao/analise.md
"""

from src import comparar


if __name__ == "__main__":
    comparar.comparar()
