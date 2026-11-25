"""
Pipeline completo de segmentacao + OCR de placas usando TESSERACT.

Uso:
    python run_tesseract.py                    # roda sobre dataset/placas_exercicio
    python run_tesseract.py --dataset CAM      # roda sobre outra pasta
    python run_tesseract.py --sem-artefatos    # nao grava recortes/debug
    python run_tesseract.py --detector haar    # detecta a placa com Haar Cascade
    python run_tesseract.py --sem-roi          # OCR direto na imagem inteira

Saida (execucao de referencia, --detector classico e com ROI):
    resultados/tesseract/placas_tesseract.txt      (entregavel principal)
    resultados/tesseract/resultados_tesseract.csv
    resultados/tesseract/resultados_tesseract.json
    resultados/tesseract/por_imagem/<n>.txt

Saida das variacoes (nunca sobrescreve o entregavel):
    resultados/experimentos/deteccao_haar/tesseract/
    resultados/experimentos/sem_roi/tesseract/
"""

import argparse
from pathlib import Path

from src import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="pasta com as imagens (padrao: dataset/placas_exercicio)")
    parser.add_argument("--sem-artefatos", action="store_true",
                        help="nao gravar recortes e imagens de debug")
    parser.add_argument("--detector", choices=("classico", "haar", "ambos"),
                        default="classico",
                        help="detector de placa (padrao: classico). "
                             "'haar' usa a cascata Haar da Aula 8")
    parser.add_argument("--sem-roi", action="store_true",
                        help="OCR direto na imagem inteira, sem deteccao nem "
                             "recorte (linha de base da Aula 9)")
    args = parser.parse_args()

    pipeline.executar(
        motor="tesseract",
        diretorio=args.dataset,
        salvar_artefatos=not args.sem_artefatos,
        detector=args.detector,
        sem_roi=args.sem_roi,
    )


if __name__ == "__main__":
    main()
