"""
Pipeline completo de segmentacao + OCR de placas usando EASYOCR.

Na primeira execucao o EasyOCR baixa automaticamente os modelos (~100 MB)
para ~/.EasyOCR -- e preciso conexao com a internet apenas nessa vez.

Uso:
    python run_easyocr.py                    # roda sobre dataset/placas_exercicio
    python run_easyocr.py --gpu              # usa CUDA, se disponivel
    python run_easyocr.py --dataset CAM      # roda sobre outra pasta
    python run_easyocr.py --sem-artefatos    # nao grava recortes/debug
    python run_easyocr.py --detector haar    # detecta a placa com Haar Cascade
    python run_easyocr.py --sem-roi          # OCR direto na imagem inteira

Saida (execucao de referencia, --detector classico e com ROI):
    resultados/easyocr/placas_easyocr.txt      (resultado consolidado)
    resultados/easyocr/resultados_easyocr.csv
    resultados/easyocr/resultados_easyocr.json
    resultados/easyocr/por_imagem/<n>.txt

Saida das variacoes (nunca sobrescreve a execucao de referencia):
    resultados/experimentos/deteccao_haar/easyocr/
    resultados/experimentos/sem_roi/easyocr/
"""

import argparse
from pathlib import Path

from src import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="pasta com as imagens (padrao: dataset/placas_exercicio)")
    parser.add_argument("--gpu", action="store_true",
                        help="usar GPU (CUDA) em vez de CPU")
    parser.add_argument("--sem-artefatos", action="store_true",
                        help="nao gravar recortes e imagens de debug")
    parser.add_argument("--detector", choices=("classico", "haar", "ambos"),
                        default="classico",
                        help="detector de placa (padrao: classico). "
                             "'haar' usa Haar Cascade (Viola-Jones)")
    parser.add_argument("--sem-roi", action="store_true",
                        help="OCR direto na imagem inteira, sem deteccao nem "
                             "recorte (linha de base da segmentacao)")
    args = parser.parse_args()

    pipeline.executar(
        motor="easyocr",
        diretorio=args.dataset,
        salvar_artefatos=not args.sem_artefatos,
        detector=args.detector,
        sem_roi=args.sem_roi,
        usar_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
