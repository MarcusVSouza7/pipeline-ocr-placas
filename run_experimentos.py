"""
Roda as duas comparacoes extras exigidas pelas aulas 8 e 9, na ordem certa.

  (1) DETECTOR  -- Haar Cascade (Aula 8) x detector classico do projeto.
  (2) ROI       -- OCR com ROI x OCR direto na imagem inteira (Aula 9).

Pre-requisito: `python run_tesseract.py` e `python run_easyocr.py` (as
execucoes de referencia, que sao o lado "classico" e o lado "com ROI" das duas
comparacoes).

Uso:
    python run_experimentos.py              # roda tudo e gera os relatorios
    python run_experimentos.py --so-haar    # so a comparacao de detector
    python run_experimentos.py --so-roi     # so a comparacao de ROI
    python run_experimentos.py --relatorios # nao reexecuta, so regera os .txt/.md

Saida:
    resultados/experimentos/deteccao_haar/<motor>/
    resultados/experimentos/sem_roi/<motor>/
    resultados/comparacao/deteccao_haar_x_classico.txt | .md
    resultados/comparacao/roi_x_sem_roi.txt | .md

Atencao: nada aqui sobrescreve os entregaveis de `resultados/tesseract/` e
`resultados/easyocr/`.
"""

import argparse

from src import comparar_experimentos, pipeline


MOTORES = ("tesseract", "easyocr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-haar", action="store_true",
                        help="rodar apenas a comparacao de detector")
    parser.add_argument("--so-roi", action="store_true",
                        help="rodar apenas a comparacao de ROI")
    parser.add_argument("--relatorios", action="store_true",
                        help="nao reexecutar os pipelines, so regerar relatorios")
    parser.add_argument("--gpu", action="store_true",
                        help="usar GPU (CUDA) no EasyOCR")
    args = parser.parse_args()

    fazer_haar = not args.so_roi
    fazer_roi = not args.so_haar

    if not args.relatorios:
        if fazer_haar:
            for motor in MOTORES:
                print("\n### DETECTOR HAAR - %s\n" % motor.upper())
                extras = {"usar_gpu": args.gpu} if motor == "easyocr" else {}
                pipeline.executar(motor=motor, detector="haar",
                                  salvar_artefatos=True, **extras)

        if fazer_roi:
            for motor in MOTORES:
                print("\n### SEM ROI (OCR direto) - %s\n" % motor.upper())
                extras = {"usar_gpu": args.gpu} if motor == "easyocr" else {}
                pipeline.executar(motor=motor, sem_roi=True,
                                  salvar_artefatos=False, **extras)

    print("\n" + "=" * 72)
    print("RELATORIOS")
    print("=" * 72)
    if fazer_haar:
        comparar_experimentos.comparar_detector()
    if fazer_roi:
        comparar_experimentos.comparar_roi()


if __name__ == "__main__":
    main()
