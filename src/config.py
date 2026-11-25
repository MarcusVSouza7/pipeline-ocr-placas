"""
Configurações globais do pipeline de ALPR (Automatic License Plate Recognition).

Centraliza caminhos, parâmetros de segmentação e parâmetros de OCR para que os
dois pipelines (Tesseract e EasyOCR) compartilhem exatamente as mesmas etapas de
pré-processamento -- condição necessária para uma comparação justa entre as duas
ferramentas de OCR.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos do projeto
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent.parent

DIR_DATASET = RAIZ / "dataset" / "placas_exercicio"
DIR_DADOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_HAAR = RAIZ / "haar_xml"               # cascatas Haar

DIR_SAIDA_TESSERACT = DIR_RESULTADOS / "tesseract"
DIR_SAIDA_EASYOCR = DIR_RESULTADOS / "easyocr"
DIR_SAIDA_COMPARACAO = DIR_RESULTADOS / "comparacao"
DIR_CROPS = DIR_RESULTADOS / "crops"       # placas segmentadas e retificadas
DIR_DEBUG = DIR_RESULTADOS / "debug"       # imagens anotadas de cada etapa

# Saidas dos experimentos (detector Haar e baseline sem ROI). Ficam separadas
# de DIR_SAIDA_* para que um experimento nunca sobrescreva a referencia.
DIR_EXPERIMENTOS = DIR_RESULTADOS / "experimentos"


def dir_saida(motor, detector="classico", sem_roi=False):
    """
    Diretorio de saida de uma execucao.

    A execucao de referencia (detector classico, com ROI) grava em
    `resultados/<motor>/`. Qualquer variacao grava em
    `resultados/experimentos/<variacao>/<motor>/`.
    """
    if sem_roi:
        return DIR_EXPERIMENTOS / "sem_roi" / motor
    if detector != "classico":
        return DIR_EXPERIMENTOS / ("deteccao_" + detector) / motor
    return DIR_SAIDA_TESSERACT if motor == "tesseract" else DIR_SAIDA_EASYOCR

ARQ_GABARITO = DIR_DADOS / "ground_truth.csv"

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --------------------------------------------------------------------------- #
# Segmentação da placa
# --------------------------------------------------------------------------- #
# Placa Mercosul: 400 x 130 mm -> razão de aspecto nominal ~3.08.
# A faixa aceita é folgada para tolerar perspectiva, recorte e placas antigas.
RAZAO_ASPECTO_MIN = 1.7
RAZAO_ASPECTO_MAX = 6.5
RAZAO_ASPECTO_IDEAL = 3.08

# Área da placa em relação à área total da imagem (filtro de candidatos).
AREA_RELATIVA_MIN = 0.0025
AREA_RELATIVA_MAX = 0.60

# Largura em que a imagem é normalizada antes da detecção (acelera e uniformiza
# os parâmetros morfológicos independentemente da resolução original).
LARGURA_TRABALHO = 1024

# Faixa azul do cabeçalho Mercosul em HSV (OpenCV: H 0-179, S 0-255, V 0-255).
HSV_AZUL_MIN = (95, 70, 40)
HSV_AZUL_MAX = (130, 255, 255)

# --------------------------------------------------------------------------- #
# Haar Cascade -- detector alternativo, usado por `detector_haar.py`
# --------------------------------------------------------------------------- #
# Os tres parametros de `detectMultiScale`:
#   scaleFactor  -- reducao da imagem em cada escala (menor = mais preciso e
#                   mais lento). 1.05 em vez de 1.1 porque as placas do
#                   dataset variam muito de tamanho.
#   minNeighbors -- deteccoes vizinhas necessarias (menor = mais falsos
#                   positivos). 3 em vez de 5: a cascata russa responde fraco
#                   em placa Mercosul e 5 perde placas demais. Os falsos
#                   positivos resultantes sao filtrados pela pontuacao
#                   heuristica do pipeline.
#   minSize      -- menor placa aceita, em px na largura de trabalho.
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS = 3
HAAR_MIN_SIZE = (60, 20)

# --------------------------------------------------------------------------- #
# Pré-processamento / OCR
# --------------------------------------------------------------------------- #
ALTURA_OCR = 160          # altura em px para a qual o recorte é reamostrado
MARGEM_OCR = 24           # borda branca adicionada (Tesseract exige "quiet zone")

ALFABETO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TAMANHO_PLACA = 7         # AAA0A00 (Mercosul) / AAA0000 (padrão antigo)

# Caminho do executável do Tesseract no Windows (detectado automaticamente em
# ocr_tesseract.py caso não esteja no PATH).
CAMINHOS_TESSERACT = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def garantir_diretorios() -> None:
    """Cria todos os diretórios de saída utilizados pelos pipelines."""
    for d in (
        DIR_DADOS,
        DIR_RESULTADOS,
        DIR_SAIDA_TESSERACT,
        DIR_SAIDA_EASYOCR,
        DIR_SAIDA_COMPARACAO,
        DIR_CROPS,
        DIR_DEBUG,
    ):
        d.mkdir(parents=True, exist_ok=True)
