"""
Deteccao de placa por HAAR CASCADE -- a tecnica da Aula 8.

Este modulo existe para que o projeto implemente o detector ENSINADO em aula
(Viola-Jones / Haar Cascade, slide "EXEMPLO: DETECTAR PLACA E SALVAR ROI") e
para medi-lo contra o detector classico proprio (`plate_detector`), que usa
cor em HSV + bordas Sobel + regioes claras.

Como a comparacao e feita
-------------------------
O Haar entra no pipeline EXATAMENTE no lugar dos tres geradores de candidatos
de `plate_detector`: ele devolve candidatos no mesmo formato (os 4 cantos de
uma "pseudo-faixa"), e todo o resto -- retificacao, segmentacao de caracteres,
pre-processamento, OCR e pos-processamento -- permanece identico. Assim a
diferenca medida e atribuivel ao DETECTOR, do mesmo jeito que a comparacao
Tesseract x EasyOCR isola o motor de OCR.

Por que "russian plate number"
------------------------------
O OpenCV nao publica cascata treinada para placa brasileira; a cascata russa
e a que o professor distribuiu em "Exemplos (1).zip" e a usada no slide da
Aula 8. Ela foi treinada em placas russas -- uma unica linha de caracteres,
sem o cabecalho azul do Mercosul -- o que ja antecipa a limitacao listada no
proprio slide "LIMITACOES DO HAAR CASCADE": o detector funciona melhor com
objetos parecidos com os do treinamento.

Duas hipoteses por caixa
------------------------
`detectMultiScale` devolve um retangulo, mas nao diz O QUE ele enquadrou: em
placa Mercosul a cascata russa as vezes pega a placa inteira (com faixa azul)
e as vezes somente a linha de caracteres. Em vez de adivinhar, cada caixa gera
DOIS candidatos (uma hipotese para cada caso) e a pontuacao heuristica de
`plate_detector._pontuar` decide qual sobrevive -- a supressao de nao-maximos
descarta o outro.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as cfg


# --------------------------------------------------------------------------- #
# Carregamento das cascatas
# --------------------------------------------------------------------------- #
# A cascata de 16 estagios e mais permissiva (detecta mais, erra mais); a de
# ~20 estagios e mais conservadora. Usamos as duas e unimos os resultados,
# porque uma placa pequena ou desfocada muitas vezes so passa na primeira.
NOMES_CASCATAS = (
    "haarcascade_russian_plate_number.xml",
    "haarcascade_license_plate_rus_16stages.xml",
)

_detectores = None


def localizar_cascata(nome):
    """
    Devolve o caminho do XML da cascata, ou None se nao for encontrado.

    Procura primeiro em `haar_xml/` (os arquivos distribuidos pelo professor em
    "Exemplos (1).zip") e depois no diretorio de dados do proprio OpenCV --
    alguns wheels do `opencv-python` nao embarcam cascata nenhuma, por isso a
    copia local do projeto e a fonte principal e nao um plano B.
    """
    local = cfg.DIR_HAAR / nome
    if local.is_file():
        return local

    try:
        embarcada = cv2.data.haarcascades
    except AttributeError:                            # build sem cv2.data
        return None
    caminho = Path(embarcada) / nome
    return caminho if caminho.is_file() else None


def obter_detectores():
    """
    Carrega as cascatas uma unica vez (singleton) e devolve [(nome, detector)].

    Levanta RuntimeError se nenhuma cascata estiver disponivel -- sem o XML o
    detector nao tem como funcionar, e falhar explicitamente e melhor que
    devolver "nenhuma placa encontrada" em todas as imagens.
    """
    global _detectores
    if _detectores is not None:
        return _detectores

    carregados = []
    faltando = []
    for nome in NOMES_CASCATAS:
        caminho = localizar_cascata(nome)
        if caminho is None:
            faltando.append(nome)
            continue
        detector = cv2.CascadeClassifier(str(caminho))
        if detector.empty():                          # XML corrompido
            faltando.append(nome)
            continue
        rotulo = nome.replace("haarcascade_", "").replace(".xml", "")
        carregados.append((rotulo, detector))

    if not carregados:
        raise RuntimeError(
            "Nenhuma cascata Haar encontrada. Copie os XMLs de "
            "Exemplos (1).zip para %s. Faltando: %s"
            % (cfg.DIR_HAAR, ", ".join(faltando))
        )

    _detectores = carregados
    return _detectores


def versao():
    """Rotulo das cascatas carregadas, para registro nos relatorios."""
    return "Haar Cascade (%s)" % ", ".join(n for n, _ in obter_detectores())


# --------------------------------------------------------------------------- #
# Deteccao bruta
# --------------------------------------------------------------------------- #
def _iou(a, b):
    """Intersection over Union entre duas caixas (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def caixas(cinza):
    """
    Roda `detectMultiScale` das duas cascatas e devolve as caixas filtradas.

    Devolve uma lista de (x, y, w, h, cascata) na resolucao da imagem recebida,
    ja sem duplicatas (IoU > 0.4) e sem caixas com proporcao ou area
    incompativeis com uma placa.

    Os parametros vem de `config` e sao os mesmos do slide da Aula 8
    (scaleFactor, minNeighbors, minSize), apenas afrouxados o suficiente para
    as placas pequenas das imagens 11 e 17.
    """
    altura_img, largura_img = cinza.shape[:2]
    brutas = []

    for nome, detector in obter_detectores():
        encontradas = detector.detectMultiScale(
            cinza,
            scaleFactor=cfg.HAAR_SCALE_FACTOR,
            minNeighbors=cfg.HAAR_MIN_NEIGHBORS,
            minSize=cfg.HAAR_MIN_SIZE,
        )
        for (x, y, w, h) in encontradas:
            if h <= 0 or w <= 0:
                continue
            razao = w / float(h)
            # A cascata russa as vezes devolve a linha de caracteres (razao
            # alta) e as vezes a placa inteira (razao baixa): a faixa aceita
            # cobre os dois casos.
            if not (cfg.RAZAO_ASPECTO_MIN <= razao <= cfg.RAZAO_ASPECTO_MAX):
                continue
            area_rel = (w * h) / float(largura_img * altura_img)
            if not (cfg.AREA_RELATIVA_MIN <= area_rel <= cfg.AREA_RELATIVA_MAX):
                continue
            brutas.append((int(x), int(y), int(w), int(h), nome))

    # Caixa maior primeiro: entre duas deteccoes sobrepostas, a maior tende a
    # conter a placa inteira em vez de um pedaco dela.
    brutas.sort(key=lambda c: c[2] * c[3], reverse=True)

    finais = []
    for c in brutas:
        if all(_iou(c[:4], f[:4]) < 0.4 for f in finais):
            finais.append(c)
    return finais


# --------------------------------------------------------------------------- #
# Conversao para o formato de candidato do pipeline
# --------------------------------------------------------------------------- #
# Proporcoes verticais da placa Mercosul usadas por `plate_detector`:
# altura_da_placa = (TOPO + 1 + BASE) * altura_da_faixa_azul = 5.0 * hf.
# Logo, dada a placa inteira de altura H: hf = H / 5 e o topo da faixa fica
# TOPO * hf = 0.12 * H abaixo do topo da placa.
_HF_EM_ALTURA_PLACA = 1.0 / 5.0
_TOPO_FAIXA_EM_ALTURA_PLACA = 0.12


def candidatos(cinza):
    """
    Devolve candidatos no formato consumido por `plate_detector`:
    uma lista de (cantos_da_pseudo_faixa, origem).

    Cada caixa do Haar gera DUAS hipoteses, porque a cascata russa nao
    distingue os dois enquadramentos possiveis numa placa Mercosul:

      (1) a caixa e a PLACA INTEIRA. A faixa azul e a banda horizontal que
          comeca a 12% da altura e tem 1/5 dela.
      (2) a caixa e so a LINHA DE CARACTERES (caso tipico da placa russa, que
          nao tem cabecalho). A faixa equivalente fica ACIMA da caixa, com 30%
          da sua altura -- a mesma convencao dos detectores `bordas` e
          `regiao_clara` de `plate_detector`.

    A pontuacao heuristica e a supressao de nao-maximos escolhem entre elas.
    """
    saida = []
    for (x, y, w, h, _nome) in caixas(cinza):
        # (1) a caixa e a placa inteira
        hf = h * _HF_EM_ALTURA_PLACA
        topo = y + h * _TOPO_FAIXA_EM_ALTURA_PLACA
        saida.append((
            np.array([[x, topo], [x + w, topo],
                      [x + w, topo + hf], [x, topo + hf]], dtype=np.float32),
            "haar",
        ))

        # (2) a caixa e so a linha de caracteres
        hf2 = h * 0.30
        saida.append((
            np.array([[x, y - hf2], [x + w, y - hf2],
                      [x + w, y], [x, y]], dtype=np.float32),
            "haar",
        ))
    return saida


def desenhar(imagem, lista_caixas):
    """Imagem de debug com as caixas cruas devolvidas pelas cascatas."""
    saida = imagem.copy()
    espessura = max(2, imagem.shape[1] // 500)
    for (x, y, w, h, nome) in lista_caixas:
        cv2.rectangle(saida, (x, y), (x + w, y + h), (255, 0, 255), espessura)
        cv2.putText(
            saida, nome, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * espessura / 2.0, (255, 0, 255), espessura, cv2.LINE_AA,
        )
    return saida
