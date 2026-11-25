"""
Etapa 3 do pipeline: PRE-PROCESSAMENTO da tarja para OCR.

Os dois pipelines (Tesseract e EasyOCR) consomem EXATAMENTE as mesmas imagens
pre-processadas -- e isso que torna a comparacao entre as ferramentas justa:
a unica variavel que muda e o motor de reconhecimento.

Em vez de apostar num unico pre-processamento, cada tarja gera um CONJUNTO de
variantes (a atividade pede explicitamente que se testem tecnicas diferentes).
Cada motor executa o OCR em todas elas e fica com a leitura mais confiavel --
estrategia conhecida como "ensemble de pre-processamentos". O relatorio final
registra qual variante venceu em cada imagem, o que mostra na pratica a
sensibilidade de cada OCR ao pre-processamento.

Variantes implementadas:
  cinza       -- tons de cinza + reamostragem + filtro bilateral
  clahe       -- equalizacao de histograma local (contraste)
  nitidez     -- mascara de nitidez (unsharp masking)
  otsu        -- binarizacao global de Otsu + abertura morfologica
  adaptativo  -- limiar adaptativo gaussiano (robusto a sombra/reflexo)
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config as cfg


def _escalar(cinza, altura=None):
    """
    Reamostra a tarja para uma altura fixa.

    Tanto o Tesseract quanto o CRNN do EasyOCR trabalham melhor com caracteres
    de ~30-60 px de altura; as tarjas do dataset variam de ~20 a ~120 px.
    Padronizar a altura remove essa variavel.
    """
    altura = altura or cfg.ALTURA_OCR
    h, w = cinza.shape[:2]
    if h == 0 or w == 0:
        return cinza
    escala = altura / float(h)
    interp = cv2.INTER_CUBIC if escala > 1 else cv2.INTER_AREA
    return cv2.resize(cinza, (max(1, int(round(w * escala))), altura), interpolation=interp)


def _moldurar(imagem, margem=None, cor=255):
    """
    Adiciona uma borda uniforme ("quiet zone").

    O Tesseract precisa de margem ao redor do texto para segmentar a linha;
    sem ela, caracteres colados na borda sao frequentemente perdidos.
    """
    margem = cfg.MARGEM_OCR if margem is None else margem
    return cv2.copyMakeBorder(
        imagem, margem, margem, margem, margem, cv2.BORDER_CONSTANT, value=cor
    )


def _garantir_texto_escuro(binaria):
    """
    Deixa a imagem binaria no padrao "texto preto sobre fundo branco", que e o
    que os motores de OCR esperam (documento impresso).

    O criterio e a MAIORIA dos pixels: numa tarja de placa o fundo sempre
    ocupa mais area que os caracteres, entao se a imagem estiver majoritaria-
    mente escura a polaridade esta trocada. Usar as bordas como referencia
    nao funciona aqui -- elas caem sobre a moldura preta da placa.
    """
    if float(np.mean(binaria)) < 127.0:
        binaria = cv2.bitwise_not(binaria)
    return binaria


def reconstruir_caracteres(tarja, caracteres, altura=None, espaco=None):
    """
    Recompoe a linha a partir dos caracteres JA SEGMENTADOS.

    Em vez de entregar ao OCR o recorte fotografico da tarja, cada caractere
    detectado na segmentacao e recortado, binarizado LOCALMENTE (Otsu numa
    caixa pequena e quase bimodal, o que o torna muito mais confiavel que um
    limiar global) e colado numa tela branca com altura e espacamento fixos.

    O resultado elimina de uma vez a textura de seguranca do fundo, a moldura
    preta, o reflexo e o espacamento irregular -- sobra texto limpo, que e o
    tipo de entrada para o qual o Tesseract foi projetado.

    Devolve None se nao houver caracteres segmentados utilizaveis.
    """
    if tarja is None or tarja.size == 0 or not caracteres:
        return None

    altura = altura or cfg.ALTURA_OCR
    espaco = espaco if espaco is not None else max(6, altura // 10)
    cinza = cv2.cvtColor(tarja, cv2.COLOR_BGR2GRAY) if tarja.ndim == 3 else tarja
    ht, wt = cinza.shape[:2]

    glifos = []
    for x, y, w, h in sorted(caracteres, key=lambda c: c[0]):
        # Uma folga de 1 px evita cortar a borda do traco.
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(wt, x + w + 1), min(ht, y + h + 1)
        recorte = cinza[y0:y1, x0:x1]
        if recorte.size == 0 or recorte.shape[0] < 4 or recorte.shape[1] < 2:
            continue

        # Otsu INVERTIDO: os caracteres de uma placa brasileira sao sempre
        # mais escuros que o corpo da placa, entao o lado escuro do limiar e
        # sempre a tinta. Como a caixa e justa, o histograma e praticamente
        # bimodal e o limiar sai muito mais preciso que um global.
        _, binario = cv2.threshold(
            recorte, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        escala = altura / float(binario.shape[0])
        largura = max(2, int(round(binario.shape[1] * escala)))
        binario = cv2.resize(binario, (largura, altura), interpolation=cv2.INTER_CUBIC)
        _, binario = cv2.threshold(binario, 127, 255, cv2.THRESH_BINARY)
        glifos.append(cv2.bitwise_not(binario))      # texto preto / fundo branco

    if not glifos:
        return None

    largura_total = sum(g.shape[1] for g in glifos) + espaco * (len(glifos) - 1)
    tela = np.full((altura, largura_total), 255, np.uint8)
    cursor = 0
    for g in glifos:
        tela[:, cursor:cursor + g.shape[1]] = g
        cursor += g.shape[1] + espaco
    return _moldurar(tela)


def gerar_variantes(tarja, caracteres=None):
    """
    Gera o dicionario {nome_da_variante: imagem} a partir da tarja recortada.

    Recebe a tarja em BGR (saida da segmentacao) e devolve imagens em tons de
    cinza, todas na mesma altura e com a mesma moldura. `caracteres` sao as
    caixas devolvidas pela segmentacao, usadas na variante reconstruida.
    """
    if tarja is None or tarja.size == 0:
        return {}

    cinza = cv2.cvtColor(tarja, cv2.COLOR_BGR2GRAY) if tarja.ndim == 3 else tarja
    cinza = _escalar(cinza)

    variantes = {}

    # (1) Cinza: filtro bilateral suaviza o ruido de compressao JPEG e a
    #     textura de seguranca ("MERCOSUL" repetido no fundo) PRESERVANDO as
    #     bordas dos caracteres -- diferente de um blur gaussiano.
    base = cv2.bilateralFilter(cinza, 7, 60, 60)
    variantes["cinza"] = _moldurar(base)

    # (2) CLAHE: equalizacao de histograma por blocos. Recupera placas lavadas
    #     (cinza-claro sobre prata) sem estourar as bem expostas.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(base)
    variantes["clahe"] = _moldurar(clahe)

    # (3) Nitidez (unsharp masking): compensa o desfoque de movimento/foco de
    #     algumas fotos, realcando as bordas dos caracteres.
    borrado = cv2.GaussianBlur(clahe, (0, 0), 3)
    nitido = cv2.addWeighted(clahe, 1.6, borrado, -0.6, 0)
    variantes["nitidez"] = _moldurar(nitido)

    # (4) Otsu: binarizacao global com limiar otimo. Funciona bem quando a
    #     iluminacao da tarja e uniforme.
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(
        otsu, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    variantes["otsu"] = _moldurar(_garantir_texto_escuro(otsu))

    # (5) Adaptativo: limiar calculado por vizinhanca. Resolve os casos de
    #     sombra parcial e reflexo especular sobre a placa.
    bloco = max(15, (cinza.shape[0] // 2) | 1)
    adaptativo = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bloco, 11
    )
    adaptativo = cv2.morphologyEx(
        adaptativo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    variantes["adaptativo"] = _moldurar(_garantir_texto_escuro(adaptativo))

    # (6) Reconstruida: caracteres segmentados, binarizados individualmente e
    #     recolados com espacamento regular. Normalmente e a melhor entrada
    #     para o Tesseract (ver `reconstruir_caracteres`).
    reconstruida = reconstruir_caracteres(tarja, caracteres)
    if reconstruida is not None:
        variantes["segmentada"] = reconstruida

    return variantes


def montar_mosaico(variantes):
    """Empilha as variantes numa unica imagem rotulada (para o relatorio)."""
    if not variantes:
        return None
    largura = max(v.shape[1] for v in variantes.values())
    linhas = []
    for nome, img in variantes.items():
        img = cv2.copyMakeBorder(
            img, 0, 0, 0, largura - img.shape[1], cv2.BORDER_CONSTANT, value=255
        )
        colorida = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            colorida, nome, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        linhas.append(colorida)
    return np.vstack(linhas)
