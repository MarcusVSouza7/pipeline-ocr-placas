"""
Etapa 4a do pipeline: OCR com TESSERACT.

O Tesseract e um motor classico (LSTM sobre linha de texto, modo `oem 3`).
Para placas, tres ajustes sao decisivos:

  * `tessedit_char_whitelist` -- restringe o alfabeto a A-Z e 0-9, eliminando
    de saida pontuacao e acentos que o motor inventaria sobre a textura de
    seguranca da placa;
  * `--psm` (page segmentation mode) -- a tarja e UMA linha de texto, entao
    usamos psm 7 (linha unica), 8 (palavra unica) e 13 (linha crua, sem
    heuristicas de layout). Modos diferentes acertam imagens diferentes;
  * moldura branca ao redor da tarja (feita no pre-processamento), sem a qual
    o Tesseract descarta caracteres colados na borda.

Cada combinacao (variante de pre-processamento x psm) e executada e a melhor
leitura e escolhida pela confianca media do proprio Tesseract, ponderada pela
aderencia ao formato de placa brasileira.
"""

from __future__ import annotations

import os
import shutil
import time

import pytesseract

from . import config as cfg
from . import postprocess as pos

# Modos de segmentacao de pagina testados (ver docstring).
PSMS = (7, 8, 13)

_configurado = False


def configurar_tesseract():
    """
    Localiza o executavel do Tesseract.

    Procura no PATH e, em seguida, nos caminhos padrao de instalacao do
    Windows -- onde o instalador oficial nao registra o PATH automaticamente.
    """
    global _configurado
    if _configurado:
        return pytesseract.pytesseract.tesseract_cmd

    caminho = shutil.which("tesseract")
    if caminho is None:
        for candidato in cfg.CAMINHOS_TESSERACT:
            if os.path.isfile(candidato):
                caminho = candidato
                break
    if caminho is None:
        raise RuntimeError(
            "Tesseract nao encontrado. Instale-o (Windows: "
            "winget install UB-Mannheim.TesseractOCR) ou ajuste "
            "CAMINHOS_TESSERACT em src/config.py."
        )
    pytesseract.pytesseract.tesseract_cmd = caminho
    _configurado = True
    return caminho


def versao():
    """Versao do binario do Tesseract, para registro no relatorio."""
    configurar_tesseract()
    return str(pytesseract.get_tesseract_version())


def _config(psm):
    """
    Monta a linha de configuracao do Tesseract.

    `load_system_dawg=0` e `load_freq_dawg=0` desligam os dicionarios de
    palavras do ingles: uma placa nao e uma palavra, e sem isso o motor tende
    a "corrigir" a leitura para algo pronunciavel.
    """
    return (
        "--oem 3 --psm %d "
        "-c tessedit_char_whitelist=%s "
        "-c load_system_dawg=0 -c load_freq_dawg=0" % (psm, cfg.ALFABETO)
    )


def _ler(imagem, psm):
    """Executa o Tesseract e devolve (texto, confianca_media_0a1)."""
    dados = pytesseract.image_to_data(
        imagem, lang="eng", config=_config(psm),
        output_type=pytesseract.Output.DICT,
    )
    partes, confiancas = [], []
    for texto, conf in zip(dados["text"], dados["conf"]):
        texto = (texto or "").strip()
        conf = float(conf)
        if texto and conf >= 0:
            partes.append(texto)
            confiancas.append(conf)
    if not partes:
        return "", 0.0
    return "".join(partes), sum(confiancas) / len(confiancas) / 100.0


def reconhecer(variantes):
    """
    Roda o Tesseract em todas as variantes de pre-processamento x todos os
    `--psm` e devolve a melhor leitura, escolhida por
    `postprocess.escolher_melhor` (confianca do motor + aderencia ao formato
    de placa + consenso entre as variantes) -- a mesma regra usada no pipeline
    do EasyOCR.

    Devolve um dicionario com o texto cru, o limpo, o corrigido pela mascara,
    a confianca, a variante/psm vencedores e o tempo gasto.
    """
    configurar_tesseract()
    inicio = time.perf_counter()

    tentativas = []
    for nome, imagem in variantes.items():
        for psm in PSMS:
            try:
                cru, conf = _ler(imagem, psm)
            except Exception:
                continue
            limpo, corrigido = pos.processar(cru)
            tentativas.append({
                "variante": nome, "psm": psm, "texto_cru": cru,
                "texto": limpo, "texto_corrigido": corrigido,
                "confianca": round(conf, 3),
            })

    melhor, tentativas = pos.escolher_melhor(tentativas)
    if melhor is None:
        melhor = {
            "texto_cru": "", "texto": "", "texto_corrigido": "",
            "confianca": 0.0, "variante": "", "psm": None, "nota": 0.0,
        }
    melhor = dict(melhor)
    melhor["tempo_s"] = round(time.perf_counter() - inicio, 3)
    melhor["tentativas"] = tentativas
    return melhor
