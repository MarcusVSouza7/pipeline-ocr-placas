"""
Linha de base SEM ROI -- a "DEMONSTRACAO IMPORTANTE" da Aula 9.

O slide compara:

    OCR DIRETO                     OCR COM ROI
    Imagem inteira do carro.       Somente a placa recortada.
    Resultado tende a ser ruim.    Resultado tende a ser melhor.

O pipeline principal do projeto e o lado direito (OCR com ROI). Este modulo
implementa o lado esquerdo, para que o ganho da segmentacao seja MEDIDO em vez
de apenas afirmado.

Honestidade da comparacao
-------------------------
Seria facil -- e desonesto -- fabricar um baseline ruim. Duas decisoes evitam
isso:

  * Os `--psm` sao DIFERENTES dos do pipeline com ROI, e de proposito. Com ROI
    a tarja e uma linha unica de texto (psm 7/8/13); numa foto inteira de
    veiculo o texto e esparso e espalhado, entao usamos psm 3 (automatico,
    o padrao do slide "OCR MAIS SIMPLES POSSIVEL"), 6 (bloco uniforme) e 11
    (texto esparso). Rodar psm 7 numa foto inteira produziria um baseline
    artificialmente pessimo.
  * O EasyOCR roda com os limiares PADRAO, nao com os limiares afrouxados do
    pipeline com ROI (que assumem uma linha unica e larga). Numa cena inteira
    os limiares padrao sao os corretos: o CRAFT encontra varios blocos de
    texto e cada bloco e uma leitura candidata.

Em ambos os motores, cada bloco de texto encontrado na imagem conta como um
candidato, e o vencedor e escolhido pela MESMA regra do pipeline com ROI
(`postprocess.escolher_melhor`). Ou seja: o baseline tem todas as chances de
encontrar a placa na foto inteira -- o que ele nao tem e a segmentacao.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config as cfg
from . import postprocess as pos


# Modos de segmentacao de pagina apropriados para uma CENA INTEIRA (ver
# docstring do modulo). Nada a ver com os PSMS do pipeline com ROI.
PSMS_CENA = (3, 6, 11)

# Largura maxima da imagem enviada ao OCR. Fotos de 4000 px deixam o Tesseract
# muito lento sem melhorar a leitura; 1600 px preserva placas legiveis.
LARGURA_MAX = 1600


def _preparar(imagem):
    """
    Gera as variantes da IMAGEM INTEIRA, seguindo o pipeline de
    pre-processamento do slide "PIPELINE DE PRE-PROCESSAMENTO" da Aula 9:
    imagem colorida -> escala de cinza -> binarizacao.

    Devolve {nome: imagem}. Sao duas variantes (e nao as seis do pipeline com
    ROI) porque as outras quatro -- CLAHE, nitidez, adaptativo e `segmentada`
    -- dependem de uma tarja ja recortada e nao fazem sentido numa cena.
    """
    h, w = imagem.shape[:2]
    if w > LARGURA_MAX:
        escala = LARGURA_MAX / float(w)
        imagem = cv2.resize(
            imagem, (LARGURA_MAX, max(1, int(round(h * escala)))),
            interpolation=cv2.INTER_AREA,
        )

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"cena_cinza": cinza, "cena_otsu": otsu}


def _candidatos_para_tentativas(variante, blocos):
    """
    Converte os blocos de texto de uma variante em tentativas de leitura.

    Cada bloco de >= 4 caracteres e um candidato (a placa pode ter sido lida
    inteira ou partida em dois pedacos), e a concatenacao de todos os blocos
    tambem -- exatamente a estrategia do exemplo `placa_ocr.py` do professor,
    que ordena os blocos por X e junta.
    """
    tentativas = []
    for texto, conf in blocos:
        limpo, corrigido = pos.processar(texto)
        if len(limpo) < 4:
            continue
        tentativas.append({
            "variante": variante, "texto_cru": texto,
            "texto": limpo, "texto_corrigido": corrigido,
            "confianca": round(float(conf), 3),
        })

    if len(blocos) > 1:
        juntos = "".join(t for t, _ in blocos)
        confs = [c for _, c in blocos]
        limpo, corrigido = pos.processar(juntos)
        tentativas.append({
            "variante": variante + "_concat", "texto_cru": juntos,
            "texto": limpo, "texto_corrigido": corrigido,
            "confianca": round(float(sum(confs) / max(1, len(confs))), 3),
        })
    return tentativas


# --------------------------------------------------------------------------- #
# Tesseract na imagem inteira
# --------------------------------------------------------------------------- #
def _blocos_tesseract(imagem, psm):
    """
    Roda o Tesseract na cena e devolve [(texto_da_palavra, confianca)].

    Usa `image_to_data` (slide "CAIXAS DE TEXTO DETECTADAS") para obter uma
    confianca por palavra -- numa cena inteira interessa cada palavra
    isoladamente, nao a string unica devolvida por `image_to_string`.
    """
    import pytesseract

    from . import ocr_tesseract

    ocr_tesseract.configurar_tesseract()
    config = (
        "--oem 3 --psm %d "
        "-c tessedit_char_whitelist=%s "
        "-c load_system_dawg=0 -c load_freq_dawg=0" % (psm, cfg.ALFABETO)
    )
    dados = pytesseract.image_to_data(
        imagem, lang="eng", config=config,
        output_type=pytesseract.Output.DICT,
    )

    blocos = []
    for texto, conf in zip(dados["text"], dados["conf"]):
        texto = (texto or "").strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if texto and conf >= 0:
            blocos.append((texto, conf / 100.0))
    return blocos


def reconhecer_tesseract(imagem, **_kwargs):
    """OCR direto com Tesseract na imagem inteira (sem deteccao, sem ROI)."""
    inicio = time.perf_counter()
    tentativas = []
    chamadas = 0

    for nome, variante in _preparar(imagem).items():
        for psm in PSMS_CENA:
            try:
                blocos = _blocos_tesseract(variante, psm)
            except Exception:
                continue
            chamadas += 1
            tentativas += _candidatos_para_tentativas(
                "%s_psm%d" % (nome, psm), blocos
            )

    return _finalizar(tentativas, inicio, chamadas)


# --------------------------------------------------------------------------- #
# EasyOCR na imagem inteira
# --------------------------------------------------------------------------- #
def reconhecer_easyocr(imagem, usar_gpu=False, **_kwargs):
    """OCR direto com EasyOCR na imagem inteira (sem deteccao, sem ROI)."""
    from . import ocr_easyocr

    leitor = ocr_easyocr.obter_leitor(usar_gpu)
    inicio = time.perf_counter()
    tentativas = []
    chamadas = 0

    for nome, variante in _preparar(imagem).items():
        entrada = np.stack([variante] * 3, axis=-1) if variante.ndim == 2 else variante
        try:
            # Limiares PADRAO: numa cena inteira o CRAFT deve separar os
            # blocos de texto, nao fundi-los como no pipeline com ROI.
            resultados = leitor.readtext(
                entrada, allowlist=cfg.ALFABETO, detail=1, paragraph=False,
            )
        except Exception:
            continue
        chamadas += 1

        # Ordena por X para que a concatenacao respeite a ordem de leitura.
        blocos = []
        for caixa, texto, conf in resultados:
            x = min(p[0] for p in caixa)
            blocos.append((x, texto, float(conf)))
        blocos.sort(key=lambda t: t[0])
        tentativas += _candidatos_para_tentativas(
            nome, [(t, c) for _, t, c in blocos]
        )

    return _finalizar(tentativas, inicio, chamadas)


# --------------------------------------------------------------------------- #
def _finalizar(tentativas, inicio, chamadas):
    """Escolhe a melhor leitura com a MESMA regra do pipeline com ROI."""
    melhor, tentativas = pos.escolher_melhor(tentativas)
    if melhor is None:
        melhor = {
            "texto_cru": "", "texto": "", "texto_corrigido": "",
            "confianca": 0.0, "variante": "", "nota": 0.0,
        }
    melhor = dict(melhor)
    melhor["tempo_s"] = round(time.perf_counter() - inicio, 3)
    melhor["chamadas"] = chamadas
    melhor["tentativas"] = tentativas
    return melhor


def reconhecer(motor):
    """Devolve a funcao de reconhecimento sem ROI do motor pedido."""
    if motor == "tesseract":
        return reconhecer_tesseract
    if motor == "easyocr":
        return reconhecer_easyocr
    raise ValueError("motor deve ser 'tesseract' ou 'easyocr'")
