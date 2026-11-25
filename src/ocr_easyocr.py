"""
Etapa 4b do pipeline: OCR com EASYOCR.

O EasyOCR e um motor baseado em deep learning: um detector de texto (CRAFT)
seguido de um reconhecedor sequencial CRNN+CTC. Em relacao ao Tesseract isso
implica tres diferencas praticas:

  * tolera muito melhor desfoque, baixo contraste e caracteres deformados,
    porque reconhece a linha inteira em vez de caractere a caractere;
  * NAO gosta de imagem binarizada -- foi treinado com fotos naturais, entao
    recebe melhor as variantes em tons de cinza;
  * e ordens de grandeza mais lento em CPU (carrega dois modelos neurais).

Usamos `allowlist` (equivalente a whitelist do Tesseract) para restringir a
saida a A-Z e 0-9, e a confianca devolvida pelo proprio modelo para escolher
entre as variantes de pre-processamento.

O modelo e baixado automaticamente na primeira execucao (~100 MB) e fica em
cache em ~/.EasyOCR.
"""

from __future__ import annotations

import time

import numpy as np

from . import config as cfg
from . import postprocess as pos

_leitor = None


def obter_leitor(usar_gpu=False):
    """
    Devolve o leitor do EasyOCR (singleton).

    O carregamento dos modelos custa alguns segundos, entao e feito uma unica
    vez e reaproveitado em todas as imagens do dataset.
    """
    global _leitor
    if _leitor is None:
        import easyocr                       # import tardio: biblioteca pesada
        _leitor = easyocr.Reader(["en"], gpu=usar_gpu, verbose=False)
    return _leitor


def versao():
    """Versao do pacote EasyOCR, para registro no relatorio."""
    import easyocr
    return getattr(easyocr, "__version__", "desconhecida")


def _ler(leitor, imagem):
    """
    Executa o EasyOCR numa variante e devolve (texto, confianca).

    `readtext` pode devolver varias caixas (ex.: a placa partida em dois
    blocos); elas sao reordenadas pela coordenada X e concatenadas, de modo a
    reconstruir a sequencia na ordem correta de leitura.
    """
    resultados = leitor.readtext(
        imagem,
        allowlist=cfg.ALFABETO,
        detail=1,
        paragraph=False,
        # A tarja e uma linha unica e larga: afrouxar os limiares de ligacao
        # evita que o detector quebre a placa em blocos separados.
        width_ths=1.2,
        height_ths=1.2,
        text_threshold=0.5,
        low_text=0.3,
        mag_ratio=1.5,
    )
    if not resultados:
        return "", 0.0

    itens = []
    for caixa, texto, conf in resultados:
        x = min(p[0] for p in caixa)
        itens.append((x, texto, float(conf)))
    itens.sort(key=lambda t: t[0])

    texto = "".join(t for _, t, _ in itens)
    confianca = sum(c for _, _, c in itens) / len(itens)
    return texto, confianca


def reconhecer(variantes, usar_gpu=False):
    """
    Roda o EasyOCR em todas as variantes de pre-processamento e devolve a
    melhor leitura, usando `postprocess.escolher_melhor` -- exatamente a mesma
    regra de selecao do pipeline do Tesseract (confianca do motor + aderencia
    ao formato de placa + consenso entre as variantes).
    """
    leitor = obter_leitor(usar_gpu)
    inicio = time.perf_counter()

    tentativas = []
    for nome, imagem in variantes.items():
        # O EasyOCR espera imagem natural; convertemos cinza -> BGR.
        entrada = np.stack([imagem] * 3, axis=-1) if imagem.ndim == 2 else imagem
        try:
            cru, conf = _ler(leitor, entrada)
        except Exception:
            continue
        limpo, corrigido = pos.processar(cru)
        tentativas.append({
            "variante": nome, "texto_cru": cru,
            "texto": limpo, "texto_corrigido": corrigido,
            "confianca": round(conf, 3),
        })

    # Mesma regra de selecao usada no pipeline do Tesseract.
    melhor, tentativas = pos.escolher_melhor(tentativas)
    if melhor is None:
        melhor = {
            "texto_cru": "", "texto": "", "texto_corrigido": "",
            "confianca": 0.0, "variante": "", "nota": 0.0,
        }
    melhor = dict(melhor)
    melhor["tempo_s"] = round(time.perf_counter() - inicio, 3)
    melhor["tentativas"] = tentativas
    return melhor
