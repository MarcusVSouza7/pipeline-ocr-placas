"""
Metricas de avaliacao e comparacao entre os pipelines de OCR.

Duas metricas sao usadas, complementares:

  * ACERTO EXATO (placa inteira): e o que importa na pratica -- uma placa com
    um caractere errado e uma placa errada.
  * ACURACIA POR CARACTERE: 1 - distancia_de_levenshtein/len(gabarito). Mostra
    o quao PERTO a leitura chegou, distinguindo "errou um caractere" de
    "nao leu nada". Sem ela, dois motores muito diferentes podem empatar em
    0% de acerto exato e parecer equivalentes.

Ambas sao calculadas duas vezes: sobre o texto apenas limpo e sobre o texto
apos a mascara de formato, para isolar quanto do desempenho vem do OCR e
quanto vem do pos-processamento.
"""

from __future__ import annotations

import csv

from . import config as cfg


def levenshtein(a, b):
    """Distancia de edicao entre duas strings (programacao dinamica, O(n*m))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        atual = [i]
        for j, cb in enumerate(b, start=1):
            atual.append(min(
                anterior[j] + 1,            # remocao
                atual[j - 1] + 1,           # insercao
                anterior[j - 1] + (ca != cb),   # substituicao
            ))
        anterior = atual
    return anterior[-1]


def acuracia_caractere(predito, gabarito):
    """Fracao de caracteres corretos em [0, 1], via distancia de edicao."""
    if not gabarito:
        return 0.0
    return max(0.0, 1.0 - levenshtein(predito or "", gabarito) / len(gabarito))


def carregar_gabarito(caminho=None):
    """Le data/ground_truth.csv e devolve {nome_da_imagem: placa}."""
    caminho = caminho or cfg.ARQ_GABARITO
    gabarito = {}
    if not caminho.exists():
        return gabarito
    with open(caminho, "r", encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            gabarito[linha["imagem"].strip()] = linha["placa"].strip().upper()
    return gabarito


def avaliar(resultados, gabarito):
    """
    Calcula as metricas agregadas de um pipeline.

    `resultados` e a lista de dicionarios produzida por `pipeline.processar`.
    Imagens sem entrada no gabarito sao ignoradas nas metricas (mas nao na
    listagem de saida).
    """
    total = acertos = acertos_brutos = 0
    soma_car = soma_car_bruto = 0.0
    soma_tempo = 0.0
    detectadas = leituras = 0

    for r in resultados:
        soma_tempo += r.get("tempo_s", 0.0)
        leituras += len(r.get("tentativas", []))
        if r.get("detectada"):
            detectadas += 1
        esperado = gabarito.get(r["imagem"])
        if not esperado:
            continue
        total += 1
        bruto = r.get("texto", "") or ""
        corrigido = r.get("texto_corrigido", "") or ""
        acertos += int(corrigido == esperado)
        acertos_brutos += int(bruto == esperado)
        soma_car += acuracia_caractere(corrigido, esperado)
        soma_car_bruto += acuracia_caractere(bruto, esperado)

    n = max(1, total)
    return {
        "imagens": len(resultados),
        "avaliadas": total,
        "placas_detectadas": detectadas,
        "taxa_deteccao": round(100.0 * detectadas / max(1, len(resultados)), 2),
        "acertos": acertos,
        "acerto_exato_pct": round(100.0 * acertos / n, 2),
        "acuracia_caractere_pct": round(100.0 * soma_car / n, 2),
        "acertos_sem_mascara": acertos_brutos,
        "acerto_exato_sem_mascara_pct": round(100.0 * acertos_brutos / n, 2),
        "acuracia_caractere_sem_mascara_pct": round(100.0 * soma_car_bruto / n, 2),
        "tempo_total_s": round(soma_tempo, 2),
        "tempo_medio_s": round(soma_tempo / max(1, len(resultados)), 3),
        # Cada imagem passa por varias variantes de pre-processamento (e, no
        # Tesseract, por varios --psm). Este e o custo de UMA chamada ao motor,
        # que e a comparacao de velocidade realmente justa entre os dois.
        "chamadas_ocr": leituras,
        "tempo_por_leitura_s": round(soma_tempo / max(1, leituras), 3),
    }
