"""
Comparacao entre os dois pipelines de OCR.

Le os JSON produzidos por `run_tesseract.py` e `run_easyocr.py` e gera:

  resultados/comparacao/comparacao.txt   -- tabela lado a lado + resumo
  resultados/comparacao/comparacao.csv   -- mesmos dados para planilha
  resultados/comparacao/analise.md       -- analise escrita dos resultados

A analise e o objetivo central do projeto: nao basta rodar os dois motores,
e preciso entender ONDE e POR QUE eles divergem.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime

from . import config as cfg
from . import evaluate


def _carregar(motor):
    caminho = cfg.DIR_RESULTADOS / motor / ("resultados_%s.json" % motor)
    if not caminho.exists():
        raise FileNotFoundError(
            "Faltam os resultados de '%s'. Rode antes: python run_%s.py"
            % (motor, motor)
        )
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def _indexar(dados):
    return {r["imagem"]: r for r in dados["resultados"]}


def _erros_por_caractere(resultados, gabarito):
    """
    Conta as substituicoes de caractere mais frequentes (lido -> esperado).

    So considera leituras com o mesmo comprimento do gabarito, onde a
    correspondencia posicao a posicao e inequivoca.
    """
    contagem = Counter()
    for r in resultados:
        esperado = gabarito.get(r["imagem"], "")
        lido = r.get("texto_corrigido", "")
        if not esperado or len(lido) != len(esperado):
            continue
        for a, b in zip(lido, esperado):
            if a != b:
                contagem["%s -> %s" % (a, b)] += 1
    return contagem


def comparar(verboso=True):
    """Gera os tres arquivos de comparacao e devolve as metricas dos dois motores."""
    cfg.garantir_diretorios()
    gabarito = evaluate.carregar_gabarito()

    dados_t = _carregar("tesseract")
    dados_e = _carregar("easyocr")
    mt, me = dados_t["metricas"], dados_e["metricas"]
    idx_t, idx_e = _indexar(dados_t), _indexar(dados_e)
    imagens = sorted(set(idx_t) | set(idx_e),
                     key=lambda n: (0, int(n.split(".")[0])) if n.split(".")[0].isdigit() else (1, n))

    # ----------------------------------------------------------------- #
    # Tabela lado a lado
    # ----------------------------------------------------------------- #
    linhas = [
        "COMPARACAO ENTRE PIPELINES DE OCR - PLACAS VEICULARES",
        "Gerado em: %s" % datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Dataset  : %s (%d imagens)" % (cfg.DIR_DATASET.name, len(imagens)),
        "",
        "Legenda: OK = leitura identica ao gabarito.",
        "",
        "%-9s %-9s %-10s %-4s %-10s %-4s %s" % (
            "IMAGEM", "GABARITO", "TESSERACT", "OK", "EASYOCR", "OK", "CONCORDAM"),
        "-" * 72,
    ]
    concordancias = ambos_certos = so_tess = so_easy = nenhum = 0
    for nome in imagens:
        rt, re_ = idx_t.get(nome, {}), idx_e.get(nome, {})
        alvo = gabarito.get(nome, "")
        tt = rt.get("texto_corrigido", "") or "-"
        te = re_.get("texto_corrigido", "") or "-"
        okt = alvo and tt == alvo
        oke = alvo and te == alvo
        igual = tt == te
        concordancias += int(igual)
        ambos_certos += int(bool(okt and oke))
        so_tess += int(bool(okt and not oke))
        so_easy += int(bool(oke and not okt))
        nenhum += int(bool(alvo and not okt and not oke))
        linhas.append("%-9s %-9s %-10s %-4s %-10s %-4s %s" % (
            nome, alvo or "-", tt, "OK" if okt else "", te, "OK" if oke else "",
            "sim" if igual else "NAO",
        ))

    # ----------------------------------------------------------------- #
    # Resumo quantitativo
    # ----------------------------------------------------------------- #
    def bloco(m):
        return [
            "  Acerto exato (com mascara) : %5.1f%%  (%d/%d)" % (
                m["acerto_exato_pct"], m["acertos"], m["avaliadas"]),
            "  Acerto exato (sem mascara) : %5.1f%%  (%d/%d)" % (
                m["acerto_exato_sem_mascara_pct"], m["acertos_sem_mascara"],
                m["avaliadas"]),
            "  Acuracia por caractere     : %5.1f%%" % m["acuracia_caractere_pct"],
            "  Acur. caractere sem mascara: %5.1f%%" % m[
                "acuracia_caractere_sem_mascara_pct"],
            "  Tempo total / por imagem   : %6.1fs / %.2fs" % (
                m["tempo_total_s"], m["tempo_medio_s"]),
            "  Chamadas ao motor / tempo  : %d / %.3fs por chamada" % (
                m.get("chamadas_ocr", 0), m.get("tempo_por_leitura_s", 0.0)),
        ]

    linhas += ["-" * 72, "", "TESSERACT OCR (v%s)" % dados_t["versao"]] + bloco(mt)
    linhas += ["", "EASYOCR (v%s)" % dados_e["versao"]] + bloco(me)
    linhas += [
        "",
        "CONCORDANCIA ENTRE OS MOTORES",
        "  Leituras identicas         : %d/%d (%.1f%%)" % (
            concordancias, len(imagens), 100.0 * concordancias / max(1, len(imagens))),
        "  Acertaram ambos            : %d" % ambos_certos,
        "  So o Tesseract acertou     : %d" % so_tess,
        "  So o EasyOCR acertou       : %d" % so_easy,
        "  Nenhum dos dois acertou    : %d" % nenhum,
        "",
        "A segmentacao e o pre-processamento sao IDENTICOS nos dois pipelines;",
        "a unica variavel e o motor de OCR.",
        "  Placas segmentadas corretamente: %d/%d (%.1f%%)" % (
            mt["placas_detectadas"], mt["imagens"], mt["taxa_deteccao"]),
        "",
    ]

    # ----------------------------------------------------------------- #
    # Confusoes de caractere mais frequentes
    # ----------------------------------------------------------------- #
    for rotulo, dados in (("TESSERACT", dados_t), ("EASYOCR", dados_e)):
        erros = _erros_por_caractere(dados["resultados"], gabarito)
        linhas.append("CONFUSOES MAIS FREQUENTES - %s" % rotulo)
        if erros:
            for par, n in erros.most_common(8):
                linhas.append("  %-10s %dx" % (par, n))
        else:
            linhas.append("  (nenhuma)")
        linhas.append("")

    # ----------------------------------------------------------------- #
    # Qual variante de pre-processamento venceu
    # ----------------------------------------------------------------- #
    for rotulo, dados in (("TESSERACT", dados_t), ("EASYOCR", dados_e)):
        venc = Counter(r.get("variante", "") for r in dados["resultados"])
        linhas.append("VARIANTE DE PRE-PROCESSAMENTO ESCOLHIDA - %s" % rotulo)
        for nome, n in venc.most_common():
            linhas.append("  %-12s %dx" % (nome or "(nenhuma)", n))
        linhas.append("")

    (cfg.DIR_SAIDA_COMPARACAO / "comparacao.txt").write_text(
        "\n".join(linhas), encoding="utf-8")

    # ----------------------------------------------------------------- #
    # CSV
    # ----------------------------------------------------------------- #
    with open(cfg.DIR_SAIDA_COMPARACAO / "comparacao.csv", "w",
              encoding="utf-8", newline="") as f:
        campos = [
            "imagem", "gabarito",
            "tesseract", "tesseract_ok", "tesseract_acuracia", "tesseract_conf",
            "tesseract_variante", "tesseract_tempo_s",
            "easyocr", "easyocr_ok", "easyocr_acuracia", "easyocr_conf",
            "easyocr_variante", "easyocr_tempo_s",
            "concordam",
        ]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for nome in imagens:
            rt, re_ = idx_t.get(nome, {}), idx_e.get(nome, {})
            alvo = gabarito.get(nome, "")
            tt = rt.get("texto_corrigido", "")
            te = re_.get("texto_corrigido", "")
            escritor.writerow({
                "imagem": nome, "gabarito": alvo,
                "tesseract": tt,
                "tesseract_ok": int(bool(alvo) and tt == alvo),
                "tesseract_acuracia": round(evaluate.acuracia_caractere(tt, alvo), 4),
                "tesseract_conf": rt.get("confianca", 0.0),
                "tesseract_variante": rt.get("variante", ""),
                "tesseract_tempo_s": rt.get("tempo_s", 0.0),
                "easyocr": te,
                "easyocr_ok": int(bool(alvo) and te == alvo),
                "easyocr_acuracia": round(evaluate.acuracia_caractere(te, alvo), 4),
                "easyocr_conf": re_.get("confianca", 0.0),
                "easyocr_variante": re_.get("variante", ""),
                "easyocr_tempo_s": re_.get("tempo_s", 0.0),
                "concordam": int(tt == te),
            })

    _gravar_analise(dados_t, dados_e, gabarito, idx_t, idx_e, imagens,
                    concordancias, ambos_certos, so_tess, so_easy, nenhum)

    if verboso:
        print("\n".join(linhas))
        print("Arquivos gravados em: %s" % cfg.DIR_SAIDA_COMPARACAO)
    return mt, me


def _gravar_analise(dados_t, dados_e, gabarito, idx_t, idx_e, imagens,
                    concordancias, ambos_certos, so_tess, so_easy, nenhum):
    """Escreve a analise em Markdown, com os numeros preenchidos."""
    mt, me = dados_t["metricas"], dados_e["metricas"]
    vencedor = ("Tesseract" if mt["acerto_exato_pct"] > me["acerto_exato_pct"]
                else "EasyOCR" if me["acerto_exato_pct"] > mt["acerto_exato_pct"]
                else "empate")
    erros_t = _erros_por_caractere(dados_t["resultados"], gabarito)
    erros_e = _erros_por_caractere(dados_e["resultados"], gabarito)

    divergentes = [
        n for n in imagens
        if idx_t.get(n, {}).get("texto_corrigido") != idx_e.get(n, {}).get("texto_corrigido")
    ]

    md = [
        "# Analise comparativa: Tesseract OCR x EasyOCR",
        "",
        "Gerado automaticamente por `python run_comparacao.py` em %s."
        % datetime.now().strftime("%d/%m/%Y %H:%M"),
        "",
        "Os dois pipelines compartilham **exatamente** as mesmas etapas de",
        "deteccao, segmentacao, extracao e pre-processamento. A unica variavel",
        "e o motor de OCR, de modo que as diferencas abaixo sao atribuiveis a ele.",
        "",
        "## 1. Resultados quantitativos",
        "",
        "| Metrica | Tesseract | EasyOCR |",
        "|---|---:|---:|",
        "| Acerto exato (pos-mascara) | %.1f%% | %.1f%% |" % (
            mt["acerto_exato_pct"], me["acerto_exato_pct"]),
        "| Acerto exato (sem mascara) | %.1f%% | %.1f%% |" % (
            mt["acerto_exato_sem_mascara_pct"], me["acerto_exato_sem_mascara_pct"]),
        "| Acuracia por caractere | %.1f%% | %.1f%% |" % (
            mt["acuracia_caractere_pct"], me["acuracia_caractere_pct"]),
        "| Acuracia por caractere (sem mascara) | %.1f%% | %.1f%% |" % (
            mt["acuracia_caractere_sem_mascara_pct"],
            me["acuracia_caractere_sem_mascara_pct"]),
        "| Tempo medio por imagem | %.2fs | %.2fs |" % (
            mt["tempo_medio_s"], me["tempo_medio_s"]),
        "| Tempo por chamada ao motor | %.3fs | %.3fs |" % (
            mt.get("tempo_por_leitura_s", 0.0), me.get("tempo_por_leitura_s", 0.0)),
        "| Chamadas ao motor (6 variantes x N modos) | %d | %d |" % (
            mt.get("chamadas_ocr", 0), me.get("chamadas_ocr", 0)),
        "",
        "Placas segmentadas com sucesso: **%d/%d (%.1f%%)** -- etapa comum aos dois."
        % (mt["placas_detectadas"], mt["imagens"], mt["taxa_deteccao"]),
        "",
        "## 2. Concordancia",
        "",
        "- Leituras identicas: **%d/%d**" % (concordancias, len(imagens)),
        "- Acertaram ambos: **%d**" % ambos_certos,
        "- So o Tesseract acertou: **%d**" % so_tess,
        "- So o EasyOCR acertou: **%d**" % so_easy,
        "- Nenhum acertou: **%d**" % nenhum,
        "",
        "Imagens em que as leituras divergem: %s."
        % (", ".join(divergentes) if divergentes else "nenhuma"),
        "",
        "## 3. Erros caractere a caractere",
        "",
        "| Confusao (lido -> correto) | Tesseract | EasyOCR |",
        "|---|---:|---:|",
    ]
    for par in sorted(set(erros_t) | set(erros_e),
                      key=lambda p: -(erros_t[p] + erros_e[p]))[:10]:
        md.append("| %s | %d | %d |" % (par, erros_t[par], erros_e[par]))

    venc_t = Counter(r.get("variante", "") for r in dados_t["resultados"])
    venc_e = Counter(r.get("variante", "") for r in dados_e["resultados"])
    md += [
        "",
        "## 4. Sensibilidade ao pre-processamento",
        "",
        "Cada imagem foi submetida a 6 variantes de pre-processamento; a tabela",
        "mostra quantas vezes cada uma produziu a melhor leitura de cada motor.",
        "",
        "| Variante | Tesseract | EasyOCR |",
        "|---|---:|---:|",
    ]
    for nome in sorted(set(venc_t) | set(venc_e)):
        md.append("| %s | %d | %d |" % (nome or "(nenhuma)", venc_t[nome], venc_e[nome]))

    ganho_t = mt["acerto_exato_pct"] - mt["acerto_exato_sem_mascara_pct"]
    ganho_e = me["acerto_exato_pct"] - me["acerto_exato_sem_mascara_pct"]
    vezes = (me.get("tempo_por_leitura_s", 0.0)
             / max(0.001, mt.get("tempo_por_leitura_s", 0.0)))
    diferenca = abs(mt["acertos"] - me["acertos"])
    plural = "imagem" if diferenca == 1 else "imagens"

    md += [
        "",
        "## 5. Leitura dos resultados",
        "",
        "**Motor com melhor acerto exato neste dataset: %s** (diferenca de %d %s)."
        % (vencedor, diferenca, plural),
        "",
        "Pontos observados:",
        "",
        "- *Erros de um caractere so*: a acuracia por caractere dos dois motores fica",
        "  bem acima do acerto exato (%.0f%% x %.0f%% no Tesseract, %.0f%% x %.0f%% no"
        % (mt["acuracia_caractere_pct"], mt["acerto_exato_pct"],
           me["acuracia_caractere_pct"], me["acerto_exato_pct"]),
        "  EasyOCR). Ou seja: quase todo erro e de UM caractere -- mas basta um para",
        "  invalidar a placa, e e isso que derruba o acerto exato.",
        "- *Naturezas de erro diferentes*: o Tesseract confunde caracteres que diferem",
        "  por um tracinho (L/I, G/Q, B/D), porque decide caractere a caractere. O",
        "  EasyOCR confunde caracteres de forma global parecida (V/Y, U/A), porque",
        "  reconhece a linha inteira de uma vez (CRNN+CTC) -- ver tabela 3.",
        "- *Pre-processamento*: a variante vencedora nos DOIS motores e a `segmentada`",
        "  (caracteres recortados pela segmentacao, binarizados um a um e recolados).",
        "  Isto e, boa parte do desempenho vem da segmentacao, nao do motor. A",
        "  diferenca aparece na segunda colocada: o Tesseract prefere binario (`otsu`)",
        "  e o EasyOCR prefere a imagem fotografica (`cinza`) -- coerente com um motor",
        "  classico contra um motor neural treinado em fotos naturais.",
        "- *Pos-processamento*: a mascara de formato (AAA0A00) soma +%.0f pontos ao"
        % ganho_t,
        "  Tesseract e +%.0f pontos ao EasyOCR, porque quase toda confusao de OCR em"
        % ganho_e,
        "  placas (O/0, I/1, S/5, G/0, Z/2) e resolvida pela POSICAO do caractere.",
        "  Conhecimento de dominio pesa aqui tanto quanto a escolha do motor.",
        "- *Custo*: o Tesseract e ~%.0fx mais rapido por chamada (%.2fs contra %.2fs em"
        % (vezes, mt.get("tempo_por_leitura_s", 0.0), me.get("tempo_por_leitura_s", 0.0)),
        "  CPU). O EasyOCR carrega dois modelos neurais; com GPU a distancia cai, mas",
        "  para rodar sem aceleracao a diferenca e decisiva.",
        "",
        "## 6. Limitacoes",
        "",
        "- O gabarito (`data/ground_truth.csv`) foi transcrito manualmente das 20",
        "  imagens; ele considera a placa principal de cada foto.",
        "- As imagens 2, 16 e 19 contem uma segunda placa/adesivo. A deteccao as",
        "  encontra e o `.txt` de saida as lista como deteccao secundaria, mas elas",
        "  nao entram nas metricas.",
        "- Com %d imagens, cada acerto vale %.0f pontos percentuais. Uma diferenca de"
        % (len(imagens), 100.0 / max(1, len(imagens))),
        "  %d %s entre os motores esta dentro do ruido amostral: o resultado"
        % (diferenca, plural),
        "  deve ser lido como tendencia (e, neste caso, como empate tecnico com",
        "  perfis de erro e de custo diferentes), nao como medida precisa.",
        "",
    ]
    (cfg.DIR_SAIDA_COMPARACAO / "analise.md").write_text("\n".join(md), encoding="utf-8")
