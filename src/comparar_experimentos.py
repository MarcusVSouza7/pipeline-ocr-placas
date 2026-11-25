"""
Relatorios das duas comparacoes experimentais do projeto.

  (1) DETECTOR -- Haar Cascade x detector classico proprio.
      Saida: resultados/comparacao/deteccao_haar_x_classico.txt | .md

  (2) ROI -- OCR com ROI x OCR direto na imagem inteira.
      Saida: resultados/comparacao/roi_x_sem_roi.txt | .md

Cada comparacao isola UMA variavel. Em (1) muda apenas a origem dos candidatos
de placa -- retificacao, segmentacao, pre-processamento, OCR e
pos-processamento sao os mesmos. Em (2) muda apenas a existencia da etapa de
segmentacao -- o motor e o pos-processamento sao os mesmos.

Pre-requisito: as execucoes correspondentes ja terem sido rodadas
(`run_experimentos.py` faz isso na ordem certa).
"""

from __future__ import annotations

import json
from datetime import datetime

from . import config as cfg
from . import evaluate


MOTORES = ("tesseract", "easyocr")
ROTULOS = {"tesseract": "Tesseract", "easyocr": "EasyOCR"}


def _carregar(motor, detector="classico", sem_roi=False):
    """Le o JSON de uma execucao; devolve None se ela nao foi rodada."""
    caminho = (cfg.dir_saida(motor, detector=detector, sem_roi=sem_roi)
               / ("resultados_%s.json" % motor))
    if not caminho.exists():
        return None
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def _indexar(dados):
    return {r["imagem"]: r for r in dados["resultados"]}


def _ordenar(nomes):
    def chave(n):
        raiz = n.split(".")[0]
        return (0, int(raiz)) if raiz.isdigit() else (1, n)
    return sorted(nomes, key=chave)


def _segmentou_sete(registro):
    """True se a etapa de segmentacao entregou os 7 caracteres da placa."""
    return bool(registro) and registro.get("n_caracteres", 0) == 7


# --------------------------------------------------------------------------- #
# (1) Haar Cascade x detector classico
# --------------------------------------------------------------------------- #
def comparar_detector(detector="haar", verboso=True):
    """
    Gera o relatorio Haar x classico.

    Mede o detector em tres niveis, do mais cru ao mais util -- a distincao
    importa porque "a cascata respondeu" nao e a mesma coisa que "o pipeline
    conseguiu ler a placa":

      1. CAIXAS CRUAS   -- `detectMultiScale` devolveu alguma caixa plausivel?
      2. SEGMENTACAO    -- a partir dessa caixa, os 7 caracteres foram isolados?
      3. OCR            -- a placa foi lida corretamente no fim?
    """
    gabarito = evaluate.carregar_gabarito()
    pares = {}
    for motor in MOTORES:
        base = _carregar(motor, detector="classico")
        alt = _carregar(motor, detector=detector)
        if base is not None and alt is not None:
            pares[motor] = (base, alt)

    if not pares:
        if verboso:
            print("Comparacao de detector: faltam execucoes. Rode antes:")
            print("  python run_tesseract.py --detector %s" % detector)
            print("  python run_easyocr.py   --detector %s" % detector)
        return None

    linhas = [
        "COMPARACAO DE DETECTOR DE PLACA - %s x CLASSICO" % detector.upper(),
        "Gerado em: %s" % datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Dataset  : %s" % cfg.DIR_DATASET.name,
        "",
        "O detector e a UNICA variavel: retificacao, segmentacao de caracteres,",
        "pre-processamento, OCR e pos-processamento sao identicos nos dois lados.",
        "",
        "  classico = HSV (faixa azul) + Sobel (bordas) + regiao clara",
        "  %-8s = cv2.CascadeClassifier, cascata de placa" % detector,
        "",
    ]

    md = [
        "# Deteccao: Haar Cascade x detector classico",
        "",
        "Gerado por `python run_experimentos.py` em %s."
        % datetime.now().strftime("%d/%m/%Y %H:%M"),
        "",
        "Este relatorio substitui o detector proprio do projeto pelo **Haar "
        "Cascade** (`cv2.CascadeClassifier` + `detectMultiScale`), mantendo "
        "todo o resto do pipeline igual, para medir o que o detector muda de "
        "fato.",
        "",
    ]

    for motor, (base, alt) in pares.items():
        mb, ma = base["metricas"], alt["metricas"]
        ib, ia = _indexar(base), _indexar(alt)
        imagens = _ordenar(set(ib) | set(ia))

        # Nivel 1: a cascata respondeu?
        com_caixa = sum(1 for n in imagens if ia.get(n, {}).get("caixas_haar", 0) > 0)
        caixas_total = sum(ia.get(n, {}).get("caixas_haar", 0) for n in imagens)
        # Nivel 2: a segmentacao isolou os 7 caracteres?
        seg_b = sum(1 for n in imagens if _segmentou_sete(ib.get(n)))
        seg_a = sum(1 for n in imagens if _segmentou_sete(ia.get(n)))

        linhas += [
            "=" * 72,
            "MOTOR: %s" % ROTULOS[motor].upper(),
            "=" * 72,
            "",
            "%-9s %-9s %-10s %-4s %-10s %-4s %s" % (
                "IMAGEM", "GABARITO", "CLASSICO", "OK", detector.upper(), "OK",
                "CAIXAS"),
            "-" * 72,
        ]
        for n in imagens:
            alvo = gabarito.get(n, "")
            tb = ib.get(n, {}).get("texto_corrigido", "") or "-"
            ta = ia.get(n, {}).get("texto_corrigido", "") or "-"
            linhas.append("%-9s %-9s %-10s %-4s %-10s %-4s %d" % (
                n, alvo or "-", tb, "OK" if alvo and tb == alvo else "",
                ta, "OK" if alvo and ta == alvo else "",
                ia.get(n, {}).get("caixas_haar", 0),
            ))

        linhas += [
            "-" * 72,
            "",
            "NIVEL 1 - A CASCATA RESPONDEU (caixas cruas de detectMultiScale)",
            "  Imagens com >= 1 caixa     : %d/%d (%.1f%%)" % (
                com_caixa, len(imagens), 100.0 * com_caixa / max(1, len(imagens))),
            "  Caixas cruas no total      : %d (media %.1f por imagem)" % (
                caixas_total, caixas_total / max(1, len(imagens))),
            "",
            "NIVEL 2 - SEGMENTACAO DOS 7 CARACTERES",
            "  classico                   : %d/%d (%.1f%%)" % (
                seg_b, len(imagens), 100.0 * seg_b / max(1, len(imagens))),
            "  %-26s : %d/%d (%.1f%%)" % (
                detector, seg_a, len(imagens), 100.0 * seg_a / max(1, len(imagens))),
            "",
            "NIVEL 3 - LEITURA CORRETA DA PLACA (fim do pipeline)",
            "  classico                   : %5.1f%%  (%d/%d)" % (
                mb["acerto_exato_pct"], mb["acertos"], mb["avaliadas"]),
            "  %-26s : %5.1f%%  (%d/%d)" % (
                detector, ma["acerto_exato_pct"], ma["acertos"], ma["avaliadas"]),
            "  classico, por caractere    : %5.1f%%" % mb["acuracia_caractere_pct"],
            "  %-26s : %5.1f%%" % (
                detector + ", por caractere", ma["acuracia_caractere_pct"]),
            "",
            "CUSTO",
            "  Tempo por imagem, classico : %.2fs" % mb["tempo_medio_s"],
            "  Tempo por imagem, %-9s: %.2fs" % (detector, ma["tempo_medio_s"]),
            "",
        ]

        # Onde cada detector foi melhor
        so_base = [n for n in imagens
                   if gabarito.get(n)
                   and ib.get(n, {}).get("texto_corrigido") == gabarito.get(n)
                   and ia.get(n, {}).get("texto_corrigido") != gabarito.get(n)]
        so_alt = [n for n in imagens
                  if gabarito.get(n)
                  and ia.get(n, {}).get("texto_corrigido") == gabarito.get(n)
                  and ib.get(n, {}).get("texto_corrigido") != gabarito.get(n)]
        sem_caixa = [n for n in imagens
                     if ia.get(n, {}).get("caixas_haar", 0) == 0]
        linhas += [
            "DIVERGENCIAS",
            "  So o classico acertou      : %s" % (", ".join(so_base) or "nenhuma"),
            "  So o %-21s: %s" % (detector + " acertou",
                                  ", ".join(so_alt) or "nenhuma"),
            "  Cascata nao achou nada     : %s" % (", ".join(sem_caixa) or "nenhuma"),
            "",
        ]

        md += [
            "## %s" % ROTULOS[motor],
            "",
            "| Nivel | Metrica | classico | %s |" % detector,
            "|---|---|---:|---:|",
            "| 1 | Imagens com caixa crua da cascata | - | %d/%d |" % (
                com_caixa, len(imagens)),
            "| 2 | 7 caracteres segmentados | %d/%d | %d/%d |" % (
                seg_b, len(imagens), seg_a, len(imagens)),
            "| 3 | Acerto exato da placa | %.1f%% | %.1f%% |" % (
                mb["acerto_exato_pct"], ma["acerto_exato_pct"]),
            "| 3 | Acuracia por caractere | %.1f%% | %.1f%% |" % (
                mb["acuracia_caractere_pct"], ma["acuracia_caractere_pct"]),
            "| - | Tempo por imagem | %.2fs | %.2fs |" % (
                mb["tempo_medio_s"], ma["tempo_medio_s"]),
            "",
            "- So o classico acertou: %s" % (", ".join(so_base) or "nenhuma"),
            "- So o %s acertou: %s" % (detector, ", ".join(so_alt) or "nenhuma"),
            "- A cascata nao devolveu caixa nenhuma em: %s"
            % (", ".join(sem_caixa) or "nenhuma imagem"),
            "",
        ]

    destino = cfg.DIR_SAIDA_COMPARACAO / ("deteccao_%s_x_classico" % detector)
    destino.with_suffix(".txt").write_text("\n".join(linhas), encoding="utf-8")
    destino.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    if verboso:
        print("\n".join(linhas))
        print("Gravado: %s.txt | .md" % destino)
    return linhas


# --------------------------------------------------------------------------- #
# (2) OCR com ROI x OCR direto na imagem inteira
# --------------------------------------------------------------------------- #
def comparar_roi(verboso=True):
    """
    Gera o relatorio ROI x sem ROI.

    Mede o ganho da etapa de segmentacao: o mesmo motor, com o mesmo
    pos-processamento, lendo (a) a tarja recortada e (b) a foto inteira.
    """
    gabarito = evaluate.carregar_gabarito()
    pares = {}
    for motor in MOTORES:
        com = _carregar(motor, detector="classico")
        sem = _carregar(motor, sem_roi=True)
        if com is not None and sem is not None:
            pares[motor] = (com, sem)

    if not pares:
        if verboso:
            print("Comparacao de ROI: faltam execucoes. Rode antes:")
            print("  python run_tesseract.py --sem-roi")
            print("  python run_easyocr.py   --sem-roi")
        return None

    linhas = [
        "OCR COM ROI x OCR DIRETO NA IMAGEM INTEIRA",
        "Gerado em: %s" % datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "Dataset  : %s" % cfg.DIR_DATASET.name,
        "",
        "Os dois lados da comparacao:",
        "  OCR DIRETO  -> imagem inteira do carro, sem etapa de segmentacao",
        "  OCR COM ROI -> somente a placa recortada pelo pipeline",
        "",
        "O motor e o pos-processamento (limpeza + mascara AAA0A00) sao os mesmos",
        "nos dois lados. A unica diferenca e a existencia da etapa de deteccao e",
        "segmentacao da placa.",
        "",
    ]
    md = [
        "# OCR com ROI x OCR direto na imagem inteira",
        "",
        "Gerado por `python run_experimentos.py` em %s."
        % datetime.now().strftime("%d/%m/%Y %H:%M"),
        "",
        "Rodar OCR na foto inteira do veiculo tende a dar resultado ruim, e "
        "recortar a ROI da placa tende a melhorar muito a leitura. Este "
        "relatorio mede esse ganho no dataset em vez de apenas afirma-lo.",
        "",
        "O baseline sem ROI nao foi sabotado: ele usa os `--psm` apropriados a "
        "uma cena (3, 6 e 11, nao os 7/8/13 de linha unica), os limiares padrao "
        "do EasyOCR, e cada bloco de texto encontrado na foto conta como "
        "candidato, escolhido pela mesma regra do pipeline com ROI.",
        "",
    ]

    for motor, (com, sem) in pares.items():
        mc, ms = com["metricas"], sem["metricas"]
        ic, is_ = _indexar(com), _indexar(sem)
        imagens = _ordenar(set(ic) | set(is_))

        linhas += [
            "=" * 72,
            "MOTOR: %s" % ROTULOS[motor].upper(),
            "=" * 72,
            "",
            "%-9s %-9s %-10s %-4s %-12s %-4s" % (
                "IMAGEM", "GABARITO", "COM ROI", "OK", "SEM ROI", "OK"),
            "-" * 72,
        ]
        vazias = 0
        for n in imagens:
            alvo = gabarito.get(n, "")
            tc = ic.get(n, {}).get("texto_corrigido", "") or "-"
            ts = is_.get(n, {}).get("texto_corrigido", "") or "-"
            if ts == "-":
                vazias += 1
            linhas.append("%-9s %-9s %-10s %-4s %-12s %-4s" % (
                n, alvo or "-", tc, "OK" if alvo and tc == alvo else "",
                ts, "OK" if alvo and ts == alvo else "",
            ))

        ganho = mc["acerto_exato_pct"] - ms["acerto_exato_pct"]
        ganho_car = mc["acuracia_caractere_pct"] - ms["acuracia_caractere_pct"]
        linhas += [
            "-" * 72,
            "",
            "  %-28s %10s %10s %10s" % ("METRICA", "COM ROI", "SEM ROI", "GANHO"),
            "  %-28s %9.1f%% %9.1f%% %9.1f pp" % (
                "Acerto exato (com mascara)", mc["acerto_exato_pct"],
                ms["acerto_exato_pct"], ganho),
            "  %-28s %9.1f%% %9.1f%% %9.1f pp" % (
                "Acerto exato (sem mascara)",
                mc["acerto_exato_sem_mascara_pct"],
                ms["acerto_exato_sem_mascara_pct"],
                mc["acerto_exato_sem_mascara_pct"]
                - ms["acerto_exato_sem_mascara_pct"]),
            "  %-28s %9.1f%% %9.1f%% %9.1f pp" % (
                "Acuracia por caractere", mc["acuracia_caractere_pct"],
                ms["acuracia_caractere_pct"], ganho_car),
            "  %-28s %9.2fs %9.2fs" % (
                "Tempo por imagem", mc["tempo_medio_s"], ms["tempo_medio_s"]),
            "",
            "  Imagens em que o OCR direto nao devolveu nada: %d/%d" % (
                vazias, len(imagens)),
            "",
        ]

        md += [
            "## %s" % ROTULOS[motor],
            "",
            "| Metrica | Com ROI | Sem ROI | Ganho |",
            "|---|---:|---:|---:|",
            "| Acerto exato (com mascara) | %.1f%% | %.1f%% | %+.1f pp |" % (
                mc["acerto_exato_pct"], ms["acerto_exato_pct"], ganho),
            "| Acerto exato (sem mascara) | %.1f%% | %.1f%% | %+.1f pp |" % (
                mc["acerto_exato_sem_mascara_pct"],
                ms["acerto_exato_sem_mascara_pct"],
                mc["acerto_exato_sem_mascara_pct"]
                - ms["acerto_exato_sem_mascara_pct"]),
            "| Acuracia por caractere | %.1f%% | %.1f%% | %+.1f pp |" % (
                mc["acuracia_caractere_pct"], ms["acuracia_caractere_pct"],
                ganho_car),
            "| Tempo por imagem | %.2fs | %.2fs | - |" % (
                mc["tempo_medio_s"], ms["tempo_medio_s"]),
            "",
            "Sem ROI o OCR nao devolveu nada em %d das %d imagens."
            % (vazias, len(imagens)),
            "",
        ]

    destino = cfg.DIR_SAIDA_COMPARACAO / "roi_x_sem_roi"
    destino.with_suffix(".txt").write_text("\n".join(linhas), encoding="utf-8")
    destino.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    if verboso:
        print("\n".join(linhas))
        print("Gravado: %s.txt | .md" % destino)
    return linhas


def comparar_tudo(verboso=True):
    """Gera as duas comparacoes, pulando as que nao tem dados."""
    cfg.garantir_diretorios()
    comparar_detector(verboso=verboso)
    comparar_roi(verboso=verboso)
