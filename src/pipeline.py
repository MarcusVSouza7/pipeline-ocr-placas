"""
Orquestracao do pipeline completo.

    imagem -> deteccao/segmentacao -> extracao da placa -> pre-processamento
           -> OCR (Tesseract OU EasyOCR) -> pos-processamento -> relatorios

O MESMO caminho e usado pelos dois motores: a unica diferenca entre
`executar("tesseract")` e `executar("easyocr")` e a funcao de reconhecimento
chamada na etapa de OCR. Isso garante que a comparacao final meca o OCR, e
nao diferencas de pre-processamento.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime

import cv2
import numpy as np

from . import config as cfg
from . import evaluate
from . import plate_detector as det
from . import postprocess as pos
from . import preprocessing as pre


# --------------------------------------------------------------------------- #
# Entrada/saida de imagens
# --------------------------------------------------------------------------- #
def listar_imagens(diretorio=None):
    """Lista as imagens do dataset em ordem numerica natural (1, 2, ..., 10)."""
    diretorio = diretorio or cfg.DIR_DATASET
    arquivos = [
        p for p in diretorio.iterdir()
        if p.is_file() and p.suffix.lower() in cfg.EXTENSOES_IMAGEM
    ]

    def chave(p):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)

    return sorted(arquivos, key=chave)


def ler_imagem(caminho):
    """
    Le a imagem de forma robusta a acentos no caminho.

    `cv2.imread` falha silenciosamente com caracteres nao-ASCII no Windows;
    por isso lemos os bytes e decodificamos com `cv2.imdecode`.
    """
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho, imagem):
    """Grava a imagem (tambem robusto a acentos no caminho)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".png", imagem)
    if ok:
        buffer.tofile(str(caminho))
    return ok


# --------------------------------------------------------------------------- #
# Processamento de uma imagem
# --------------------------------------------------------------------------- #
def processar_imagem(caminho, reconhecer, salvar_artefatos=True,
                     detector="classico", sem_roi=False, **kwargs):
    """
    Executa o pipeline completo numa imagem.

    `reconhecer` e a funcao de OCR (de `ocr_tesseract` ou `ocr_easyocr`), o que
    mantem esta funcao agnostica ao motor usado.

    `detector` ("classico" | "haar" | "ambos") escolhe o detector de placa.

    `sem_roi=True` pula as etapas 1, 2 e 3 e roda o OCR na imagem INTEIRA --
    a linha de base da "demonstracao importante" da Aula 9. Nesse modo
    `reconhecer` deve vir de `baseline_sem_roi`.
    """
    inicio = time.perf_counter()
    nome = caminho.name
    registro = {
        "imagem": nome, "detectada": False, "score_deteccao": 0.0,
        "metodo_deteccao": "", "angulo": 0.0, "n_caracteres": 0,
        "texto_cru": "", "texto": "", "texto_corrigido": "",
        "confianca": 0.0, "variante": "", "formato_valido": False,
        "tempo_s": 0.0, "deteccoes_extras": [],
        "detector": "-" if sem_roi else detector, "caixas_haar": 0,
    }

    imagem = ler_imagem(caminho)
    if imagem is None:
        registro["tempo_s"] = round(time.perf_counter() - inicio, 3)
        registro["erro"] = "falha ao ler a imagem"
        return registro

    # --- Linha de base SEM ROI: OCR direto na foto inteira ---------------- #
    if sem_roi:
        leitura = reconhecer(imagem, **kwargs)
        registro.update({k: v for k, v in leitura.items() if k != "tentativas"})
        registro["tentativas"] = leitura.get("tentativas", [])
        registro["formato_valido"] = pos.valida(registro["texto_corrigido"])
        registro["tempo_s"] = round(time.perf_counter() - inicio, 3)
        return registro

    # --- Etapas 1 e 2: deteccao, segmentacao e extracao da placa ---------- #
    deteccoes = det.detectar_placas(imagem, detector=detector)

    # Caixas CRUAS do Haar, antes de qualquer validacao do pipeline. E a
    # medida honesta de "o Haar achou a placa?", separada de "o pipeline
    # conseguiu aproveitar o que o Haar achou?".
    caixas_haar = []
    if detector in ("haar", "ambos"):
        from . import detector_haar
        trabalho, _escala = det._redimensionar(imagem, cfg.LARGURA_TRABALHO)
        cinza_h = cv2.cvtColor(trabalho, cv2.COLOR_BGR2GRAY)
        caixas_haar = detector_haar.caixas(cinza_h)
        registro["caixas_haar"] = len(caixas_haar)
        if salvar_artefatos:
            salvar_imagem(
                cfg.DIR_DEBUG / ("%s_haar.jpg" % caminho.stem),
                detector_haar.desenhar(trabalho, caixas_haar),
            )

    if not deteccoes:
        registro["tempo_s"] = round(time.perf_counter() - inicio, 3)
        return registro

    principal = deteccoes[0]
    registro.update(
        detectada=True,
        score_deteccao=principal.score,
        metodo_deteccao=principal.origem,
        angulo=principal.angulo,
        n_caracteres=len(principal.caracteres),
        caixa=[int(v) for v in principal.caixa],
    )

    # --- Etapa 3: pre-processamento (varias variantes) -------------------- #
    variantes = pre.gerar_variantes(principal.tarja, principal.caracteres)
    if not variantes:
        registro["tempo_s"] = round(time.perf_counter() - inicio, 3)
        return registro

    # --- Etapa 4: OCR ------------------------------------------------------ #
    leitura = reconhecer(variantes, **kwargs)
    registro.update({k: v for k, v in leitura.items() if k != "tentativas"})
    registro["tentativas"] = leitura.get("tentativas", [])
    registro["formato_valido"] = pos.valida(registro["texto_corrigido"])

    # Leituras das deteccoes secundarias (ex.: a 2a placa da imagem 2).
    for extra in deteccoes[1:]:
        if extra.score < 0.55 or not extra.caracteres:
            continue
        vars_extra = pre.gerar_variantes(extra.tarja, extra.caracteres)
        if not vars_extra:
            continue
        leitura_extra = reconhecer(vars_extra, **kwargs)
        # So reporta a deteccao secundaria se a leitura tiver formato de placa
        # valido e for DIFERENTE da principal -- caso contrario seria ruido
        # (texto da carroceria, grade do radiador) ou a mesma placa recortada
        # de outro jeito.
        ja_vistas = {registro["texto_corrigido"]} | {
            e["texto"] for e in registro["deteccoes_extras"]
        }
        if (pos.valida(leitura_extra["texto_corrigido"])
                and leitura_extra["texto_corrigido"] not in ja_vistas):
            registro["deteccoes_extras"].append({
                "texto": leitura_extra["texto_corrigido"],
                "score": extra.score,
                "metodo": extra.origem,
            })

    # --- Artefatos visuais (conferencia e relatorio) ---------------------- #
    if salvar_artefatos:
        raiz = caminho.stem
        salvar_imagem(cfg.DIR_DEBUG / ("%s_deteccao.jpg" % raiz),
                      det.desenhar_deteccoes(imagem, deteccoes))
        salvar_imagem(cfg.DIR_CROPS / ("%s_placa.png" % raiz), principal.placa)
        salvar_imagem(cfg.DIR_CROPS / ("%s_tarja.png" % raiz), principal.tarja)
        mosaico = pre.montar_mosaico(variantes)
        if mosaico is not None:
            salvar_imagem(cfg.DIR_DEBUG / ("%s_preproc.png" % raiz), mosaico)

    registro["tempo_s"] = round(time.perf_counter() - inicio, 3)
    return registro


# --------------------------------------------------------------------------- #
# Execucao sobre o dataset inteiro
# --------------------------------------------------------------------------- #
def executar(motor="tesseract", diretorio=None, salvar_artefatos=True,
             verboso=True, detector="classico", sem_roi=False, **kwargs):
    """
    Roda o pipeline inteiro sobre o dataset com o motor escolhido.

    Devolve (resultados, metricas) e grava os arquivos de saida do motor.

    `detector` e `sem_roi` selecionam as variacoes experimentais. A execucao de
    referencia (detector classico, com ROI) grava em `resultados/<motor>/`; as
    variacoes gravam em `resultados/experimentos/...`, para que um experimento
    nunca sobrescreva o entregavel.
    """
    cfg.garantir_diretorios()

    if motor == "tesseract":
        from . import ocr_tesseract as ocr
        rotulo = "Tesseract OCR"
    elif motor == "easyocr":
        from . import ocr_easyocr as ocr
        rotulo = "EasyOCR"
    else:
        raise ValueError("motor deve ser 'tesseract' ou 'easyocr'")

    destino = cfg.dir_saida(motor, detector=detector, sem_roi=sem_roi)

    try:
        versao = ocr.versao()
    except Exception as exc:                      # pragma: no cover
        raise RuntimeError("Motor '%s' indisponivel: %s" % (motor, exc))

    # A funcao de reconhecimento muda no modo sem ROI: o OCR passa a receber a
    # foto inteira, nao uma tarja ja recortada e pre-processada.
    if sem_roi:
        from . import baseline_sem_roi
        reconhecer = baseline_sem_roi.reconhecer(motor)
        modo = "SEM ROI (OCR direto na imagem inteira)"
    else:
        reconhecer = ocr.reconhecer
        modo = "com ROI, detector=%s" % detector
        if detector in ("haar", "ambos"):
            from . import detector_haar
            modo += " [%s]" % detector_haar.versao()

    imagens = listar_imagens(diretorio)
    if verboso:
        print("=" * 72)
        print("Pipeline: %s (versao %s)" % (rotulo, versao))
        print("Modo    : %s" % modo)
        print("Dataset : %s  (%d imagens)" % (cfg.DIR_DATASET, len(imagens)))
        print("=" * 72)

    resultados = []
    for i, caminho in enumerate(imagens, start=1):
        registro = processar_imagem(
            caminho, reconhecer, salvar_artefatos=salvar_artefatos,
            detector=detector, sem_roi=sem_roi, **kwargs
        )
        resultados.append(registro)
        if verboso:
            print("[%2d/%d] %-10s -> %-9s (conf %.2f, %s, %.2fs)" % (
                i, len(imagens), registro["imagem"],
                registro["texto_corrigido"] or "(vazio)",
                registro.get("confianca", 0.0),
                registro.get("variante", "-") or "-",
                registro["tempo_s"],
            ))

    gabarito = evaluate.carregar_gabarito()
    metricas = evaluate.avaliar(resultados, gabarito)
    metricas["motor"] = rotulo
    metricas["versao"] = versao
    metricas["modo"] = modo
    metricas["detector"] = "-" if sem_roi else detector
    metricas["sem_roi"] = bool(sem_roi)
    # Caixas cruas do Haar: mede o detector em si, nao o pipeline inteiro.
    if detector in ("haar", "ambos") and not sem_roi:
        metricas["imagens_com_caixa_haar"] = sum(
            1 for r in resultados if r.get("caixas_haar", 0) > 0
        )
        metricas["caixas_haar_total"] = sum(
            r.get("caixas_haar", 0) for r in resultados
        )

    gravar_saidas(motor, rotulo, versao, resultados, metricas, gabarito, destino)

    if verboso:
        print("-" * 72)
        print("Acerto exato        : %5.1f%%  (%d/%d)" % (
            metricas["acerto_exato_pct"], metricas["acertos"], metricas["avaliadas"]))
        print("Acuracia por caract.: %5.1f%%" % metricas["acuracia_caractere_pct"])
        if not sem_roi:
            print("Placas segmentadas  : %d/%d" % (
                metricas["placas_detectadas"], metricas["imagens"]))
            if "imagens_com_caixa_haar" in metricas:
                print("Imagens com caixa Haar: %d/%d (%d caixas cruas)" % (
                    metricas["imagens_com_caixa_haar"], metricas["imagens"],
                    metricas["caixas_haar_total"]))
        print("Tempo medio/imagem  : %5.2fs" % metricas["tempo_medio_s"])
        print("Saida em            : %s" % destino)
    return resultados, metricas


# --------------------------------------------------------------------------- #
# Relatorios
# --------------------------------------------------------------------------- #
def gravar_saidas(motor, rotulo, versao, resultados, metricas, gabarito, destino):
    """Grava os .txt, .csv e .json de um pipeline."""
    destino.mkdir(parents=True, exist_ok=True)
    por_imagem = destino / "por_imagem"
    por_imagem.mkdir(exist_ok=True)

    # (1) Arquivo .txt consolidado -- entregavel principal da atividade.
    linhas = [
        "RESULTADO DO OCR - %s" % rotulo.upper(),
        "Versao do motor : %s" % versao,
        "Modo            : %s" % metricas.get("modo", "com ROI, detector=classico"),
        "Dataset         : %s" % cfg.DIR_DATASET.name,
        "Gerado em       : %s" % datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "",
        "%-10s %-10s %-10s %-9s %-6s %-12s %s" % (
            "IMAGEM", "LIDO", "CORRIGIDO", "GABARITO", "OK", "VARIANTE", "CONF"),
        "-" * 78,
    ]
    for r in resultados:
        esperado = gabarito.get(r["imagem"], "")
        ok = "-" if not esperado else ("SIM" if r["texto_corrigido"] == esperado else "NAO")
        linhas.append("%-10s %-10s %-10s %-9s %-6s %-12s %.2f" % (
            r["imagem"], r["texto"] or "-", r["texto_corrigido"] or "-",
            esperado or "-", ok, r.get("variante", "-") or "-",
            r.get("confianca", 0.0),
        ))
        for extra in r.get("deteccoes_extras", []):
            linhas.append("%-10s %-10s %-10s %-9s %-6s %s" % (
                "", "", extra["texto"], "", "", "(2a placa detectada na imagem)"))

    linhas += ["-" * 78, "", "RESUMO",
               "  Imagens processadas       : %d" % metricas["imagens"]]

    if metricas.get("sem_roi"):
        # Sem ROI nao existe etapa de deteccao: reportar 0% de deteccao seria
        # confundir "nao detectou" com "nao ha detector".
        linhas.append("  Etapa de deteccao         : nao se aplica (OCR direto)")
    else:
        linhas.append("  Placas detectadas         : %d (%.1f%%)" % (
            metricas["placas_detectadas"], metricas["taxa_deteccao"]))
        if "imagens_com_caixa_haar" in metricas:
            linhas.append(
                "  Imagens com caixa Haar    : %d (%d caixas cruas no total)" % (
                    metricas["imagens_com_caixa_haar"],
                    metricas["caixas_haar_total"]))

    linhas += [
        "  Acerto exato (com mascara): %d/%d (%.1f%%)" % (
            metricas["acertos"], metricas["avaliadas"], metricas["acerto_exato_pct"]),
        "  Acerto exato (sem mascara): %d/%d (%.1f%%)" % (
            metricas["acertos_sem_mascara"], metricas["avaliadas"],
            metricas["acerto_exato_sem_mascara_pct"]),
        "  Acuracia por caractere    : %.1f%%" % metricas["acuracia_caractere_pct"],
        "  Tempo total / por imagem  : %.2fs / %.2fs" % (
            metricas["tempo_total_s"], metricas["tempo_medio_s"]),
        "",
    ]
    arquivo_txt = destino / ("placas_%s.txt" % motor)
    arquivo_txt.write_text("\n".join(linhas), encoding="utf-8")

    # (2) Um .txt por imagem, contendo apenas a placa reconhecida.
    for r in resultados:
        nome = r["imagem"].rsplit(".", 1)[0] + ".txt"
        (por_imagem / nome).write_text(
            (r["texto_corrigido"] or "") + "\n", encoding="utf-8"
        )

    # (3) CSV para analise quantitativa.
    with open(destino / ("resultados_%s.csv" % motor), "w",
              encoding="utf-8", newline="") as f:
        campos = [
            "imagem", "gabarito", "texto", "texto_corrigido", "acertou",
            "acuracia_caractere", "confianca", "variante", "detector",
            "metodo_deteccao", "score_deteccao", "angulo", "n_caracteres",
            "caixas_haar", "formato_valido", "tempo_s",
        ]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for r in resultados:
            esperado = gabarito.get(r["imagem"], "")
            escritor.writerow({
                "imagem": r["imagem"],
                "gabarito": esperado,
                "texto": r["texto"],
                "texto_corrigido": r["texto_corrigido"],
                "acertou": int(bool(esperado) and r["texto_corrigido"] == esperado),
                "acuracia_caractere": round(
                    evaluate.acuracia_caractere(r["texto_corrigido"], esperado), 4),
                "confianca": r.get("confianca", 0.0),
                "variante": r.get("variante", ""),
                "detector": r.get("detector", ""),
                "metodo_deteccao": r.get("metodo_deteccao", ""),
                "score_deteccao": r.get("score_deteccao", 0.0),
                "angulo": r.get("angulo", 0.0),
                "n_caracteres": r.get("n_caracteres", 0),
                "caixas_haar": r.get("caixas_haar", 0),
                "formato_valido": int(r.get("formato_valido", False)),
                "tempo_s": r.get("tempo_s", 0.0),
            })

    # (4) JSON completo (inclui todas as tentativas de pre-processamento).
    with open(destino / ("resultados_%s.json" % motor), "w", encoding="utf-8") as f:
        json.dump(
            {"motor": rotulo, "versao": versao, "metricas": metricas,
             "resultados": resultados},
            f, ensure_ascii=False, indent=2,
        )

    return arquivo_txt
