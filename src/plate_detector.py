"""
Etapa 1 e 2 do pipeline: DETECCAO, SEGMENTACAO e EXTRACAO da placa veicular.

Abordagem 100% visao computacional classica (sem rede neural de deteccao),
organizada em quatro estagios:

  1) GERACAO DE CANDIDATOS -- tres detectores independentes e complementares:
       (A) Faixa azul Mercosul: segmentacao por cor em HSV. E o indicio mais
           especifico do dataset -- toda placa Mercosul tem um cabecalho azul
           horizontal ("MERCOSUL / BRASIL" + bandeira) no topo.
       (B) Densidade de bordas verticais: Sobel em X + fechamento morfologico.
           Os caracteres produzem muitas transicoes verticais proximas, que o
           fechamento funde num blob retangular alongado.
       (C) Regioes claras e pouco saturadas: o corpo da placa e branco/cinza,
           o que o separa bem da lataria colorida do veiculo.

  2) RETIFICACAO -- a faixa azul da a orientacao exata da placa. A imagem e
     rotacionada para deixar a faixa horizontal, corrigindo inclinacao e
     rotacao antes de qualquer recorte (ver imagens 2, 5, 19 e 20 do dataset).

  3) SEGMENTACAO DE CARACTERES -- analise de componentes conexos sob varias
     binarizacoes e varios tamanhos de janela, escolhendo a LINHA de
     componentes mutuamente coerentes. Descarta a sigla "BR", o texto pequeno
     da faixa azul e a sujeira da moldura.

  4) RECORTE DA PLACA E DA TARJA -- a geometria final e ancorada nos proprios
     caracteres (referencia bem mais estavel que a mascara de cor): a tarja e
     a caixa que os envolve e a placa e reconstruida a partir dela.

A contagem de caracteres encontrados realimenta a pontuacao do candidato --
uma placa brasileira tem exatamente 7 componentes alinhados, e esse e o sinal
que melhor separa um recorte correto de um recorte parcial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as cfg


# --------------------------------------------------------------------------- #
# Estruturas de dados
# --------------------------------------------------------------------------- #
@dataclass
class Deteccao:
    """Uma placa detectada, retificada e recortada."""

    score: float                 # pontuacao heuristica do candidato
    origem: str                  # detector que gerou o candidato
    quadrilatero: np.ndarray     # 4 cantos da placa na imagem ORIGINAL
    angulo: float                # inclinacao corrigida, em graus
    placa: np.ndarray = field(repr=False, default=None)   # placa retificada
    tarja: np.ndarray = field(repr=False, default=None)   # so os caracteres
    caracteres: list = field(repr=False, default_factory=list)  # caixas dos chars

    @property
    def caixa(self):
        """Bounding box axis-aligned da placa na imagem original (x, y, w, h)."""
        return cv2.boundingRect(self.quadrilatero.astype(np.int32))


# --------------------------------------------------------------------------- #
# Proporcoes verticais da placa Mercosul (medidas no proprio dataset).
# As placas aparecem com larguras esticadas, mas as razoes VERTICAIS em
# relacao a altura da faixa azul sao estaveis -- por isso a reconstrucao da
# geometria e ancorada na altura da faixa (hf), nunca na razao de aspecto.
# --------------------------------------------------------------------------- #
TOPO_PLACA_EM_HF = 0.60      # quanto subir acima da faixa para pegar a moldura
BASE_PLACA_EM_HF = 3.40      # quanto descer abaixo da faixa ate a base da placa
FOLGA_LATERAL = 0.035        # folga horizontal relativa a largura da faixa
# A janela de BUSCA dos caracteres e deliberadamente mais larga que a placa:
# os detectores (B) e (C) entregam uma extensao horizontal imprecisa e, sem a
# folga, o primeiro/ultimo caractere acaba cortado. O filtro de componentes
# conexos se encarrega de descartar o que nao for caractere.
FOLGA_BUSCA = {"faixa_azul": 0.09, "bordas": 0.16, "regiao_clara": 0.16,
               # O Haar entrega uma caixa de extensao horizontal imprecisa
               # (treinada em placa russa, mais estreita), entao recebe a
               # folga maior.
               "haar": 0.16}


# --------------------------------------------------------------------------- #
# Utilitarios de imagem
# --------------------------------------------------------------------------- #
def _redimensionar(imagem, largura):
    """Normaliza a largura da imagem; devolve a imagem e a escala aplicada."""
    h, w = imagem.shape[:2]
    if w == largura:
        return imagem, 1.0
    escala = largura / float(w)
    nova = cv2.resize(
        imagem,
        (largura, max(1, int(round(h * escala)))),
        interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC,
    )
    return nova, escala


def _mascara_azul(hsv):
    """Mascara binaria do azul do cabecalho Mercosul."""
    return cv2.inRange(hsv, np.array(cfg.HSV_AZUL_MIN), np.array(cfg.HSV_AZUL_MAX))


def _mascara_clara(hsv):
    """Mascara das regioes claras e pouco saturadas (corpo branco da placa)."""
    _, s, v = cv2.split(hsv)
    # Limiar adaptativo no canal V: tolera sombra e superexposicao.
    limiar_v = max(110, int(np.percentile(v, 70)))
    return cv2.inRange(s, 0, 90) & cv2.inRange(v, limiar_v, 255)


def _ordenar_cantos(pts):
    """Ordena 4 pontos como [sup-esq, sup-dir, inf-dir, inf-esq]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    soma = pts.sum(axis=1)
    dif = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(soma)],   # superior esquerdo  (x+y minimo)
            pts[np.argmin(dif)],    # superior direito   (x-y maximo)
            pts[np.argmax(soma)],   # inferior direito   (x+y maximo)
            pts[np.argmax(dif)],    # inferior esquerdo  (x-y minimo)
        ],
        dtype=np.float32,
    )


def _aplicar_afim(M, pontos):
    """Aplica uma matriz afim 2x3 a um conjunto de pontos (N, 2)."""
    pts = np.hstack([np.asarray(pontos, np.float32), np.ones((len(pontos), 1), np.float32)])
    return (M @ pts.T).T


def _recortar_seguro(imagem, x0, y0, x1, y1):
    """Recorte com clamp nos limites da imagem."""
    h, w = imagem.shape[:2]
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, (x0, y0, 0, 0)
    return imagem[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def _iou(a, b):
    """Intersecao sobre uniao entre duas caixas (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    uniao = aw * ah + bw * bh - inter
    return inter / uniao if uniao > 0 else 0.0


# --------------------------------------------------------------------------- #
# (A) Candidatos pela faixa azul Mercosul
# --------------------------------------------------------------------------- #
def _candidatos_faixa_azul(hsv):
    """
    Localiza as barras azuis horizontais do cabecalho Mercosul.

    Devolve, para cada barra, os 4 cantos da barra (via minAreaRect), o que ja
    carrega posicao, largura, altura E inclinacao da placa.
    """
    altura_img, largura_img = hsv.shape[:2]
    mascara = _mascara_azul(hsv)

    # Fechamento horizontal: funde "MERCOSUL", o texto "BRASIL" e a bandeira
    # numa unica barra continua.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, largura_img // 40), 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    )

    candidatos = []
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        rect = cv2.minAreaRect(c)
        lado_maior, lado_menor = max(rect[1]), min(rect[1])
        # IMPORTANTE: as medidas vem do retangulo de area minima, nao do
        # bounding box axis-aligned. Para uma barra inclinada o bounding box
        # infla muito, e os testes de proporcao/solidez falhariam (placas
        # tortas das imagens 2, 5, 19 e 20).
        if lado_menor < 4 or lado_maior < 24 or lado_maior / lado_menor < 2.5:
            continue
        if cv2.contourArea(c) / (lado_maior * lado_menor) < 0.60:  # barra cheia
            continue
        candidatos.append((cv2.boxPoints(rect), "faixa_azul"))
    return candidatos


# --------------------------------------------------------------------------- #
# (B) Candidatos por densidade de bordas verticais
# --------------------------------------------------------------------------- #
def _candidatos_bordas(cinza):
    """Sobel em X + fechamento morfologico: agrupa os caracteres num blob."""
    suave = cv2.bilateralFilter(cinza, 9, 75, 75)
    sobel = cv2.convertScaleAbs(cv2.Sobel(suave, cv2.CV_16S, 1, 0, ksize=3))
    _, binaria = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    fechada = cv2.morphologyEx(
        binaria, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)), iterations=2,
    )
    fechada = cv2.morphologyEx(
        fechada, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    )

    candidatos = []
    contornos, _ = cv2.findContours(fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w < 48 or h < 16:
            continue
        if not (cfg.RAZAO_ASPECTO_MIN <= w / float(h) <= cfg.RAZAO_ASPECTO_MAX):
            continue
        # O blob cobre so os caracteres: a faixa azul equivalente fica acima.
        # Convertemos para "pseudo-faixa" = terco superior da caixa.
        hf = h * 0.30
        cantos = np.array(
            [[x, y - hf], [x + w, y - hf], [x + w, y], [x, y]], dtype=np.float32
        )
        candidatos.append((cantos, "bordas"))
    return candidatos


# --------------------------------------------------------------------------- #
# (C) Candidatos por regioes claras
# --------------------------------------------------------------------------- #
def _candidatos_regiao_clara(hsv):
    """Blobs claros e pouco saturados com proporcao compativel com uma placa."""
    mascara = _mascara_clara(hsv)
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    )
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    candidatos = []
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w < 48 or h < 16:
            continue
        if not (cfg.RAZAO_ASPECTO_MIN <= w / float(h) <= cfg.RAZAO_ASPECTO_MAX):
            continue
        # A regiao clara e o corpo da placa (sem o cabecalho azul): a faixa
        # equivalente fica logo acima, com ~1/3 da altura do corpo.
        hf = h * 0.33
        cantos = np.array(
            [[x, y - hf], [x + w, y - hf], [x + w, y], [x, y]], dtype=np.float32
        )
        candidatos.append((cantos, "regiao_clara"))
    return candidatos


# --------------------------------------------------------------------------- #
# Segmentacao de caracteres (componentes conexos)
# --------------------------------------------------------------------------- #
def _variantes_binarias(cinza):
    """
    Gera varias binarizacoes da janela, deixando os CARACTERES em branco.

    Nenhum limiar unico funciona em todo o dataset: ha placas bem expostas
    (Otsu global resolve), placas lavadas/de baixo contraste -- cinza sobre
    prata, como as imagens 5, 6 e 8 -- que so aparecem depois de equalizacao
    local (CLAHE), e placas com sombra parcial ou reflexo, em que so o limiar
    adaptativo funciona. Produzimos todas e deixamos a segmentacao escolher.
    """
    suave = cv2.bilateralFilter(cinza, 5, 50, 50)
    # CLAHE: equalizacao de histograma local -- recupera caracteres de baixo
    # contraste sem estourar as regioes ja bem expostas.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(suave)

    variantes = []
    for origem in (suave, clahe):
        _, otsu = cv2.threshold(origem, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        variantes.append(otsu)

    bloco = max(15, (cinza.shape[0] // 2) | 1)      # impar, ~metade da janela
    for b, c in ((bloco, 10), (min(255, 2 * bloco + 1), 14)):
        variantes.append(
            cv2.adaptiveThreshold(
                clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, b, c,
            )
        )

    # Black-hat: realca estruturas ESCURAS menores que o elemento estruturante,
    # removendo a variacao lenta do fundo. E o que salva placas prateadas com
    # caracteres cinza-claro (imagem 5), onde o contraste absoluto e minimo.
    # Elemento retangular (separavel, portanto muito mais barato que um
    # elipsoidal) com lado um pouco maior que um caractere.
    lado = max(3, min(45, (cinza.shape[0] // 2) | 1))
    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (lado, lado))
    blackhat = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, elem)
    _, bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variantes.append(bh)

    # Versoes erodidas das binarizacoes mais ruidosas: rompem as pontes finas
    # (1-2 px) que ligam os caracteres a moldura da placa. Sem isso, toda a
    # sequencia vira um unico componente gigante e e descartada pela forma.
    nucleo = np.ones((3, 3), np.uint8)
    variantes += [cv2.erode(v, nucleo) for v in variantes[2:]]
    return variantes


def _componentes(binaria, altura):
    """Componentes conexos com forma compativel com um caractere."""
    binaria = cv2.morphologyEx(
        binaria, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    caixas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < max(8, 0.10 * altura) or h > 0.98 * altura or w < 2:
            continue
        # Forma tipica de caractere. O limite superior e folgado (1.55) porque
        # varias placas do dataset aparecem esticadas na horizontal -- nelas um
        # "0" chega a ser mais largo que alto.
        if not (0.06 <= w / float(h) <= 1.55):
            continue
        if area < 0.12 * w * h:                    # componente oco/esparso
            continue
        caixas.append((int(x), int(y), int(w), int(h)))
    return caixas


def _nota_linha(linha, altura_referencia):
    """
    Avalia o quanto um conjunto de componentes parece a sequencia de uma placa:
    quantidade proxima de 7, alturas e larguras uniformes e caracteres grandes.

    `altura_referencia` e a maior altura mediana observada entre TODAS as
    tentativas da imagem, o que torna a nota comparavel entre janelas de busca
    de tamanhos diferentes e penaliza linhas pequenas -- tipicamente o texto
    "BRASIL"/"MERCOSUL" da faixa ou o adesivo da concessionaria.
    """
    if len(linha) < 3:
        return -1e9
    alturas = np.array([b[3] for b in linha], dtype=np.float32)
    larguras = np.array([b[2] for b in linha], dtype=np.float32)
    nota = -1.5 * abs(len(linha) - cfg.TAMANHO_PLACA)
    nota += 2.5 * min(1.0, float(np.median(alturas)) / max(1.0, altura_referencia))
    nota -= 3.0 * float(alturas.std() / max(1.0, alturas.mean()))   # uniformidade
    nota -= 1.5 * float(larguras.std() / max(1.0, larguras.mean()))
    return float(nota)


def _localizar_caracteres(rotacionada, faixa, origem):
    """
    Procura a linha de caracteres variando o tamanho da janela de busca.

    Duas incertezas justificam a busca: (i) a ALTURA da faixa azul e ruidosa
    (gradientes e reflexos esticam a mascara de cor), de modo que uma janela
    fixa ora corta os caracteres ora engloba meio para-choque -- e area demais
    desvia o limiar de Otsu (foi o que quebrava a imagem 5); (ii) a extensao
    HORIZONTAL do candidato pode cortar o primeiro ou o ultimo caractere.

    Testamos entao tres alturas e duas folgas laterais, com todas as
    binarizacoes, e ficamos com a melhor linha segundo `_nota_linha`.

    Devolve (caracteres, janela_recortada, caixa_da_janela).
    """
    fx, fy, fw, fh = faixa
    base = FOLGA_BUSCA.get(origem, 0.10)
    # Tres alturas: justa (Otsu enxerga quase so a placa), intermediaria e
    # folgada (garante que a linha inteira cabe mesmo com a faixa subestimada).
    alturas = sorted({3.2, 4.2, max(5.5, 0.30 * fw / max(fh, 1.0))})

    tentativas = []
    for fator in alturas:
        for folga in (base, base + 0.13):
            janela, caixa = _recortar_seguro(
                rotacionada,
                fx - folga * fw, fy, fx + fw + folga * fw, fy + fator * fh,
            )
            if janela is None:
                continue
            cinza = cv2.cvtColor(janela, cv2.COLOR_BGR2GRAY)
            for binaria in _variantes_binarias(cinza):
                linha = _linha_de_caracteres(_componentes(binaria, cinza.shape[0]))
                if len(linha) >= 3:
                    tentativas.append((linha, janela, caixa))

    if not tentativas:
        # Nenhuma linha: devolve a janela padrao para o recorte geometrico.
        janela, caixa = _recortar_seguro(
            rotacionada, fx - base * fw, fy, fx + fw + base * fw, fy + 4.5 * fh
        )
        return [], janela, caixa

    h_ref = max(float(np.median([b[3] for b in l])) for l, _, _ in tentativas)
    return max(tentativas, key=lambda t: _nota_linha(t[0], h_ref))


def _linha_de_caracteres(brutos):
    """
    Extrai, de um conjunto de componentes conexos, a LINHA alfanumerica.

    O criterio nao depende de limiares absolutos (que variariam com o tamanho
    da placa na foto), e sim de coerencia mutua:

      1. VOTACAO -- cada componente "apoia" os que tem altura semelhante e
         centro vertical proximo; vence o grupo com maior suporte, ponderado
         pela altura. E isso que descarta o texto pequeno "BRASIL"/"MERCOSUL"
         da faixa azul e a sigla "BR" do canto inferior.
      2. REPESCAGEM -- uma segunda passada, com tolerancia maior e restrita a
         vizinhanca da linha ja encontrada, recupera caracteres que ficaram de
         fora por borrao, sombra ou uma letra mais estreita.

    Devolve as caixas (x, y, w, h) ordenadas da esquerda para a direita.
    """
    if not brutos:
        return []

    def _coerentes(ref, tol_h, tol_c):
        """Componentes com altura e linha de base compativeis com `ref`."""
        hr = ref[3]
        cr = ref[1] + hr / 2.0
        return [
            b for b in brutos
            if (1.0 - tol_h) * hr <= b[3] <= (1.0 + tol_h) * hr
            and abs((b[1] + b[3] / 2.0) - cr) <= tol_c * hr
        ]

    # (3) Votacao: o melhor grupo e o mais numeroso; a altura entra como
    #     desempate para privilegiar os caracteres grandes da placa.
    melhor, melhor_nota = [], -1.0
    for ref in brutos:
        grupo = _coerentes(ref, tol_h=0.30, tol_c=0.40)
        nota = len(grupo) * (ref[3] ** 0.5)
        if nota > melhor_nota:
            melhor, melhor_nota = grupo, nota
    if not melhor:
        return []

    # (4) Repescagem com tolerancia maior, restrita a vizinhanca da linha.
    h_med = float(np.median([b[3] for b in melhor]))
    w_med = float(np.median([b[2] for b in melhor]))
    c_med = float(np.median([b[1] + b[3] / 2.0 for b in melhor]))
    esq = min(b[0] for b in melhor) - 1.7 * w_med
    dir_ = max(b[0] + b[2] for b in melhor) + 1.7 * w_med

    linha = []
    for b in brutos:
        if not (0.50 * h_med <= b[3] <= 1.55 * h_med):
            continue
        if abs((b[1] + b[3] / 2.0) - c_med) > 0.45 * h_med:
            continue
        if b[0] + b[2] < esq or b[0] > dir_:
            continue
        linha.append(b)

    linha.sort(key=lambda b: b[0])
    return linha


# --------------------------------------------------------------------------- #
# Retificacao e recorte a partir da faixa
# --------------------------------------------------------------------------- #
def _extrair_candidato(imagem, cantos_faixa, origem):
    """
    A partir dos 4 cantos da faixa azul (ou pseudo-faixa), retifica a imagem,
    recorta a placa inteira, isola a tarja de caracteres e pontua o candidato.
    """
    cantos = _ordenar_cantos(cantos_faixa)
    tl, tr, br, bl = cantos

    # Inclinacao medida na aresta superior da faixa.
    vetor = tr - tl
    angulo = math.degrees(math.atan2(float(vetor[1]), float(vetor[0])))
    if abs(angulo) > 35:                      # faixa quase vertical: descarta
        return None

    centro = tuple(cantos.mean(axis=0).tolist())
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotacionada = cv2.warpAffine(
        imagem, M, (imagem.shape[1], imagem.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    # Caixa da faixa ja na imagem endireitada.
    fx, fy, fw, fh = cv2.boundingRect(_aplicar_afim(M, cantos).astype(np.int32))
    if fw < 20 or fh < 4:
        return None

    # Janela de busca + segmentacao dos caracteres.
    chars, busca, caixa_busca = _localizar_caracteres(
        rotacionada, (fx, fy, fw, fh), origem
    )
    if busca is None:
        return None
    bx, by, bw, bh = caixa_busca

    if chars:
        # --- Geometria ancorada nos CARACTERES (mais estavel que a faixa) -- #
        cx0 = min(c[0] for c in chars)
        cy0 = min(c[1] for c in chars)
        cx1 = max(c[0] + c[2] for c in chars)
        cy1 = max(c[1] + c[3] for c in chars)
        hc = cy1 - cy0
        margem_x = 0.04 * (cx1 - cx0)
        margem_y = 0.16 * hc
        tarja, caixa_tarja = _recortar_seguro(
            busca, cx0 - margem_x, cy0 - margem_y, cx1 + margem_x, cy1 + margem_y
        )
        # Reposiciona as caixas dos caracteres no referencial da TARJA (e nao
        # no da janela de busca) -- o recorte pode ter sido limitado pelas
        # bordas da imagem, entao usamos a origem efetivamente aplicada.
        tx, ty = caixa_tarja[0], caixa_tarja[1]
        chars = [(c[0] - tx, c[1] - ty, c[2], c[3]) for c in chars]

        # Placa inteira: a linha de caracteres fica a ~0.95 h abaixo do topo
        # da placa e a ~0.50 h acima da base (proporcoes medidas no dataset).
        y_topo = by + cy0 - 0.95 * hc
        y_base = by + cy1 + 0.50 * hc
    else:
        # Sem linha de caracteres confiavel: recorte puramente geometrico.
        tarja, _ = _recortar_seguro(busca, 0.10 * bw, 1.05 * fh, bw, 0.80 * bh)
        y_topo = fy - TOPO_PLACA_EM_HF * fh
        y_base = fy + fh + BASE_PLACA_EM_HF * fh
    if tarja is None or tarja.size == 0:
        return None

    folga = FOLGA_LATERAL * fw
    x0, x1 = fx - folga, fx + fw + folga
    placa, _ = _recortar_seguro(rotacionada, x0, y_topo, x1, y_base)
    if placa is None:
        return None

    score = _pontuar(placa, tarja, chars, imagem.shape, origem)

    # Quadrilatero da placa de volta nas coordenadas da imagem ORIGINAL.
    Minv = cv2.invertAffineTransform(M)
    quad_ret = np.array(
        [[x0, y_topo], [x1, y_topo], [x1, y_base], [x0, y_base]], dtype=np.float32
    )
    quad = _aplicar_afim(Minv, quad_ret).astype(np.float32)

    return Deteccao(
        score=round(float(score), 4),
        origem=origem,
        quadrilatero=quad,
        angulo=round(float(angulo), 2),
        placa=placa,
        tarja=tarja,
        caracteres=chars,
    )


# --------------------------------------------------------------------------- #
# Pontuacao do candidato
# --------------------------------------------------------------------------- #
def _pontuar(placa, tarja, chars, shape_original, origem):
    """
    Heuristica de verossimilhanca "isto e uma placa", em [0, 1].

    Evidencias combinadas (cada uma normalizada em [0, 1]):
      * numero de caracteres segmentados -- exatamente 7 numa placa brasileira;
      * presenca de azul no cabecalho (assinatura Mercosul);
      * corpo claro e pouco saturado;
      * contraste interno alto (texto escuro sobre fundo claro);
      * tamanho relativo da placa (desempata a favor da placa principal,
        evitando adesivos e plaquinhas de concessionaria);
      * razao largura/altura da tarja (~4.3 para 7 caracteres).
    """
    if placa is None or placa.size == 0:
        return 0.0
    hp, wp = placa.shape[:2]
    hsv = cv2.cvtColor(placa, cv2.COLOR_BGR2HSV)

    topo = hsv[: max(1, int(0.34 * hp))]
    base = hsv[int(0.40 * hp):]
    s_azul = min(1.0, (np.count_nonzero(_mascara_azul(topo)) / max(1, topo[..., 0].size)) / 0.55)
    s_claro = min(1.0, (np.count_nonzero(_mascara_clara(base)) / max(1, base[..., 0].size)) / 0.45)

    # Contagem de caracteres: a placa brasileira tem exatamente 7. A curva e
    # proposital mente abrupta -- um recorte que corta um caractere costuma
    # pontuar bem em todos os outros criterios, entao a contagem precisa ter
    # peso suficiente para desempatar sozinha.
    n = len(chars)
    s_chars = {0: 0.0, 1: 0.55, 2: 0.25}.get(abs(n - cfg.TAMANHO_PLACA), 0.0) if n else 0.0
    if n == cfg.TAMANHO_PLACA:
        s_chars = 1.0

    # Forma da tarja: 7 caracteres com o espacamento padrao produzem uma faixa
    # de razao largura/altura em torno de 4.3. Este termo e o que distingue a
    # linha da placa de outras linhas de texto encontradas na cena -- em
    # especial o proprio "BRASIL" da faixa azul, que e muito mais achatado.
    ht, wt = tarja.shape[:2]
    razao_tarja = wt / float(max(1, ht))
    s_forma = max(0.0, 1.0 - abs(razao_tarja - 4.3) / 3.0)

    cinza = cv2.cvtColor(tarja, cv2.COLOR_BGR2GRAY)
    s_contraste = min(1.0, float(cinza.std()) / 55.0)

    area_rel = (wp * hp) / float(shape_original[0] * shape_original[1])
    if not (cfg.AREA_RELATIVA_MIN <= area_rel <= cfg.AREA_RELATIVA_MAX):
        return 0.0
    s_area = min(1.0, area_rel / 0.07)

    score = (
        0.30 * s_chars
        + 0.22 * s_azul
        + 0.10 * s_claro
        + 0.10 * s_contraste
        + 0.12 * s_area
        + 0.16 * s_forma
    )
    if origem == "faixa_azul":       # indicio mais confiavel: leve bonus
        score += 0.05
    return min(1.0, score)


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def detectar_placas(imagem, maximo=3, limiar=0.30, detector="classico"):
    """
    Detecta, retifica, recorta e pontua as placas presentes na imagem.

    Devolve as deteccoes ordenadas da mais provavel para a menos provavel, com
    `placa` (placa inteira retificada) e `tarja` (somente os caracteres)
    prontos para o pre-processamento de OCR.

    `detector` escolhe a origem dos candidatos -- e SO isso: a retificacao, a
    segmentacao de caracteres e a pontuacao sao as mesmas nos tres casos, de
    modo que a comparacao entre eles mede o detector.

      "classico" -- os tres detectores proprios (HSV + Sobel + regiao clara).
      "haar"     -- Haar Cascade da Aula 8 (`detector_haar`).
      "ambos"    -- a uniao dos dois conjuntos de candidatos.
    """
    trabalho, escala = _redimensionar(imagem, cfg.LARGURA_TRABALHO)
    hsv = cv2.cvtColor(trabalho, cv2.COLOR_BGR2HSV)
    cinza = cv2.cvtColor(trabalho, cv2.COLOR_BGR2GRAY)

    if detector not in ("classico", "haar", "ambos"):
        raise ValueError("detector deve ser 'classico', 'haar' ou 'ambos'")

    brutos = []
    if detector in ("classico", "ambos"):
        brutos += _candidatos_faixa_azul(hsv)
        brutos += _candidatos_bordas(cinza)
        brutos += _candidatos_regiao_clara(hsv)
    if detector in ("haar", "ambos"):
        from . import detector_haar
        brutos += detector_haar.candidatos(cinza)

    deteccoes = []
    for cantos, origem in brutos:
        # Trabalha sempre na resolucao ORIGINAL (melhor qualidade para o OCR).
        d = _extrair_candidato(imagem, np.asarray(cantos, np.float32) / escala, origem)
        if d is not None and d.score >= limiar:
            deteccoes.append(d)

    deteccoes.sort(key=lambda d: d.score, reverse=True)

    # Supressao de nao-maximos: descarta deteccoes sobrepostas.
    finais = []
    for d in deteccoes:
        if all(_iou(d.caixa, f.caixa) < 0.3 for f in finais):
            finais.append(d)
        if len(finais) >= maximo:
            break
    return finais


def desenhar_deteccoes(imagem, deteccoes):
    """Imagem de debug com os quadrilateros detectados e suas pontuacoes."""
    saida = imagem.copy()
    espessura = max(2, imagem.shape[1] // 500)
    for i, d in enumerate(deteccoes):
        cor = (0, 255, 0) if i == 0 else (0, 165, 255)
        cv2.polylines(saida, [d.quadrilatero.astype(np.int32)], True, cor, espessura)
        x, y = d.quadrilatero.astype(np.int32)[0]
        cv2.putText(
            saida, "%s %.2f (%d chars)" % (d.origem, d.score, len(d.caracteres)),
            (int(x), max(22, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * espessura / 2.0, cor, espessura, cv2.LINE_AA,
        )
    return saida
