"""
Etapa 5 do pipeline: POS-PROCESSAMENTO do texto reconhecido.

O OCR devolve texto livre; a placa brasileira tem formato conhecido. Esta
etapa transforma um em outro, em tres passos:

  1. LIMPEZA   -- maiusculas, remocao de acentos/simbolos e descarte dos
                  rotulos fixos da placa ("BRASIL", "MERCOSUL", "BR").
  2. RECORTE   -- se sobrarem mais de 7 caracteres, escolhe a janela de 7 que
                  melhor se encaixa no formato oficial.
  3. MASCARA   -- corrige confusoes classicas do OCR usando a posicao: na
                  placa Mercosul (AAA0A00) as posicoes 0,1,2 e 4 sao LETRAS e
                  as posicoes 3,5,6 sao DIGITOS, entao um "0" lido na posicao
                  1 so pode ser a letra "O", e um "S" na posicao 6 so pode
                  ser "5".

O texto limpo (sem mascara) e o corrigido sao reportados separadamente, para
que a comparacao entre os motores possa ser feita com e sem a ajuda do
conhecimento de dominio.
"""

from __future__ import annotations

import re
import unicodedata

from . import config as cfg

# Rotulos impressos na propria placa que nunca fazem parte do codigo.
RUIDOS = ("MERCOSUL", "MERCOSU", "BRASIL", "BRASI", "ERCOSUL")

# Confusoes classicas do OCR, resolvidas pela POSICAO do caractere.
# As trocas foram escolhidas pela semelhanca de forma na fonte das placas
# brasileiras (FE-Schrift). Nota: o zero da placa e um oval estreito que os
# dois motores leem com frequencia como "G" ou "C" (a textura de seguranca
# do fundo cria um tracinho interno), dai G -> 0 e C -> 0.
LETRA_PARA_DIGITO = {
    "O": "0", "Q": "0", "D": "0", "U": "0", "G": "0", "C": "0",
    "I": "1", "L": "1", "T": "1", "J": "1",
    "Z": "2", "E": "3", "A": "4", "H": "4",
    "S": "5", "B": "8", "R": "8", "Y": "9",
}
DIGITO_PARA_LETRA = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}

# AAA0A00 (Mercosul, vigente desde 2018) e AAA0000 (padrao antigo).
MASCARA_MERCOSUL = "LLLNLNN"
MASCARA_ANTIGA = "LLLNNNN"

# Todas as placas do dataset sao Mercosul (a faixa azul e detectada em todas),
# por isso a mascara Mercosul e a padrao. O formato antigo continua disponivel
# via `mascaras=AMBAS` para quem rodar o pipeline sobre outro conjunto.
MASCARAS = (MASCARA_MERCOSUL,)
AMBAS = (MASCARA_MERCOSUL, MASCARA_ANTIGA)

LETRAS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITOS = set("0123456789")


def limpar(texto):
    """Normaliza o texto cru do OCR: maiusculas, so A-Z0-9, sem rotulos fixos."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii").upper()
    for ruido in RUIDOS:                       # antes de remover separadores
        texto = texto.replace(ruido, " ")
    texto = re.sub(r"[^A-Z0-9]", "", texto)
    # "BR" so e ruido quando aparece grudado no inicio da leitura.
    if texto.startswith("BR") and len(texto) > cfg.TAMANHO_PLACA:
        texto = texto[2:]
    return texto


def _aderencia(trecho, mascara):
    """Quantos caracteres de `trecho` ja respeitam `mascara` (0..7)."""
    pontos = 0
    for c, tipo in zip(trecho, mascara):
        if (tipo == "L" and c in LETRAS) or (tipo == "N" and c in DIGITOS):
            pontos += 1
    return pontos


def recortar_sete(texto, mascaras=MASCARAS):
    """
    Reduz a leitura a 7 caracteres.

    Com mais de 7, desliza uma janela e escolhe a posicao cuja sequencia mais
    se parece com uma placa valida -- isso descarta lixo colado nas pontas
    (restos da moldura, da sigla "BR" ou da bandeira) sem precisar saber de
    que lado estava o ruido.
    """
    n = cfg.TAMANHO_PLACA
    if len(texto) <= n:
        return texto
    melhor, melhor_nota = texto[:n], -1
    for i in range(len(texto) - n + 1):
        trecho = texto[i:i + n]
        nota = max(_aderencia(trecho, m) for m in mascaras)
        if nota > melhor_nota:
            melhor, melhor_nota = trecho, nota
    return melhor


def aplicar_mascara(texto, mascaras=MASCARAS):
    """
    Corrige confusoes do OCR impondo o formato oficial da placa.

    Quando mais de uma mascara e permitida, escolhe a que exigir menos
    correcoes; so entao troca os caracteres incompativeis com a posicao.
    """
    texto = recortar_sete(texto, mascaras)
    if len(texto) != cfg.TAMANHO_PLACA:
        return texto

    mascara = max(mascaras, key=lambda m: _aderencia(texto, m))
    saida = []
    for c, tipo in zip(texto, mascara):
        if tipo == "L" and c in DIGITOS:
            saida.append(DIGITO_PARA_LETRA.get(c, c))
        elif tipo == "N" and c in LETRAS:
            saida.append(LETRA_PARA_DIGITO.get(c, c))
        else:
            saida.append(c)
    return "".join(saida)


def valida(texto, mascaras=AMBAS):
    """True se o texto respeita integralmente um dos formatos oficiais."""
    if len(texto) != cfg.TAMANHO_PLACA:
        return False
    return any(_aderencia(texto, m) == cfg.TAMANHO_PLACA for m in mascaras)


def qualidade(texto):
    """
    Nota em [0, 1] de "cara de placa", usada para escolher entre as leituras
    das varias variantes de pre-processamento quando o motor nao fornece uma
    confianca comparavel.
    """
    if not texto:
        return 0.0
    texto = texto[:12]
    nota_tamanho = 1.0 - min(1.0, abs(len(texto) - cfg.TAMANHO_PLACA) / 4.0)
    trecho = recortar_sete(texto)
    nota_formato = max(_aderencia(trecho, m) for m in MASCARAS) / float(cfg.TAMANHO_PLACA)
    return 0.45 * nota_tamanho + 0.55 * nota_formato


def processar(texto_cru, mascaras=MASCARAS):
    """Devolve (texto_limpo, texto_corrigido) a partir da saida crua do OCR."""
    limpo = limpar(texto_cru)
    return limpo, aplicar_mascara(limpo, mascaras)


# --------------------------------------------------------------------------- #
# Escolha da melhor leitura entre as variantes de pre-processamento
# --------------------------------------------------------------------------- #
PESO_CONFIANCA = 0.40
PESO_FORMATO = 0.45
PESO_CONSENSO = 0.15


def escolher_melhor(tentativas):
    """
    Escolhe, entre as leituras das varias variantes de pre-processamento, a
    mais provavel -- e devolve `(melhor, tentativas_com_nota)`.

    A nota combina tres sinais:

      * CONFIANCA do proprio motor;
      * FORMATO -- o quanto a leitura se parece com uma placa brasileira
        (`qualidade`), o que evita escolher uma leitura curta so porque o
        motor estava confiante nos poucos caracteres que enxergou;
      * CONSENSO -- a fracao de variantes que chegou AO MESMO resultado. E uma
        votacao simples: uma leitura que se repete em varios pre-processamentos
        diferentes e mais confiavel que uma leitura isolada com confianca alta.

    A MESMA regra e usada pelos dois motores -- caso contrario a comparacao
    entre Tesseract e EasyOCR estaria medindo tambem a regra de selecao.
    """
    if not tentativas:
        return None, []

    total = float(len(tentativas))
    apoio = {}
    for t in tentativas:
        chave = t.get("texto_corrigido", "")
        apoio[chave] = apoio.get(chave, 0) + 1

    for t in tentativas:
        consenso = apoio.get(t.get("texto_corrigido", ""), 0) / total
        t["consenso"] = round(consenso, 3)
        t["nota"] = round(
            PESO_CONFIANCA * float(t.get("confianca", 0.0))
            + PESO_FORMATO * qualidade(t.get("texto", ""))
            + PESO_CONSENSO * consenso,
            4,
        )

    melhor = max(tentativas, key=lambda t: t["nota"])
    return melhor, tentativas
