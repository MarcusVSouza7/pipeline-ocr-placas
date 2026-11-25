# Segmentacao e OCR de Placas Veiculares — Tesseract x EasyOCR

Pipeline completo de visao computacional que **detecta**, **segmenta** e **extrai** placas
veiculares brasileiras (padrao Mercosul) a partir de fotos de veiculos e, em seguida,
aplica **OCR** com duas ferramentas diferentes — **Tesseract** e **EasyOCR** — para
comparar o desempenho das duas abordagens sobre exatamente o mesmo pipeline de
processamento.

---

## 1. Visao geral do pipeline

```
   imagem original
         |
   [1] DETECCAO           tres detectores de candidatos (cor / bordas / brilho)
         |                + pontuacao heuristica + supressao de nao-maximos
   [2] SEGMENTACAO        retificacao por rotacao, recorte da placa e da tarja,
         |                segmentacao dos caracteres (componentes conexos)
   [3] PRE-PROCESSAMENTO  6 variantes (cinza, clahe, nitidez, otsu,
         |                adaptativo, segmentada)
   [4] OCR                Tesseract  |  EasyOCR     <-- UNICA etapa que muda
         |
   [5] POS-PROCESSAMENTO  limpeza + mascara de formato AAA0A00
         |
   [6] RELATORIOS         .txt, .csv, .json, analise comparativa
```

As etapas 1, 2, 3 e 5 sao **identicas** nos dois pipelines. So a etapa 4 muda — e isso que
torna a comparacao entre Tesseract e EasyOCR valida: qualquer diferenca de resultado vem
do motor de OCR, nao do tratamento da imagem.

### [1] Deteccao — tres detectores complementares

Nenhuma rede neural de deteccao e usada; tudo e visao computacional classica.

| Detector | Tecnica | Quando e o melhor |
|---|---|---|
| `faixa_azul` | Segmentacao de cor em **HSV** + fechamento morfologico horizontal | Caso geral: toda placa Mercosul tem o cabecalho azul "BRASIL" no topo |
| `bordas` | **Sobel em X** + limiar de Otsu + fechamento morfologico | Placas com a faixa azul em sombra ou estourada |
| `regiao_clara` | Mascara de baixa saturacao + alto valor (corpo branco da placa) | Placas pequenas, em que a faixa azul tem poucos pixels |

Cada candidato recebe uma **pontuacao heuristica** em `[0,1]` combinando seis evidencias:
numero de caracteres segmentados (peso maior — a placa brasileira tem exatamente 7),
presenca de azul no cabecalho, corpo claro e pouco saturado, contraste interno, area
relativa e razao largura/altura da tarja. Os candidatos sobrepostos sao eliminados por
**supressao de nao-maximos** (IoU > 0.3).

### [2] Segmentacao e extracao

- **Retificacao**: a aresta superior da faixa azul da a inclinacao exata da placa. A
  imagem e rotacionada para deixa-la horizontal — o que corrige as placas tortas das
  imagens 2, 5, 19 e 20 antes de qualquer recorte.
- **Recorte da tarja**: so a regiao alfanumerica e extraida; o cabecalho azul
  ("MERCOSUL/BRASIL" + bandeira) e a sigla "BR" sao descartados, pois sao ruido puro
  para o OCR.
- **Segmentacao de caracteres**: analise de **componentes conexos** sobre 8 binarizacoes
  diferentes da janela de busca. A linha de caracteres e encontrada por **votacao** —
  vence o maior conjunto de componentes mutuamente coerentes (mesma altura, mesma linha
  de base), o que descarta automaticamente o texto pequeno da faixa e a sigla "BR".
  A geometria final da placa e reconstruida a partir dos proprios caracteres, que sao
  uma referencia muito mais estavel que a mascara de cor.

> O ponto mais delicado foi a **altura da faixa azul**: gradientes e reflexos esticam a
> mascara de cor e a estimativa fica ruidosa. A solucao foi nao depender dela — a janela
> de busca e testada em tres tamanhos e duas folgas laterais, e a melhor linha de
> caracteres e escolhida entre todas as combinacoes.

### [3] Pre-processamento — 6 variantes por imagem

| Variante | Tecnica |
|---|---|
| `cinza` | Tons de cinza + reamostragem para altura fixa + **filtro bilateral** (remove o ruido JPEG e a textura de seguranca preservando as bordas) |
| `clahe` | **Equalizacao de histograma local** — recupera placas lavadas (cinza-claro sobre prata) |
| `nitidez` | **Unsharp masking** — compensa desfoque de foco/movimento |
| `otsu` | Binarizacao global de Otsu + abertura morfologica |
| `adaptativo` | Limiar adaptativo gaussiano — robusto a sombra parcial e reflexo |
| `segmentada` | Cada caractere e recortado, binarizado **individualmente** (Otsu numa caixa quase bimodal) e recolado numa tela branca com espacamento regular |

Cada motor roda em **todas** as variantes e fica com a leitura mais confiavel
(confianca do proprio motor combinada com a aderencia ao formato de placa). O relatorio
registra qual variante venceu em cada imagem — e assim se mede, na pratica, a
sensibilidade de cada OCR ao pre-processamento.

### [4] OCR

- **Tesseract** (`pytesseract`): `--oem 3`, whitelist `A-Z0-9`, e tres modos de
  segmentacao de pagina (`--psm 7`, `8` e `13`), porque modos diferentes acertam imagens
  diferentes. Confianca obtida via `image_to_data`.
- **EasyOCR**: detector CRAFT + reconhecedor CRNN/CTC, com `allowlist` equivalente e
  limiares de agrupamento afrouxados para que a placa nao seja quebrada em varios blocos.

### [5] Pos-processamento

1. **Limpeza**: maiusculas, apenas `A-Z0-9`, remocao dos rotulos impressos na propria
   placa (`BRASIL`, `MERCOSUL`, `BR`).
2. **Recorte para 7 caracteres**: janela deslizante que escolhe a sequencia mais
   compativel com o formato oficial.
3. **Mascara de formato** `AAA0A00`: as posicoes 1, 2, 3 e 5 sao **letras** e as
   posicoes 4, 6 e 7 sao **digitos**. Isso resolve sozinho as confusoes classicas do OCR
   (`O/0`, `I/1`, `S/5`, `Z/2`, `G/0`, `B/8`), porque a posicao determina o tipo.

Os relatorios trazem a leitura **com** e **sem** a mascara, para separar o que e merito
do OCR e o que e merito do conhecimento de dominio.

---

## 2. Estrutura do projeto

```
.
├── README.md                      este arquivo
├── requirements.txt               dependencias (versao do OpenCV fixada)
├── dataset/placas_exercicio/      20 imagens de veiculos (dataset da atividade)
├── data/ground_truth.csv          gabarito transcrito manualmente
├── haar_xml/                      cascatas Haar de placa (Aula 8, opcional)
├── src/
│   ├── config.py                  parametros e caminhos centralizados
│   ├── plate_detector.py          [1][2] deteccao, retificacao e segmentacao
│   ├── detector_haar.py           [1] deteccao alternativa com Haar Cascade
│   ├── preprocessing.py           [3] as 6 variantes de pre-processamento
│   ├── ocr_tesseract.py           [4a] OCR com Tesseract
│   ├── ocr_easyocr.py             [4b] OCR com EasyOCR
│   ├── baseline_sem_roi.py        [4c] OCR direto na imagem inteira (linha de base)
│   ├── postprocess.py             [5] limpeza + mascara de formato
│   ├── evaluate.py                metricas (acerto exato, Levenshtein)
│   ├── pipeline.py                orquestracao e geracao de relatorios
│   ├── comparar.py                comparacao Tesseract x EasyOCR
│   └── comparar_experimentos.py   comparacoes de detector e de ROI
├── run_tesseract.py               executa o pipeline com Tesseract
├── run_easyocr.py                 executa o pipeline com EasyOCR
├── run_comparacao.py              gera a comparacao final
├── run_experimentos.py            gera as comparacoes de detector e de ROI
└── resultados/
    ├── tesseract/placas_tesseract.txt      <- entregavel
    ├── tesseract/por_imagem/*.txt
    ├── easyocr/placas_easyocr.txt          <- entregavel
    ├── easyocr/por_imagem/*.txt
    ├── comparacao/comparacao.txt | .csv | analise.md
    ├── crops/                     placas e tarjas extraidas
    ├── debug/                     deteccoes anotadas e variantes de pre-processamento
    └── experimentos/              [opcional] saidas de run_experimentos.py
```

Os modulos `detector_haar.py`, `baseline_sem_roi.py`, `comparar_experimentos.py`
e `run_experimentos.py` sao os experimentos opcionais descritos na secao 3; eles
gravam em `resultados/experimentos/` justamente para nunca sobrescrever os dois
entregaveis principais.

---

## 3. Como executar

### Dependencias

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

O Tesseract precisa tambem do **binario** do motor (nao vem pelo pip):

```bash
winget install UB-Mannheim.TesseractOCR     # Windows
sudo apt install tesseract-ocr              # Ubuntu
brew install tesseract                      # macOS
```

O EasyOCR baixa os modelos (~100 MB) automaticamente na primeira execucao.

### Execucao

```bash
python run_tesseract.py     # pipeline com Tesseract
python run_easyocr.py       # pipeline com EasyOCR
python run_comparacao.py    # tabela e analise comparativa
```

Esses tres comandos produzem tudo o que a atividade pede.

Opcoes uteis: `--dataset CAMINHO` (outra pasta de imagens), `--sem-artefatos`
(nao gravar recortes/debug) e, no EasyOCR, `--gpu`.

> **A versao do OpenCV e fixada (`==5.0.0.93`) de proposito.** Os resultados
> variam entre builds: a mesma pipeline da 65% de acerto exato com Tesseract no
> 5.0.0 e 55% no 4.14, porque `resize`/`bilateralFilter`/`CLAHE` mudaram o
> suficiente para virar a leitura de um caractere em duas imagens. Sem fixar a
> versao os numeros da secao 4 nao sao reproduziveis.

### Experimentos opcionais

Dois experimentos acompanham o projeto. Eles **nao fazem parte dos requisitos
da atividade** -- existem para quantificar duas decisoes do pipeline em vez de
apenas afirma-las:

```bash
python run_tesseract.py --sem-roi         # OCR direto na imagem inteira
python run_experimentos.py --so-roi       # relatorio ROI x sem ROI
python run_tesseract.py --detector haar   # deteccao por Haar Cascade
python run_experimentos.py --so-haar      # relatorio Haar x detector classico
```

O experimento do **Haar Cascade exige OpenCV 4.x** (`cv2.CascadeClassifier` foi
removido no 5.0):

```bash
pip install opencv-python==4.14.0.94 opencv-python-headless==4.14.0.94
```

Rodar nessa versao muda tambem os numeros do pipeline principal, por isso ela
nao e a versao padrao do projeto -- e por isso tambem que o relatorio do Haar
**nao acompanha o repositorio**: os resultados versionados em `resultados/` sao
todos da versao fixada. O experimento de ROI, que roda na versao fixada, tem o
relatorio publicado em `resultados/comparacao/roi_x_sem_roi.txt` e `.md`.

---

## 4. Resultados

Os numeros completos ficam em `resultados/comparacao/comparacao.txt` e a analise escrita
em `resultados/comparacao/analise.md`. Resumo sobre as 20 imagens do dataset:

### Etapa comum aos dois pipelines

| Metrica | Resultado |
|---|---|
| Placas detectadas e segmentadas | **20/20 (100%)** |
| Placas com os 7 caracteres segmentados | **20/20** |

### Evidencia visual da segmentacao

As metricas dizem que o numero fechou; as imagens mostram que o pipeline enxergou a
placa. Para qualquer imagem `N` do dataset:

| Arquivo | O que mostra |
|---|---|
| `resultados/debug/N_deteccao.jpg` | a placa localizada na foto original, com o detector que a achou, a pontuacao e o numero de caracteres segmentados |
| `resultados/crops/N_placa.png` | a placa retificada e recortada |
| `resultados/crops/N_tarja.png` | so a faixa alfanumerica — e **esta** imagem que vai ao OCR |
| `resultados/debug/N_preproc.png` | mosaico com as 6 variantes de pre-processamento |

Nos 20 `*_deteccao.jpg` o retangulo verde esta sobre a placa principal e o rotulo
indica `7 chars`. Os casos que exercitam cada parte do pipeline:

| Imagens | O que demonstram |
|---|---|
| 2, 5, 19, 20 | placas tortas — a retificacao por rotacao deixa a tarja horizontal |
| 8 | caracteres cinza-claro sobre fundo prateado — variante `clahe` |
| 10, 11 | imagens desfocadas — variante `nitidez` |
| 11, 17 | placas pequenas — detector `regiao_clara` |
| 16 | o 5o caractere e a **letra O**, nao o digito zero |
| 2, 16, 19 | segunda placa ou adesivo — reportados como "2a placa detectada" |

### OCR

| Metrica | Tesseract 5.4.0 | EasyOCR 1.7.2 |
|---|---:|---:|
| **Acerto exato** (placa inteira) | **65,0%** (13/20) | 60,0% (12/20) |
| **Acuracia por caractere** | 90,0% | **92,9%** |
| Acerto exato *sem* a mascara de formato | 35,0% | 35,0% |
| Acuracia por caractere *sem* a mascara | 83,6% | 85,7% |
| Tempo por chamada ao motor | **0,81 s** | 7,72 s |
| Tempo por imagem (todas as variantes) | **14,5 s** | 46,3 s |

### Concordancia entre os motores

| | Imagens |
|---|---:|
| Leituras identicas | 10/20 |
| Acertaram os dois | 10 |
| So o Tesseract acertou | 3 (imagens 6, 8, 15) |
| So o EasyOCR acertou | 2 (imagens 7, 10) |
| Nenhum dos dois acertou | 5 (imagens 1, 3, 4, 13, 20) |

### O que os numeros mostram

1. **Os dois motores empatam em "quase certo" e divergem em "certo".** A acuracia por
   caractere e praticamente a mesma (90% x 93%) — os erros sao quase sempre de **um
   unico caractere**. Mas como basta um caractere errado para invalidar a placa, essa
   diferenca pequena se traduz em apenas 60–65% de acerto exato.

2. **Os erros sao de naturezas diferentes.** O Tesseract erra em pares que diferem por
   um tracinho (`L -> I` 2x, `G -> Q`, `B -> D`): ele decide caractere a caractere e um
   pixel a mais ou a menos muda a decisao. O EasyOCR erra em pares que diferem pela
   *forma global* (`V -> Y` 2x, `U -> A`): ele reconhece a linha inteira de uma vez
   (CRNN+CTC) e, quando erra, erra de forma mais "plausivel" contextualmente.

3. **O pos-processamento quase dobra o acerto.** Sem a mascara de formato `AAA0A00`,
   ambos ficam em 35%; com ela, sobem para 60–65%. Isso porque praticamente toda
   confusao de OCR em placas (`O/0`, `I/1`, `S/5`, `G/0`, `Z/2`) e resolvida pela
   **posicao** do caractere. Conhecimento de dominio vale, aqui, tanto quanto o motor.

4. **A variante de pre-processamento que mais vence e a mesma para os dois: a
   `segmentada`** (9 de 20, nos dois casos) — os caracteres recortados pela segmentacao,
   binarizados individualmente e recolados numa tela branca. Ou seja: **a maior parte do
   ganho vem da segmentacao, nao do motor**. A diferenca aparece na segunda colocada: o
   Tesseract prefere `otsu` (7x, binario) e o EasyOCR prefere `cinza` (5x, fotografico),
   exatamente como se espera de um motor classico contra um motor neural.

5. **Custo**: o Tesseract e ~10x mais rapido por chamada (0,81 s contra 7,72 s em CPU).
   Para rodar em tempo real sem GPU, a diferenca e decisiva.

6. **A diferenca de uma imagem nao sobrevive a troca da versao do OpenCV.**
   Rodando a pipeline identica com `opencv-python==4.14` em vez de `5.0.0`, o
   Tesseract cai de 65,0% para 55,0%: as imagens 5 (`USQ0T32` -> `USA0T32`) e 9
   (`AOA4J16` -> `AQA4J16`) passam a errar um caractere. Nada no codigo mudou --
   apenas as implementacoes de `resize`, `bilateralFilter` e `CLAHE` entre os
   dois builds, o suficiente para mover alguns pixels e virar a decisao do
   classificador. E a evidencia mais direta de que, com 20 imagens, uma
   diferenca de uma ou duas placas e ruido: por isso a versao esta fixada no
   `requirements.txt`.

**Conclusao**: neste dataset o Tesseract leva por uma imagem no acerto exato, mas o
EasyOCR e mais preciso caractere a caractere e erra "menos feio" — e os dois so chegam a
esse patamar por causa da segmentacao e do pos-processamento, que sao a mesma coisa nos
dois pipelines. Com 20 imagens, cada acerto vale 5 pontos percentuais: a vantagem de uma
imagem do Tesseract esta dentro do ruido — o item 6 mostra que ela nem sobrevive a uma
troca de build do OpenCV — e a leitura honesta e de **empate tecnico**.

---

## 5. Atendimento aos requisitos da atividade

| Requisito do enunciado | Onde esta |
|---|---|
| Detectar e segmentar as placas das imagens do dataset | `src/plate_detector.py` -- 20/20 imagens |
| Extrair a regiao correspondente a placa | `resultados/crops/*_placa.png` (placa retificada) e `*_tarja.png` (so os caracteres) |
| Aplicar OCR com **Tesseract** | `src/ocr_tesseract.py` + `run_tesseract.py` |
| Aplicar OCR com **EasyOCR** | `src/ocr_easyocr.py` + `run_easyocr.py` |
| Gerar arquivos .txt com os resultados de cada abordagem | `resultados/tesseract/placas_tesseract.txt` e `resultados/easyocr/placas_easyocr.txt`, mais um `.txt` por imagem em `por_imagem/` |
| Comparar as duas abordagens sobre o mesmo pipeline | `src/comparar.py` -> `resultados/comparacao/` (etapas 1-3 e 5 identicas; so o motor muda) |
| Organizacao separando processamento, OCR e resultados | `src/` por etapa, `run_*.py` como pontos de entrada, `resultados/` por abordagem (secao 2) |
| Codigo comentado, sobretudo segmentacao e pre-processamento | docstrings de modulo em `plate_detector.py` e `preprocessing.py` explicando cada tecnica e o porque |
| Testar diferentes tecnicas de pre-processamento | 6 variantes por imagem (secao 1, etapa 3); o relatorio registra qual venceu em cada caso |

### Tecnicas das aulas usadas no projeto

Material de apoio, nao requisito -- mapeamento de onde cada tecnica vista em
aula aparece no codigo.

| Aula | Conteudo | Onde esta implementado |
|---|---|---|
| 3 | Fotografia digital, compressao, artefatos JPEG | `preprocessing.py` -- o filtro bilateral da variante `cinza` existe para remover ruido de compressao preservando as bordas dos caracteres |
| 4 | Espacos de cor, conversoes (RGB, HSV, YUV) | `plate_detector.py` -- `cvtColor` BGR->HSV no detector de faixa azul e BGR->GRAY em todo o pipeline |
| 5 | Filtros de ruido: media, mediana, gaussiano, bilateral | `preprocessing.py` (bilateral, gaussiano na mascara de nitidez), `plate_detector.py` (bilateral antes do Sobel) |
| 6 | Segmentacao por cor, `inRange`, mascaras, morfologia | `plate_detector.py` -- `_mascara_azul` / `_mascara_clara` com `inRange` em HSV + `morphologyEx` (CLOSE/OPEN) |
| 7 | Deteccao de movimento em video | nao se aplica: o dataset e de fotos estaticas. As operacoes morfologicas da aula sao usadas nas etapas 1-3 |
| 8 | **Haar Cascade**, imagem integral, `detectMultiScale`, ROI | `detector_haar.py` -- **verificacao extra, fora dos requisitos** (o enunciado nao exige tecnica especifica de deteccao). Usa as cascatas de placa distribuidas em `Exemplos (1).zip`. A medicao foi feita a parte, em OpenCV 4.14 -- a versao fixada do projeto (5.0) removeu `cv2.CascadeClassifier` --, por isso nao ha relatorio dela em `resultados/`: a cascata acha a placa em 19/20 imagens, contra 20/20 do detector classico, falhando na imagem 20 (placa sobreposta a uma antiga) |
| 8 | HOG+SVM, YOLO | nao usados -- o enunciado nao exige tecnica especifica de deteccao e o detector classico resolve 20/20 |
| 9 | Cinza, threshold simples/Otsu/adaptativo, blur, morfologia | `preprocessing.py` -- as variantes `cinza`, `otsu`, `adaptativo`, `clahe`, `nitidez`, `segmentada` |
| 9 | Tesseract, `--psm`, `image_to_data`, whitelist | `ocr_tesseract.py` -- `--oem 3`, `--psm 7/8/13`, `tessedit_char_whitelist`, confianca via `image_to_data` |
| 9 | EasyOCR | `ocr_easyocr.py` -- `easyocr.Reader` com `allowlist` equivalente |
| 9 | Pos-processamento com regex do padrao Mercosul | `postprocess.py` -- a mascara `AAA0A00` generaliza o `r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"` do slide, corrigindo o caractere em vez de so validar |
| 9 | **OCR direto x OCR com ROI** ("demonstracao importante") | `baseline_sem_roi.py` (experimento opcional) -- roda o OCR na foto inteira para medir o ganho da segmentacao em vez de apenas afirma-lo |

### Tecnicas usadas que vao alem das aulas

Estas nao aparecem nos slides e estao aqui porque o dataset exigiu; cada uma
resolve um problema concreto das 20 imagens:

- **Sobel em X** (`plate_detector._candidatos_bordas`) -- detector de borda
  direcional, para placas cuja faixa azul esta em sombra ou estourada.
- **CLAHE** (`preprocessing`) -- equalizacao de histograma *local*, para as
  placas lavadas (cinza-claro sobre prata) da imagem 8.
- **Unsharp masking** (`preprocessing`) -- compensa o desfoque das imagens 10 e 11.
- **Componentes conexos** (`plate_detector._componentes`) -- segmentacao de
  caracteres, que e o que permite a variante `segmentada` e a contagem de 7
  caracteres usada na pontuacao.
- **Supressao de nao-maximos** -- descarta deteccoes sobrepostas.
- **Distancia de Levenshtein** (`evaluate`) -- acuracia por caractere.

---

## 6. Observacoes sobre o dataset

- As 20 imagens originais (`placas_exercício/`) foram movidas para
  `dataset/placas_exercicio/` — sem acento — para evitar problemas de codificacao de
  caminho entre Windows, Linux e Git. O conteudo e exatamente o mesmo.
- As placas foram sobrepostas digitalmente as fotos e aparecem **esticadas na
  horizontal** em graus variaveis — por isso o pipeline nunca usa a razao de aspecto
  nominal da placa (400x130 mm) como criterio rigido; as proporcoes **verticais**, essas
  sim, se mantem.
- Imagem **16**: o 5o caractere e a **letra O** (visivelmente mais larga que o digito 0,
  que na fonte das placas e um oval estreito) — coerente com o formato Mercosul, em que
  a 5a posicao e sempre uma letra.
- Imagens **2**, **16** e **19** contem uma segunda placa ou adesivo de concessionaria.
  O detector as encontra e as reporta como *deteccao secundaria* nos `.txt`, mas o
  gabarito e as metricas consideram apenas a placa principal.
