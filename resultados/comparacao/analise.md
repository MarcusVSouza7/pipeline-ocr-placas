# Analise comparativa: Tesseract OCR x EasyOCR

Gerado automaticamente por `python run_comparacao.py` em 25/11/2026 11:09.

Os dois pipelines compartilham **exatamente** as mesmas etapas de
deteccao, segmentacao, extracao e pre-processamento. A unica variavel
e o motor de OCR, de modo que as diferencas abaixo sao atribuiveis a ele.

## 1. Resultados quantitativos

| Metrica | Tesseract | EasyOCR |
|---|---:|---:|
| Acerto exato (pos-mascara) | 65.0% | 60.0% |
| Acerto exato (sem mascara) | 35.0% | 35.0% |
| Acuracia por caractere | 90.0% | 92.9% |
| Acuracia por caractere (sem mascara) | 83.6% | 85.7% |
| Tempo medio por imagem | 9.00s | 39.18s |
| Tempo por chamada ao motor | 0.500s | 6.531s |
| Chamadas ao motor (6 variantes x N modos) | 360 | 120 |

Placas segmentadas com sucesso: **20/20 (100.0%)** -- etapa comum aos dois.

## 2. Concordancia

- Leituras identicas: **10/20**
- Acertaram ambos: **10**
- So o Tesseract acertou: **3**
- So o EasyOCR acertou: **2**
- Nenhum acertou: **5**

Imagens em que as leituras divergem: 1.JPG, 3.JPG, 4.JPG, 6.JPG, 7.JPG, 8.JPG, 10.JPG, 13.JPG, 15.JPG, 20.JPG.

## 3. Erros caractere a caractere

| Confusao (lido -> correto) | Tesseract | EasyOCR |
|---|---:|---:|
| V -> Y | 0 | 2 |
| L -> I | 2 | 0 |
| 2 -> 7 | 1 | 1 |
| 0 -> 4 | 1 | 0 |
| 0 -> 9 | 1 | 0 |
| A -> U | 0 | 1 |
| I -> J | 1 | 0 |
| E -> G | 1 | 0 |
| 1 -> 2 | 1 | 0 |
| 2 -> 1 | 1 | 0 |

## 4. Sensibilidade ao pre-processamento

Cada imagem foi submetida a 6 variantes de pre-processamento; a tabela
mostra quantas vezes cada uma produziu a melhor leitura de cada motor.

| Variante | Tesseract | EasyOCR |
|---|---:|---:|
| adaptativo | 0 | 2 |
| cinza | 4 | 5 |
| nitidez | 0 | 1 |
| otsu | 7 | 3 |
| segmentada | 9 | 9 |

## 5. Leitura dos resultados

**Motor com melhor acerto exato neste dataset: Tesseract** (diferenca de 1 imagem).

Pontos observados:

- *Erros de um caractere so*: a acuracia por caractere dos dois motores fica
  bem acima do acerto exato (90% x 65% no Tesseract, 93% x 60% no
  EasyOCR). Ou seja: quase todo erro e de UM caractere -- mas basta um para
  invalidar a placa, e e isso que derruba o acerto exato.
- *Naturezas de erro diferentes*: o Tesseract confunde caracteres que diferem
  por um tracinho (L/I, G/Q, B/D), porque decide caractere a caractere. O
  EasyOCR confunde caracteres de forma global parecida (V/Y, U/A), porque
  reconhece a linha inteira de uma vez (CRNN+CTC) -- ver tabela 3.
- *Pre-processamento*: a variante vencedora nos DOIS motores e a `segmentada`
  (caracteres recortados pela segmentacao, binarizados um a um e recolados).
  Isto e, boa parte do desempenho vem da segmentacao, nao do motor. A
  diferenca aparece na segunda colocada: o Tesseract prefere binario (`otsu`)
  e o EasyOCR prefere a imagem fotografica (`cinza`) -- coerente com um motor
  classico contra um motor neural treinado em fotos naturais.
- *Pos-processamento*: a mascara de formato (AAA0A00) soma +30 pontos ao
  Tesseract e +25 pontos ao EasyOCR, porque quase toda confusao de OCR em
  placas (O/0, I/1, S/5, G/0, Z/2) e resolvida pela POSICAO do caractere.
  Conhecimento de dominio pesa aqui tanto quanto a escolha do motor.
- *Custo*: o Tesseract e ~13x mais rapido por chamada (0.50s contra 6.53s em
  CPU). O EasyOCR carrega dois modelos neurais; com GPU a distancia cai, mas
  para rodar sem aceleracao a diferenca e decisiva.

## 6. Limitacoes

- O gabarito (`data/ground_truth.csv`) foi transcrito manualmente das 20
  imagens; ele considera a placa principal de cada foto.
- As imagens 2, 16 e 19 contem uma segunda placa/adesivo. A deteccao as
  encontra e o `.txt` de saida as lista como deteccao secundaria, mas elas
  nao entram nas metricas.
- Com 20 imagens, cada acerto vale 5 pontos percentuais. Uma diferenca de
  1 imagem entre os motores esta dentro do ruido amostral: o resultado
  deve ser lido como tendencia (e, neste caso, como empate tecnico com
  perfis de erro e de custo diferentes), nao como medida precisa.
