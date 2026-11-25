# OCR com ROI x OCR direto na imagem inteira

Gerado por `python run_experimentos.py` em 25/11/2026 09:21.

Rodar OCR na foto inteira do veiculo tende a dar resultado ruim, e recortar a ROI da placa tende a melhorar muito a leitura. Este relatorio mede esse ganho no dataset em vez de apenas afirma-lo.

O baseline sem ROI nao foi sabotado: ele usa os `--psm` apropriados a uma cena (3, 6 e 11, nao os 7/8/13 de linha unica), os limiares padrao do EasyOCR, e cada bloco de texto encontrado na foto conta como candidato, escolhido pela mesma regra do pipeline com ROI.

## Tesseract

| Metrica | Com ROI | Sem ROI | Ganho |
|---|---:|---:|---:|
| Acerto exato (com mascara) | 65.0% | 10.0% | +55.0 pp |
| Acerto exato (sem mascara) | 35.0% | 5.0% | +30.0 pp |
| Acuracia por caractere | 90.0% | 32.1% | +57.9 pp |
| Tempo por imagem | 14.49s | 2.93s | - |

Sem ROI o OCR nao devolveu nada em 0 das 20 imagens.

## EasyOCR

| Metrica | Com ROI | Sem ROI | Ganho |
|---|---:|---:|---:|
| Acerto exato (com mascara) | 60.0% | 5.0% | +55.0 pp |
| Acerto exato (sem mascara) | 35.0% | 0.0% | +35.0 pp |
| Acuracia por caractere | 92.9% | 50.0% | +42.9 pp |
| Tempo por imagem | 46.31s | 7.32s | - |

Sem ROI o OCR nao devolveu nada em 0 das 20 imagens.
