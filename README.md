# Projeto didático: detecção de falhas e soft sensor

Projeto completo com dados sintéticos de 30 dias, frequência de 5 minutos, nove notebooks encadeados e pastas de resultados.

## Ordem de execução

1. `01_integracao_sincronizacao.ipynb`
2. `02_analise_exploratoria_dados_brutos.ipynb`
3. `03_qualidade_tratamento_dados.ipynb`
4. `04_preparacao_soft_sensor.ipynb`
5. `05_treinamento_soft_sensor.ipynb`
6. `06_preparacao_classificacao_falhas.ipynb`
7. `07_treinamento_classificador_falhas.ipynb`
8. `08_integracao_avaliacao_final.ipynb`
9. `09_resumo_apresentacao.ipynb`

## Configuração no VS Code

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Instale as extensões **Python** e **Jupyter** da Microsoft, abra esta pasta no VS Code, selecione o kernel `.venv` e execute os notebooks em ordem.

## Observação

Os dados principais são sintéticos e incluem eventos marcados para fins didáticos. A coluna `evento_referencia` é a verdade de geração e não deve ser usada como entrada dos modelos.

Também foram incluídos arquivos reais para validação:
- `dados/01_brutos/leituras_202610221721.csv`: leituras reais do equipamento da estufa;
- `dados/01_brutos/leitura_climatica_202610221722.csv`: registros meteorológicos da estação Open-Meteo, integrados ao fluxo de sincronização.
