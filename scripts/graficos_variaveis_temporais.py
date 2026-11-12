from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARQUIVO = ROOT / "dados" / "01_brutos" / "leituras_202610221721.csv"
PASTA_SAIDA = ROOT / "resultados" / "figuras" / "series_temporais_estufa"
PDF_SAIDA = ROOT / "resultados" / "figuras" / "series_temporais_estufa.pdf"


def nome_arquivo(coluna):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", coluna).strip("_")


def main():
    if not ARQUIVO.exists():
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {ARQUIVO}")

    dados = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
    if "tempo" not in dados.columns:
        raise KeyError("O arquivo bruto precisa conter a coluna 'tempo'.")

    dados["tempo"] = pd.to_datetime(dados["tempo"], errors="coerce", utc=True)
    dados["tempo"] = dados["tempo"].dt.tz_convert("America/Sao_Paulo")
    dados = dados.dropna(subset=["tempo"]).sort_values("tempo")

    variaveis = [col for col in dados.columns if col not in ("tempo", "equipamento")]
    variaveis = [
        col for col in variaveis
        if pd.to_numeric(dados[col], errors="coerce").notna().any()
    ]
    if not variaveis:
        raise ValueError("Não foram encontradas variáveis numéricas para plotar.")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_SAIDA) as pdf:
        for coluna in variaveis:
            valores = pd.to_numeric(dados[coluna], errors="coerce")
            figura, eixo = plt.subplots(figsize=(11.7, 8.3))
            eixo.plot(dados["tempo"], valores, linewidth=0.7, color="#176B87")
            eixo.set_title(
                f"{coluna}\n"
                f"Leituras válidas: {valores.notna().sum():,} | "
                f"ausentes: {valores.isna().mean():.1%}"
            )
            eixo.set_xlabel("Data e hora")
            eixo.set_ylabel(coluna)
            eixo.grid(True, alpha=0.3)
            eixo.xaxis.set_major_locator(mdates.AutoDateLocator())
            eixo.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(eixo.xaxis.get_major_locator())
            )
            figura.autofmt_xdate()
            figura.tight_layout()
            figura.savefig(
                PASTA_SAIDA / f"{nome_arquivo(coluna)}.png",
                dpi=200,
                bbox_inches="tight",
            )
            pdf.savefig(figura, bbox_inches="tight")
            plt.close(figura)

    print(f"Variáveis plotadas: {len(variaveis)}")
    print(f"Figuras individuais: {PASTA_SAIDA}")
    print(f"PDF para impressão: {PDF_SAIDA}")


if __name__ == "__main__":
    main()
