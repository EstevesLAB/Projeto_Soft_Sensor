from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARQUIVO = ROOT / "dados" / "01_brutos" / "leitura_climatica_202610221722.csv"
PASTA_SAIDA = ROOT / "resultados" / "figuras" / "series_temporais_clima"

VARIAVEIS = {
    "te_c": ("Temperatura do ar", "°C"),
    "i_wm2": ("Radiação solar", "W/m²"),
    "vv_ms": ("Velocidade do vento", "m/s"),
    "ure_pct": ("Umidade relativa do ar", "%"),
    "tpo_c": ("Temperatura do ponto de orvalho", "°C"),
}


def main():
    if not ARQUIVO.exists():
        raise FileNotFoundError(f"Arquivo climático não encontrado: {ARQUIVO}")

    dados = pd.read_csv(ARQUIVO, encoding="utf-8-sig")
    colunas_obrigatorias = {"fonte", "ts", *VARIAVEIS}
    faltantes = colunas_obrigatorias.difference(dados.columns)
    if faltantes:
        raise KeyError(f"Colunas obrigatórias ausentes no arquivo: {sorted(faltantes)}")

    dados = dados.loc[dados["fonte"].eq("open-meteo")].copy()
    if dados.empty:
        raise ValueError("O arquivo não contém registros da fonte 'open-meteo'.")

    dados["ts"] = pd.to_datetime(dados["ts"], errors="coerce", utc=True)
    dados = dados.dropna(subset=["ts"]).sort_values("ts")
    if dados.empty:
        raise ValueError("A fonte Open-Meteo não contém timestamps válidos.")
    dados["ts"] = dados["ts"].dt.tz_convert("America/Sao_Paulo")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    for coluna, (rotulo, unidade) in VARIAVEIS.items():
        valores = pd.to_numeric(dados[coluna], errors="coerce")
        figura, eixo = plt.subplots(figsize=(11.7, 8.3))
        eixo.plot(dados["ts"], valores, linewidth=0.9, color="#176B87")
        eixo.set_title(
            f"{rotulo} ({coluna})\n"
            f"Leituras Open-Meteo: {valores.notna().sum():,} | "
            f"ausentes: {valores.isna().mean():.1%}"
        )
        eixo.set_xlabel("Data e hora")
        eixo.set_ylabel(unidade)
        eixo.grid(True, alpha=0.3)
        eixo.xaxis.set_major_locator(mdates.AutoDateLocator())
        eixo.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(eixo.xaxis.get_major_locator())
        )
        figura.autofmt_xdate()
        figura.tight_layout()
        figura.savefig(
            PASTA_SAIDA / f"{coluna}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figura)

    print(f"Registros Open-Meteo plotados: {len(dados):,}")
    print(f"Variáveis plotadas: {len(VARIAVEIS)}")
    print(f"Gráficos individuais: {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
