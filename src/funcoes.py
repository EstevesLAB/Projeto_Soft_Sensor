
from pathlib import Path
import numpy as np, pandas as pd

def raiz_projeto(inicio=None):
    p=Path(inicio or Path.cwd()).resolve()
    for candidato in [p,*p.parents]:
        if (candidato/'dados').exists() and (candidato/'notebooks').exists(): return candidato
    raise FileNotFoundError('Execute dentro da pasta projeto_soft_sensor.')

def carregar_csv(caminho):
    df=pd.read_csv(caminho,encoding='utf-8-sig')
    col=None
    for nome in ['timestamp','tempo','datetime','data_hora','time']:
        if nome in df.columns:
            col=nome; break
    if col is None:
        raise KeyError(f'Arquivo sem coluna temporal conhecida: {list(df.columns[:10])}')
    ts=pd.to_datetime(df[col],errors='coerce',utc=True)
    if ts.dt.tz is None:
        ts=pd.to_datetime(df[col],errors='coerce').dt.tz_localize('UTC')
    df=df.copy()
    df['timestamp']=ts.dt.tz_convert('America/Sao_Paulo')
    return df.sort_values('timestamp').reset_index(drop=True)


def carregar_clima_open_meteo(caminho):
    df=pd.read_csv(caminho,encoding='utf-8-sig')
    if 'ts' not in df.columns:
        raise KeyError(f'Arquivo climático sem coluna ts: {list(df.columns[:10])}')
    df=df.copy()
    df=df[df['fonte'].fillna('').str.contains('open-meteo',case=False,na=False)].copy()
    df['timestamp']=pd.to_datetime(df['ts'], errors='coerce', utc=True).dt.tz_convert('America/Sao_Paulo')
    df=df.dropna(subset=['timestamp']).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    mapeamento={
        'te_c':'temperatura_ambiente_C',
        'i_wm2':'radiacao_solar_global_W_m2',
        'vv_ms':'velocidade_vento_m_s',
        'ure_pct':'umidade_relativa_ar_pct',
        'tpo_c':'temperatura_ponto_orvalho_C',
    }
    return df.rename(columns=mapeamento)[['timestamp','temperatura_ambiente_C','radiacao_solar_global_W_m2','velocidade_vento_m_s','umidade_relativa_ar_pct','temperatura_ponto_orvalho_C']]


def normalizar_csv_real(df):
    x=df.copy()
    if 'timestamp' not in x.columns:
        raise KeyError('O DataFrame real precisa conter a coluna timestamp.')
    x=x.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    mapeamento={
        'setor_127':'setor_127',
        'temperatura_128':'temperatura_interna_C',
        'umidade_relativa_129':'umidade_relativa_interna_pct',
        'tensiometria_130':'tensiometria_130',
        'ponto_orvalho_131':'ponto_orvalho_131',
        'dpv_inst_132':'dpv_inst_132',
        'radiacao_inst_133':'radiacao_inst_133',
        'radiacao_acum_134':'radiacao_acum_134',
        'rssi_d_169':'rssi_d_169',
        'rssi_u_170':'rssi_u_170',
        'umidade_relativa2_439':'umidade_relativa2_439',
        'ponto_orvalho2_440':'ponto_orvalho2_440',
        'dpv2_441':'dpv2_441',
        'radiacao1_442':'radiacao1_442',
        'radiacao2_443':'radiacao2_443',
        'co2_444':'co2_444',
        'setor_273':'setor_273',
        'radiacao_acum_274':'radiacao_acum_274',
        'radiacao_inst_275':'radiacao_inst_275',
        'chuva_hora_370':'chuva_hora_370',
        'chuva_dia_371':'chuva_dia_371',
        'chuva_semana_372':'chuva_semana_372',
    }
    x=x.rename(columns=mapeamento)
    return x


def montar_dados_sincronizados(caminho_sensor, caminho_clima, colunas_sensor=None, tolerancia='15min'):
    sensor_cols = colunas_sensor or [
        'temperatura_interna_C',
        'umidade_relativa_interna_pct',
        'ponto_orvalho_131',
        'radiacao_inst_133',
        'radiacao_acum_134',
        'rssi_d_169',
        'rssi_u_170',
        'umidade_relativa2_439',
        'ponto_orvalho2_440',
        'dpv2_441',
        'co2_444',
    ]
    clima_cols = [
        'temperatura_ambiente_C',
        'umidade_relativa_ar_pct',
        'radiacao_solar_global_W_m2',
        'temperatura_ponto_orvalho_C',
        'velocidade_vento_m_s',
    ]

    sens = normalizar_csv_real(carregar_csv(caminho_sensor)).sort_values('timestamp').reset_index(drop=True)
    clima = carregar_clima_open_meteo(caminho_clima).sort_values('timestamp').reset_index(drop=True)
    dados = pd.merge_asof(sens, clima, on='timestamp', direction='nearest', tolerance=pd.Timedelta(tolerancia))
    mask = dados[sensor_cols].notna().any(axis=1)
    dados = dados.loc[mask, sensor_cols + clima_cols + ['timestamp']].copy().reset_index(drop=True)

    dados.insert(0, 'tempo', dados['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z'))
    dados['tempo'] = dados['tempo'].str[:-2] + ':' + dados['tempo'].str[-2:]
    dados['status_sincronizacao'] = 'sensor+clima' if dados[clima_cols].notna().any().any() else 'sensor_sem_clima'
    return dados[['tempo'] + sensor_cols + clima_cols + ['timestamp', 'status_sincronizacao']]


def resumo_temporal(df):
    d=df['timestamp'].diff().dt.total_seconds().div(60)
    return {'linhas':len(df),'inicio':df.timestamp.min(),'fim':df.timestamp.max(),'duplicados':int(df.timestamp.duplicated().sum()),'frequencia_mediana_min':float(d.median()),'lacunas_maiores':int((d>d.median()*1.5).sum())}

def atributos_temporais(df):
    x=df.copy(); h=x.timestamp.dt.hour+x.timestamp.dt.minute/60
    x['hora_seno']=np.sin(2*np.pi*h/24); x['hora_cosseno']=np.cos(2*np.pi*h/24)
    ang=np.deg2rad(x['direcao_vento_graus']); x['vento_dir_seno']=np.sin(ang); x['vento_dir_cosseno']=np.cos(ang)
    return x

def metricas_regressao(y,p):
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    return {'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**0.5,'R2':r2_score(y,p),'vies_medio':float(np.mean(p-y))}

def extrair_janelas(df,col='temperatura_interna_C',janela=12,passo=3):
    rows=[]; vals=df[col].to_numpy(float); ext=df['temperatura_ambiente_C'].to_numpy(float); labels=df['classe_alvo'].to_numpy(int); ev=df['evento_referencia'].astype(str).to_numpy(); times=df.timestamp.to_numpy()
    for i in range(0,len(df)-janela+1,passo):
        j=i+janela; x=vals[i:j]; xe=ext[i:j]
        if np.isnan(x).any() or np.isnan(xe).any(): continue
        dif=np.diff(x); counts=np.bincount(labels[i:j],minlength=3); cls=int(np.argmax(counts))
        corr=np.corrcoef(x,xe)[0,1] if np.std(x)>0 and np.std(xe)>0 else 0
        rows.append({'timestamp_inicio':times[i],'timestamp_fim':times[j-1],'media':x.mean(),'mediana':np.median(x),'minimo':x.min(),'maximo':x.max(),'desvio_padrao':x.std(ddof=1),'variancia':x.var(ddof=1),'amplitude':x.max()-x.min(),'iqr':np.quantile(x,.75)-np.quantile(x,.25),'dif_media_abs':np.abs(dif).mean(),'dif_max_abs':np.abs(dif).max(),'var_diferencas':dif.var(ddof=1),'prop_dif_zero':np.mean(np.abs(dif)<.011),'valores_unicos':len(np.unique(np.round(x,2))),'inclinacao':np.polyfit(np.arange(janela),x,1)[0],'dif_media_externa':np.mean(x-xe),'correlacao_externa':corr,'classe':cls,'evento_dominante':pd.Series(ev[i:j]).mode().iloc[0]})
    return pd.DataFrame(rows)
