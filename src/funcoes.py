
from pathlib import Path
import numpy as np, pandas as pd

def raiz_projeto(inicio=None):
    p=Path(inicio or Path.cwd()).resolve()
    for candidato in [p,*p.parents]:
        if (candidato/'dados').exists() and (candidato/'notebooks').exists(): return candidato
    raise FileNotFoundError('Execute dentro da pasta projeto_soft_sensor.')

def carregar_csv(caminho):
    df=pd.read_csv(caminho,encoding='utf-8-sig')
    df['timestamp']=pd.to_datetime(df['timestamp'],utc=True).dt.tz_convert('America/Sao_Paulo')
    return df.sort_values('timestamp').reset_index(drop=True)

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
