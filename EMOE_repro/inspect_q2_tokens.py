import pickle, numpy as np
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'E题数据'
for p in sorted((r/'附件3-模态缺失特征样本'/'对齐版本').glob('*.pkl')):
 with p.open('rb') as f: d=pickle.load(f)['test']
 t=d['text_bert'][0]; n=int(t[1].sum()); ids=t[0,:n]; a=np.any(d['audio'][0]!=0,axis=1)[:n]; v=np.any(d['vision'][0]!=0,axis=1)[:n]
 print(p.stem, 'n',n,'idzeros',np.flatnonzero(ids==0).tolist(),'audiozeros',np.flatnonzero(~a).tolist(),'visionzeros',np.flatnonzero(~v).tolist())
