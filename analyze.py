from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score
P=Path(__file__).parent
raw=P/'dist/data/student-por.csv'
d=pd.read_csv(raw,sep=';')
assert d.shape==(649,33)
assert d.G3.between(0,20).all()
rng=np.random.default_rng(42)
labels={'studytime':{'1':'Under 2 hours','2':'2–5 hours','3':'5–10 hours','4':'Over 10 hours'},'internet':{'yes':'Internet at home','no':'No home internet'},'higher':{'yes':'Plans higher education','no':'No higher education plans'},'famsup':{'yes':'Family support','no':'No family support'},'schoolsup':{'yes':'Receives school support','no':'No school support'},'Medu':{'0':'None','1':'Primary','2':'Lower secondary','3':'Secondary','4':'Higher education'}}
groups={}
for col in labels:
 out=[]
 for v,g in d.groupby(col):
  a=g.G3.to_numpy();boots=rng.choice(a,(5000,len(a)),replace=True).mean(axis=1)
  out.append(dict(label=labels[col][str(v)],n=len(a),mean=float(a.mean()),low=float(np.quantile(boots,.025)),high=float(np.quantile(boots,.975))))
 groups[col]=out
# Prespecified association model; study time enters categorically.
X=pd.get_dummies(d[['studytime','internet','Medu','Fedu','age','sex','school','failures','absences','higher']].astype({'studytime':str}),drop_first=True,dtype=float)
m=sm.OLS(d.G3,sm.add_constant(X.astype(float))).fit(cov_type='HC3')
coefs=[dict(term=k,estimate=float(v),low=float(m.conf_int().loc[k,0]),high=float(m.conf_int().loc[k,1])) for k,v in m.params.items() if k!='const']
# Exclude earlier grades to avoid presenting late-stage grades as early prediction.
f=['studytime','internet','Medu','Fedu','age','sex','school','failures','higher']
num=['Medu','Fedu','age','failures'];cat=[x for x in f if x not in num]
pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
cv=KFold(5,shuffle=True,random_state=42)
models={}
for name,model in [('Mean baseline',DummyRegressor()),('Ridge regression',make_pipeline(pre,Ridge(alpha=10)))]:
 scores=-cross_val_score(model,d[f],d.G3,cv=cv,scoring='neg_mean_absolute_error')
 models[name]={'mae':float(scores.mean()),'fold_mae':scores.tolist()}
report={'n':len(d),'mean':float(d.G3.mean()),'median':float(d.G3.median()),'missing':int(d.isna().sum().sum()),'duplicate_rows':int(d.duplicated().sum()),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'groups':groups,'coefficients':coefs,'r_squared':float(m.rsquared),'models':models,'t_critical':{str(n):float(t.ppf(.975,n-1)) for n in range(2,650)},'records':d.to_dict(orient='records')}
(P/'dist/data/analysis.json').write_text(json.dumps(report))
print(json.dumps({k:report[k] for k in ['n','missing','duplicate_rows','models','r_squared']},indent=2))
