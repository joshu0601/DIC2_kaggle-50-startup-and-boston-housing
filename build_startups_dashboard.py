import json, itertools, pathlib, copy
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.stats import f as f_dist
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
ROOT=pathlib.Path(__file__).resolve().parent
df=pd.read_csv(ROOT/'50_Startups.csv')
X=pd.get_dummies(df.drop(columns='Profit'),columns=['State'],drop_first=True,dtype=float)
y=df.Profit.astype(float)
cols=list(X.columns)
traini,testi=train_test_split(np.arange(len(df)),test_size=.2,random_state=0)
XT,XE=X.iloc[traini],X.iloc[testi];YT,YE=y.iloc[traini],y.iloc[testi]
cv=KFold(5,shuffle=True,random_state=42)
models={'OLS':LinearRegression(),'Ridge':Ridge(alpha=1.0),'Lasso':Lasso(alpha=100,max_iter=100000),'ElasticNet':ElasticNet(alpha=.1,l1_ratio=.5,max_iter=100000),'BayesianRidge':BayesianRidge()}
subsets={}; all_metrics={}; cv_cache={}
for k in range(1,6):
 for cc in itertools.combinations(cols,k):
  key=','.join(str(cols.index(c)) for c in cc);subsets[key]=list(cc)
  all_metrics[key]={}
  for name,est in models.items():
   pipe=make_pipeline(StandardScaler(),copy.deepcopy(est))
   folds=-cross_val_score(pipe,XT[list(cc)],YT,scoring='neg_mean_squared_error',cv=cv)
   cv_rmse=float(np.sqrt(np.mean(folds)))
   pipe.fit(XT[list(cc)],YT)
   pred=pipe.predict(XE[list(cc)]);trpred=pipe.predict(XT[list(cc)])
   n=len(YT);p=len(cc)
   trainr2=r2_score(YT,trpred)
   fstat=float((trainr2/p)/((1-trainr2)/(n-p-1))) if 1-trainr2>0 else 0
   prob=float(f_dist.sf(fstat,p,n-p-1))
   te_r2=float(r2_score(YE,pred))
   coef=pipe[-1].coef_.tolist()
   raw_coef=(pipe[-1].coef_/pipe[0].scale_)
   raw_intercept=float(pipe[-1].intercept_-np.dot(raw_coef,pipe[0].mean_))
   all_metrics[key][name]={'intercept':raw_intercept,'raw_coefs':raw_coef.tolist(),'cv':cv_rmse,'rmse':float(np.sqrt(mean_squared_error(YE,pred))),'r2':te_r2,'mae':float(mean_absolute_error(YE,pred)),'adjusted':float(1-(1-te_r2)*(len(YE)-1)/(len(YE)-p-1)) if len(YE)>p+1 else None,'fstat':fstat,'pval':prob,'pred':np.round(pred,3).tolist(),'coefs':np.round(coef,5).tolist(),'trainr2':float(trainr2)}
# OLS statistical diagnostics computed on training observations for each subset.
# Inference belongs to unregularized OLS; penalized estimators retain predictive coefficients
# while their descriptive significance table is explicitly labelled an OLS reference fit.
for key, names in subsets.items():
    Xstat=sm.add_constant(XT[names].astype(float),has_constant='add')
    fitted=sm.OLS(YT,Xstat).fit()
    diag=[]
    for j,feat in enumerate(['Intercept']+names):
        ci=fitted.conf_int().iloc[j]
        vif=None if j==0 else float(variance_inflation_factor(Xstat.values,j))
        diag.append({'feature':feat,'coef':float(fitted.params.iloc[j]),'stderr':float(fitted.bse.iloc[j]),'tstat':float(fitted.tvalues.iloc[j]),'pvalue':float(fitted.pvalues.iloc[j]),'ci_lower':float(ci.iloc[0]),'ci_upper':float(ci.iloc[1]),'vif':vif})
    for name in models:
        all_metrics[key][name]['ols_statistics']=diag
print('computed',len(all_metrics),'subsets')
def keyfor(names):return ','.join(map(str,sorted(cols.index(n) for n in names)))
def get_score(names,mode):return all_metrics[keyfor(names)]['OLS']['cv' if mode=='cv' else 'rmse']
methods={}
for mode in ['cv','test']:
 chosen=[];forward=[]
 for k in range(1,6):
  nextcol=min([c for c in cols if c not in chosen],key=lambda c:get_score(chosen+[c],mode))
  chosen.append(nextcol);forward.append(keyfor(chosen))
 exhaustive=[]
 for k in range(1,6):
  keys=[key for key,features in subsets.items() if len(features)==k]
  exhaustive.append(min(keys,key=lambda key:all_metrics[key]['OLS']['cv' if mode=='cv' else 'rmse']))
 methods[mode]={'Forward SFS':forward,'Exhaustive Search':exhaustive}
scaler=StandardScaler().fit(XT);sx=scaler.transform(XT)
lasso=Lasso(alpha=100,max_iter=100000).fit(sx,YT)
elastic=ElasticNet(alpha=.1,l1_ratio=.5,max_iter=100000).fit(sx,YT)
rank_lasso=sorted(range(5),key=lambda i:-abs(lasso.coef_[i]));rank_elastic=sorted(range(5),key=lambda i:-abs(elastic.coef_[i]))
rfe=RFE(LinearRegression(),n_features_to_select=1).fit(sx,YT)
rank_rfe=sorted(range(5),key=lambda i:rfe.ranking_[i])
for mode in methods:
 for name,rank in [('Lasso (L1 Ranking)',rank_lasso),('ElasticNet (L1 + L2)',rank_elastic),('RFE (Backward)',rank_rfe)]:
  methods[mode][name]=[keyfor([cols[i] for i in rank[:k]]) for k in range(1,6)]
print('methods',methods)
blob={'features':cols,'methods':methods,'metrics':all_metrics,'subsets':subsets,'test_rows':df.iloc[testi].round(3).to_dict(orient='records'),'test_indices':(testi+1).tolist(),'train_count':len(XT),'test_count':len(XE),'rows':df.round(3).to_dict(orient='records'),'source':'https://github.com/harimittapalli/Mulitple-Linear-Reggression/raw/master/50_Startups.csv'}
(ROOT/'startup_dashboard_data.json').write_text(json.dumps(blob,separators=(',',':'),ensure_ascii=False),encoding='utf8')

# Build the standalone HTML from the template and precomputed JSON.
template_path = ROOT / 'dashboard_template.html'
output_path = ROOT / '50_Startups_Interactive_Dashboard.html'
template = template_path.read_text(encoding='utf-8')
if template.count('__DATA_BLOB__') != 1:
    raise ValueError('Expected exactly one __DATA_BLOB__ placeholder in dashboard_template.html')
# Keep user-supplied JSON safely inside the application/json script element.
serialized = json.dumps(blob, separators=(',', ':'), ensure_ascii=False).replace('<', '\\u003c')
output_path.write_text(template.replace('__DATA_BLOB__', serialized), encoding='utf-8')
print(f'Generated {output_path} ({output_path.stat().st_size:,} bytes)')
