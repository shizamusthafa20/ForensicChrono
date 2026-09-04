import io, os, pickle
from datetime import datetime
import numpy as np, pandas as pd, streamlit as st
try:
 import shap, matplotlib.pyplot as plt
 SHAP_OK=True
except Exception: SHAP_OK=False
try:
 from reportlab.lib import colors
 from reportlab.lib.pagesizes import A4
 from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
 from reportlab.lib.enums import TA_CENTER
 from reportlab.lib.units import mm
 from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
 PDF_OK=True
except Exception: PDF_OK=False

st.set_page_config(page_title='ForensicChrono | PMI Intelligence',page_icon='🔬',layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:radial-gradient(circle at 90% 0%,rgba(41,112,255,.10),transparent 28%),radial-gradient(circle at 0% 20%,rgba(20,184,166,.08),transparent 25%),#07111f;color:#edf5ff}.block-container{max-width:1450px;padding-top:1.2rem;padding-bottom:4rem}[data-testid="stSidebar"]{background:#081525;border-right:1px solid rgba(255,255,255,.08)}[data-testid="stSidebar"] *{color:#eaf3ff}.hero,.section{border:1px solid rgba(255,255,255,.09);background:rgba(10,25,43,.88);border-radius:22px;padding:26px;margin:18px 0;box-shadow:0 18px 60px rgba(0,0,0,.18)}.hero{padding:36px;background:linear-gradient(135deg,rgba(13,31,53,.97),rgba(8,20,36,.94))}.brand{font-family:'Playfair Display',serif;font-size:44px}.brand span{color:#58d5c7}.tagline{color:#9db0c7;margin-top:8px}.badge{display:inline-block;margin-top:16px;padding:7px 12px;border-radius:999px;background:rgba(88,213,199,.1);border:1px solid rgba(88,213,199,.25);color:#7ee6dc;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase}.metric-card{border:1px solid rgba(255,255,255,.1);background:linear-gradient(145deg,#102842,#0a1c30);border-radius:18px;padding:18px;min-height:115px}.metric-label{color:#8fa5bd;font-size:11px;text-transform:uppercase;letter-spacing:1px}.metric-value{font-size:28px;font-weight:700;margin-top:7px}.metric-small{color:#86d8cf;font-size:12px;margin-top:4px}.result{border:1px solid rgba(88,213,199,.28);background:linear-gradient(135deg,rgba(16,59,68,.72),rgba(9,29,46,.95));border-radius:24px;padding:30px;text-align:center}.result-cap{color:#91b7bd;font-size:12px;text-transform:uppercase;letter-spacing:1.5px}.result-num{font-family:'Playfair Display',serif;font-size:52px;color:#7ee6dc;margin:7px}.note{border-left:4px solid #58d5c7;background:rgba(88,213,199,.07);padding:13px 16px;border-radius:8px;color:#c8d8e7;font-size:13px}.warn{border-left:4px solid #f3b562;background:rgba(243,181,98,.08);padding:13px 16px;border-radius:8px;color:#d9e4ee;font-size:13px}button[kind="primary"]{background:linear-gradient(135deg,#24a99d,#2375c9)!important;border:0!important}footer{visibility:hidden}
</style>''',unsafe_allow_html=True)

RNA_PATH='models/rna_model.pkl'; MICRO_PATH='models/microbial_model.pkl'; FUSION_PATH='models/fusion_model.pkl'
TISSUES=['Muscle - Skeletal','Lung','Skin - Sun Exposed (Lower leg)','Nerve - Tibial']

def load(path):
 if not os.path.exists(path): return None
 with open(path,'rb') as f:return pickle.load(f)
@st.cache_resource
def models(): return load(RNA_PATH),load(MICRO_PATH),load(FUSION_PATH)
rna_pkg,micro_pkg,fusion_pkg=models()
def pmi(h): return f'{h:.2f} hours' if h<24 else f'{h/24:.2f} days ({h:.1f} hours)'
def num(x,d=0.):
 try:return float(x) if np.isfinite(float(x)) else d
 except:return d

# ---------- RNA ----------
def rna_features(expr,meta,info):
 genes=list(info['selected_genes']); d=np.asarray(info['degrader_indices'],int); s=np.asarray(info['stable_indices'],int); meds=info.get('clinical_medians',{})
 for g in genes:
  if g not in expr: expr[g]=0.
 G=np.log2(expr[genes].astype(float).fillna(0).values+1)
 ratios=np.column_stack([G[:,i]-G[:,j] for i in d for j in s])
 summ=np.column_stack([ratios.mean(1),ratios.std(1),ratios.min(1),ratios.max(1),G[:,d].mean(1),G[:,s].mean(1)])
 m=meta.copy()
 for c in ['rin','autolysis','age','sex','hardy']:
  m[c]=pd.to_numeric(m.get(c,np.nan),errors='coerce').fillna(num(meds.get(c,0)))
 rin,aut,age,sex,hardy=[m[c].values for c in ['rin','autolysis','age','sex','hardy']]
 clinical=np.column_stack([rin,rin**2,aut,aut**2,age,sex,hardy,rin*aut,rin/(aut+1)])
 X=np.nan_to_num(np.hstack([G,ratios,summ,clinical]))
 names=genes+[f'ratio_{genes[i]}_vs_{genes[j]}' for i in d for j in s]+['ratio_mean','ratio_std','ratio_min','ratio_max','degrader_mean','stable_mean','rin','rin_squared','autolysis','autolysis_squared','age','sex','hardy','rin_x_autolysis','rin_over_autolysis_plus_1']
 return X,names

def predict_rna(pkg,tissue,expr,meta):
 t=pkg['base_models'][tissue]; X,names=rna_features(expr,meta,t['feature_info']); bp=[]
 for n in ['xgb_direct','xgb_log','rf_direct','rf_log','extra_direct','extra_log']:
  q=float(t['base_models'][n].predict(X)[0]); bp.append(np.expm1(q) if n.endswith('_log') else q)
 y=float(t['meta_model'].predict(np.asarray(bp).reshape(1,-1))[0]); y=np.clip(y,0,pkg.get('max_time_min',1200))/60
 return y,X,names,t['base_models']['xgb_direct']

# ---------- microbiome ----------
def rel(x):
 x=np.asarray(x,float); z=x.sum(1); z[z<=0]=1; return x/z[:,None]
def clr(x):
 pos=x[x>0]; pc=max(np.min(pos)*.5,1e-8) if len(pos) else 1e-8; q=np.log(x+pc); return q-q.mean(1,keepdims=True)
def logrel(x):return np.log1p(x*10000)
def div(x):
 e=1e-12; sh=-np.sum(x*np.log(x+e),1); si=1-np.sum(x*x,1); ri=np.sum(x>0,1); ev=sh/(np.log(ri+1)+e); dom=x.max(1); z=np.sort(x,1)[:,::-1]; return np.column_stack([sh,si,ri,ev,dom,z[:,0],z[:,:3].sum(1),z[:,:5].sum(1),z[:,:10].sum(1)])

def micro_features(pkg,otu,indoor,outdoor,env,site):
 taxa=np.asarray(pkg['selected_taxa'],int); fn=list(pkg['feature_names']);
 if otu.shape[1]<=taxa.max(): raise ValueError('Uploaded OTU table has fewer taxa columns than the saved model expects.')
 q=otu.iloc[:,taxa].values.astype(float); r=rel(q); a=clr(r); b=logrel(r); d=div(r)
 # Recover the exact centering statistics from the original training metadata when available.
 ref_path='data/raw/microbiome/metadata.tsv'
 if os.path.exists(ref_path):
  ref=pd.read_csv(ref_path,sep='\t',low_memory=False); ii=pd.to_numeric(ref.get('indoor_add',pd.Series(dtype=float)),errors='coerce').dropna(); oo=pd.to_numeric(ref.get('outdoor_add',pd.Series(dtype=float)),errors='coerce').dropna(); ic=float(ii.mean()) if len(ii) else 0.; oc=float(oo.mean()) if len(oo) else 0.
 else: ic=oc=0.
 iv=num(indoor); ov=num(outdoor); envvals={'indoor_ADD_centered':iv-ic,'indoor_ADD_squared':(iv-ic)**2,'outdoor_ADD_centered':ov-oc,'ADD_indoor_minus_outdoor':iv-ov,'ADD_indoor_outdoor_ratio':iv/(abs(ov)+1),'indoor_ADD_log':np.log1p(max(iv,0)),'outdoor_ADD_log':np.log1p(max(ov,0))}
 for n in fn:
  if n.startswith('environment_'): envvals[n]=1. if n[12:].lower()==str(env).lower() else 0.
  if n.startswith('body_site_'): envvals[n]=1. if n[10:].lower()==str(site).lower() else 0.
 clr_names=[n for n in fn if n.startswith('CLR_taxon_')]; log_names=[n for n in fn if n.startswith('LOG_taxon_')]; div_names=['shannon','simpson','richness','evenness','dominance','top1_abundance','top3_abundance','top5_abundance','top10_abundance']; env_names=[n for n in fn if n not in clr_names+log_names+div_names]
 X=np.hstack([a,b,d,np.array([envvals.get(n,0.) for n in env_names]).reshape(1,-1)]); X=np.nan_to_num(X)
 if X.shape[1]!=len(fn): raise ValueError(f'Microbiome feature mismatch: model expects {len(fn)}, generated {X.shape[1]}.')
 return X,fn

def predict_micro(pkg,otu,indoor,outdoor,env,site):
 X,fn=micro_features(pkg,otu,indoor,outdoor,env,site); vals={}; total=w=0
 for n,m in pkg['fitted_models'].items():
  q=float(np.asarray(m.predict(X)).ravel()[0]); q=np.expm1(q) if 'log' in n else q; vals[n]=q; ww=float(pkg['weights'].get(n,0)); total+=ww*q; w+=ww
 return max(0,total/w if w else np.mean(list(vals.values())))*24,X,fn,pkg['fitted_models'].get('xgb_direct')

def shap_one(model,X,names):
 if not SHAP_OK or model is None:return None,None
 try:
  v=np.asarray(shap.TreeExplainer(model).shap_values(X)); v=v[0] if v.ndim==2 else v; ix=np.argsort(np.abs(v))[::-1][:10]; tab=pd.DataFrame({'Feature':[names[i] for i in ix],'SHAP contribution':[float(v[i]) for i in ix]}); fig,ax=plt.subplots(figsize=(8,5)); z=tab.iloc[::-1]; ax.barh(z.Feature,z['SHAP contribution']); ax.axvline(0,linewidth=1); ax.set_xlabel('SHAP contribution'); ax.set_title('Case-specific SHAP explanation'); plt.tight_layout(); return tab,fig
 except Exception:return None,None

def calibrate_new_prediction(component, raw_prediction_hours):
 # Use the saved Ridge calibration model from fusion_model.pkl.
 calibrated = component['calibrator'].predict(
  np.asarray([[float(raw_prediction_hours)]], dtype=float)
 )[0]
 return float(max(0.0, calibrated))


def estimate_uncertainty(component, prediction_hours):
 # Use the saved KNN uncertainty model from fusion_model.pkl.
 local_mae = float(
  component['uncertainty_model'].predict(
   np.asarray([[float(prediction_hours)]], dtype=float)
  )[0]
 )

 # For approximately Gaussian residuals:
 # MAE ~= sigma * sqrt(2/pi)
 local_sigma = local_mae / np.sqrt(2.0 / np.pi)

 # Prevent unrealistically small uncertainty.
 local_sigma = max(
  local_sigma,
  float(component['global_sigma_hours']) * 0.50,
  1.0
 )
 return float(local_sigma)


def fuse(r, m):
 # Single-modality cases do not require fusion.
 if r is None:
  return m, 'Microbiome', 0.0, 1.0
 if m is None:
  return r, 'RNA', 1.0, 0.0

 if fusion_pkg is None:
  raise FileNotFoundError(
   "models/fusion_model.pkl is required for RNA + Microbiome fusion."
  )

 # Use the same saved calibration and uncertainty objects as the
 # original ForensicChrono deployment function.
 rna_component = {
  'calibrator': fusion_pkg['rna']['calibrator'],
  'uncertainty_model': fusion_pkg['rna']['uncertainty_model'],
  'global_sigma_hours': fusion_pkg['rna']['global_sigma_hours'],
 }
 microbiome_component = {
  'calibrator': fusion_pkg['microbiome']['calibrator'],
  'uncertainty_model': fusion_pkg['microbiome']['uncertainty_model'],
  'global_sigma_hours': fusion_pkg['microbiome']['global_sigma_hours'],
 }

 # Predictions supplied by the app are already in hours.
 rna_calibrated = calibrate_new_prediction(rna_component, r)
 microbiome_calibrated = calibrate_new_prediction(microbiome_component, m)

 # Case-specific uncertainty from the saved KNN models.
 rna_sigma = estimate_uncertainty(rna_component, rna_calibrated)
 microbiome_sigma = estimate_uncertainty(
  microbiome_component, microbiome_calibrated
 )

 # Precision = 1 / variance.
 rna_precision = 1.0 / (rna_sigma ** 2)
 microbiome_precision = 1.0 / (microbiome_sigma ** 2)
 total_precision = rna_precision + microbiome_precision

 # Match the trained ForensicChrono fusion rule.
 if rna_calibrated > 24.0:
  rw = 0.05
  mw = 0.95
 else:
  rw = rna_precision / total_precision
  mw = microbiome_precision / total_precision

 final = rw * rna_calibrated + mw * microbiome_calibrated
 primary = 'RNA' if rw > mw else 'Microbiome' if mw > rw else 'Balanced'

 # Keep detailed values available for verification.
 st.session_state['fusion_details'] = {
  'rna_raw_hours': float(r),
  'microbiome_raw_hours': float(m),
  'rna_calibrated_hours': float(rna_calibrated),
  'microbiome_calibrated_hours': float(microbiome_calibrated),
  'rna_uncertainty_hours': float(rna_sigma),
  'microbiome_uncertainty_hours': float(microbiome_sigma),
  'rna_weight': float(rw),
  'microbiome_weight': float(mw),
  'final_pmi_hours': float(final),
  'primary_evidence': primary,
 }

 return float(final), primary, float(rw), float(mw)

def pdf(case,tissue,evidence,r,m,final,primary,rw,mw,tables):
 if not PDF_OK:return None
 b=io.BytesIO(); doc=SimpleDocTemplate(b,pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=15*mm); ss=getSampleStyleSheet(); title=ParagraphStyle('T',parent=ss['Title'],fontName='Helvetica-Bold',fontSize=23,alignment=TA_CENTER,textColor=colors.HexColor('#12304a')); sub=ParagraphStyle('S',parent=ss['Normal'],fontSize=9,alignment=TA_CENTER,textColor=colors.HexColor('#66788a')); h=ParagraphStyle('H',parent=ss['Heading2'],fontSize=13,textColor=colors.HexColor('#12304a')); body=ParagraphStyle('B',parent=ss['BodyText'],fontSize=9,leading=13,textColor=colors.HexColor('#263b4d')); story=[Paragraph('FORENSICCHRONO',title),Paragraph('Multimodal Postmortem Interval Estimation Report',sub),Spacer(1,8)]
 story.append(Table([['CASE ID',case,'REPORT DATE',datetime.now().strftime('%d %b %Y')],['TISSUE',tissue,'EVIDENCE',evidence]],colWidths=[25*mm,70*mm,30*mm,55*mm],style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f3f7fa')),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d7e0e7')),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)])); story.append(Spacer(1,10)); story.append(Table([['ESTIMATED POSTMORTEM INTERVAL'],[pmi(final)],[f'Primary evidence source: {primary}']],colWidths=[180*mm],style=[('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eaf7f5')),('BOX',(0,0),(-1,-1),1,colors.HexColor('#67cfc3')),('ALIGN',(0,0),(-1,-1),'CENTER'),('FONTNAME',(0,1),(-1,1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),9),('FONTSIZE',(0,1),(-1,1),19),('FONTSIZE',(0,2),(-1,2),9),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)])); story.append(Paragraph('Evidence Summary',h)); rows=[['Modality','Estimate','Fusion weight']];
 if r is not None:rows.append(['RNA degradation',pmi(r),f'{rw*100:.1f}%'])
 if m is not None:rows.append(['Microbial succession',pmi(m),f'{mw*100:.1f}%'])
 story.append(Table(rows,colWidths=[55*mm,65*mm,60*mm],style=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12304a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d7e0e7')),('FONTSIZE',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f9fb')])]))
 story.append(Paragraph('Case-specific Explainability',h)); story.append(Paragraph('SHAP was applied to the tree-based XGBoost direct-PMI component. Contributions describe model influence for this case and are not causal claims.',body))
 for mod,tab in tables.items():
  if tab is not None:
   rows=[['Feature','SHAP contribution']]+[[str(x.Feature),f'{float(x["SHAP contribution"]):.4f}'] for _,x in tab.head(8).iterrows()]; story.append(Spacer(1,5)); story.append(Paragraph(mod,h)); story.append(Table(rows,colWidths=[125*mm,55*mm],style=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf5f7')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d7e0e7')),('FONTSIZE',(0,0),(-1,-1),7.5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(Paragraph('Interpretation & Caveats',h)); story.append(Paragraph('This system is a research decision-support tool and should not be used as a standalone determination of time of death. Results should be interpreted together with conventional forensic evidence.',body)); doc.build(story); return b.getvalue()

with st.sidebar:
 st.markdown('## 🔬 ForensicChrono')
 st.caption('PMI Intelligence Console')
 case=st.text_input('Case ID','FC-2026-001')
 choice=st.radio('Biological evidence',['RNA only','Microbiome only','RNA + Microbiome'],index=2)
 st.markdown('---')
 st.caption('Research prototype • Decision support only')

st.markdown(
 '<div class="hero"><div class="brand">Forensic<span>Chrono</span></div>'
 '<div class="tagline">Multimodal biological intelligence for postmortem interval estimation</div>'
 '<div class="badge">RNA degradation • microbial succession • reliability-weighted fusion</div></div>',
 unsafe_allow_html=True
)

# ----------------------------------------------------------------------
# EVIDENCE INTAKE
# ----------------------------------------------------------------------
r_hours=m_hours=None
r_tab=m_tab=None
r_fig=m_fig=None
tissue='Not applicable'

if choice in ['RNA only','RNA + Microbiome']:
 st.markdown(
  '<div class="section"><div class="section-title">🧬 RNA evidence intake</div>'
  '<div class="section-sub">Provide the RNA expression data and case metadata.</div></div>',
  unsafe_allow_html=True
 )
 tissue=st.selectbox('Tissue type',TISSUES,index=1)
 rf=st.file_uploader('RNA expression CSV / TSV',type=['csv','tsv','txt'],key='rf')
 x1,x2,x3=st.columns(3)
 rin=x1.number_input('RIN',0.,10.,7.,.1)
 aut=x2.number_input('Autolysis score',0.,20.,0.,.1)
 age=x3.number_input('Age',0.,120.,50.,1.)
 x4,x5=st.columns(2)
 sex=x4.selectbox('Sex',['Female','Male','Unknown'])
 hardy=x5.number_input('Hardy death scale',0.,4.,0.,1.)

if choice in ['Microbiome only','RNA + Microbiome']:
 st.markdown(
  '<div class="section"><div class="section-title">🦠 Microbial succession evidence</div>'
  '<div class="section-sub">Provide the microbiome / OTU sample data and environmental metadata.</div></div>',
  unsafe_allow_html=True
 )
 mf=st.file_uploader('Microbiome / OTU CSV / TSV',type=['csv','tsv','txt'],key='mf')
 q1,q2=st.columns(2)
 indoor=q1.number_input('Indoor ADD',0.,10000.,0.,.1)
 outdoor=q2.number_input('Outdoor ADD',0.,10000.,0.,.1)
 q3,q4=st.columns(2)
 env=q3.selectbox('Environment',['unknown','indoor','outdoor','indoor_outdoor'])
 site=q4.text_input('Body site','unknown')

# ----------------------------------------------------------------------
# GENERATE ONLY AFTER EVIDENCE HAS BEEN PROVIDED
# ----------------------------------------------------------------------
if choice == 'RNA + Microbiome':
 ready = (rf is not None and mf is not None)
 button_label = '🔬 Generate Final PMI Estimate'
elif choice == 'RNA only':
 ready = (rf is not None)
 button_label = '🧬 Generate RNA PMI Estimate'
else:
 ready = (mf is not None)
 button_label = '🦠 Generate Microbiome PMI Estimate'

generate = st.button(
 button_label,
 type='primary',
 use_container_width=True,
 disabled=not ready
)

if not ready:
 if choice == 'RNA + Microbiome':
  st.info('Upload both the RNA expression file and the microbiome file to generate the final PMI estimate.')
 elif choice == 'RNA only':
  st.info('Upload the RNA expression file to generate the PMI estimate.')
 else:
  st.info('Upload the microbiome file to generate the PMI estimate.')

if generate:
 # ----------------------------- RNA -----------------------------
 if choice in ['RNA only','RNA + Microbiome']:
  try:
   sep='\t' if rf.name.lower().endswith(('.tsv','.txt')) else ','
   raw=pd.read_csv(io.BytesIO(rf.getvalue()),sep=sep,low_memory=False)
   first=raw.columns[0]

   if raw.shape[1]==2:
    expr=pd.DataFrame(
     [raw.iloc[:,1].astype(float).values],
     columns=raw.iloc[:,0].astype(str)
    )
   elif raw.shape[0]==1:
    expr=raw.copy()
    expr.columns=expr.columns.astype(str)
   else:
    expr=raw.set_index(first).T
    expr.columns=expr.columns.astype(str)

   md=pd.DataFrame([{
    'rin':rin,
    'autolysis':aut,
    'age':age,
    'sex':1 if sex=='Male' else 0 if sex=='Female' else np.nan,
    'hardy':hardy if hardy>0 else np.nan
   }])

   r_hours,rX,rnames,rxgb=predict_rna(
    rna_pkg,tissue,expr,md
   )
   r_tab,r_fig=shap_one(rxgb,rX,rnames)

  except Exception as e:
   st.error(f'RNA processing error: {e}')
   r_hours=None

 # ------------------------- MICROBIOME --------------------------
 if choice in ['Microbiome only','RNA + Microbiome']:
  try:
   sep='\t' if mf.name.lower().endswith(('.tsv','.txt')) else ','
   raw=pd.read_csv(
    io.BytesIO(mf.getvalue()),
    sep=sep,
    index_col=0,
    low_memory=False
   )
   raw=raw.apply(pd.to_numeric,errors='coerce').fillna(0)
   otu=raw.T.iloc[[0]]

   m_hours,mX,mnames,mxgb=predict_micro(
    micro_pkg,otu,indoor,outdoor,env,site
   )
   m_tab,m_fig=shap_one(mxgb,mX,mnames)

  except Exception as e:
   st.error(f'Microbiome processing error: {e}')
   m_hours=None

 # ----------------------------- RESULT ---------------------------
 if choice == 'RNA + Microbiome' and r_hours is not None and m_hours is not None:
  final,primary,rw,mw=fuse(r_hours,m_hours)

  st.markdown('---')
  st.markdown(
   f'<div class="result"><div class="result-cap">Estimated postmortem interval</div>'
   f'<div class="result-num">{pmi(final)}</div>'
   f'<div class="note">Primary evidence source: <b>{primary}</b></div></div>',
   unsafe_allow_html=True
  )

  z1,z2,z3=st.columns(3)
  z1.metric('RNA estimate',pmi(r_hours))
  z2.metric('Microbiome estimate',pmi(m_hours))
  z3.metric('Assessment','Fused')

  if 'fusion_details' in st.session_state:
   fd=st.session_state['fusion_details']
   st.caption(
    f"Saved calibration + case-specific uncertainty • "
    f"Calibrated: RNA {fd['rna_calibrated_hours']:.2f} h, "
    f"Microbiome {fd['microbiome_calibrated_hours']:.2f} h • "
    f"Uncertainty: RNA {fd['rna_uncertainty_hours']:.2f} h, "
    f"Microbiome {fd['microbiome_uncertainty_hours']:.2f} h"
   )

 elif choice == 'RNA only' and r_hours is not None:
  final=r_hours
  primary='RNA'
  rw=1.0
  mw=0.0

  st.markdown('---')
  st.markdown(
   f'<div class="result"><div class="result-cap">Estimated postmortem interval</div>'
   f'<div class="result-num">{pmi(final)}</div>'
   f'<div class="note">Primary evidence source: <b>RNA</b></div></div>',
   unsafe_allow_html=True
  )
  st.metric('RNA estimate',pmi(r_hours))

 elif choice == 'Microbiome only' and m_hours is not None:
  final=m_hours
  primary='Microbiome'
  rw=0.0
  mw=1.0

  st.markdown('---')
  st.markdown(
   f'<div class="result"><div class="result-cap">Estimated postmortem interval</div>'
   f'<div class="result-num">{pmi(final)}</div>'
   f'<div class="note">Primary evidence source: <b>Microbiome</b></div></div>',
   unsafe_allow_html=True
  )
  st.metric('Microbiome estimate',pmi(m_hours))

 else:
  final=primary=rw=mw=None

 # ------------------------- EXPLAINABILITY ------------------------
 if r_tab is not None or m_tab is not None:
  tab_labels=(['RNA'] if r_tab is not None else [])+(['Microbiome'] if m_tab is not None else [])
  tabs=st.tabs(tab_labels)
  i=0

  if r_tab is not None:
   with tabs[i]:
    st.pyplot(r_fig,use_container_width=True)
    st.dataframe(r_tab,use_container_width=True,hide_index=True)
   i+=1

  if m_tab is not None:
   with tabs[i]:
    st.pyplot(m_fig,use_container_width=True)
    st.dataframe(m_tab,use_container_width=True,hide_index=True)

 # ----------------------------- REPORT ----------------------------
 if final is not None:
  data=pdf(
   case,
   tissue,
   'RNA + Microbiome' if r_hours is not None and m_hours is not None
   else ('RNA' if r_hours is not None else 'Microbiome'),
   r_hours,
   m_hours,
   final,
   primary,
   rw,
   mw,
   {'RNA':r_tab,'Microbiome':m_tab}
  )

  if data:
   st.download_button(
    '📄 Download Professional Forensic Report',
    data=data,
    file_name=f'ForensicChrono_{case}.pdf',
    mime='application/pdf',
    type='primary'
   )

  st.markdown(
   '<div class="warn"><b>Interpretation:</b> This system is a research decision-support tool '
   'and should not be used as a standalone determination of time of death. Results should be '
   'interpreted together with conventional forensic evidence.</div>',
   unsafe_allow_html=True
  )

