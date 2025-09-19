# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd
from pathlib import Path

def trapmf_piece(x, a,b,c,d):
    eps=1e-9
    if b<=a: b=a+eps
    if d<=c: c=d-eps
    x=float(x)
    if x<=a or x>=d: return 0.0
    if a<x<b: return (x-a)/(b-a)
    if b<=x<=c: return 1.0
    if c<x<d: return (d-x)/(d-c)
    return 0.0

def eval_mfs(row):
    out={}
    for f in ['f_device','f_country','f_asn']:
        v=row[f]
        out[f]={
            'L': trapmf_piece(v,0.0,0.05,0.2,0.4),
            'M': trapmf_piece(v,0.2,0.4,0.7,0.85),
            'G': trapmf_piece(v,0.7,0.85,0.9,0.95),
            'E': trapmf_piece(v,0.9,0.95,0.999,1.000001),
        }
    for f in ['f_ip_clean','f_success']:
        out[f]={'L': float(row[f]<0.5), 'M':0.0, 'G':0.0, 'E': float(row[f]>=0.5)}
    return out

def mamdani_priority(mus):
    # Safety-first rule base
    if mus['f_ip_clean']['L'] >= 0.5:  # bad IP
        return {'L':1.0,'M':0.0,'G':0.0,'E':0.0}
    strongest={f:max(mus[f], key=lambda k: mus[f][k]) for f in mus}
    counts={g:list(strongest.values()).count(g) for g in ['L','M','G','E']}
    if counts['E']>=2 and mus['f_success']['E']>=0.5:
        return {'L':0.0,'M':0.0,'G':0.0,'E':1.0}
    if (counts['G']+counts['E'])>=3:
        return {'L':0.0,'M':0.0,'G':1.0,'E':0.0}
    unfamiliar=sum(1 for f in ['f_device','f_country','f_asn']
                   if mus[f]['L']>mus[f]['M'] and mus[f]['L']>mus[f]['G'])
    if mus['f_success']['L']>=0.5 and unfamiliar>=2:
        return {'L':1.0,'M':0.0,'G':0.0,'E':0.0}
    if counts['M']>=2:
        return {'L':0.0,'M':1.0,'G':0.0,'E':0.0}
    tot=sum(counts.values())
    return {k:counts[k]/tot for k in counts}

def defuzz(fz, y={'L':0.10,'M':0.40,'G':0.70,'E':0.95}):
    num=sum(y[k]*fz[k] for k in fz); den=sum(fz.values())+1e-12
    return num/den

def main(xlsx, out_csv):
    src=Path(xlsx); out=Path(out_csv)
    df_raw=pd.read_excel(src)

    # Per-user familiarity ratios
    user_tot=df_raw.groupby('User ID').size().rename('user_total')
    dev=df_raw.groupby(['User ID','Device Type']).size().rename('user_device_count')
    cty=df_raw.groupby(['User ID','Country']).size().rename('user_country_count')
    asn=df_raw.groupby(['User ID','ASN']).size().rename('user_asn_count')
    df=(df_raw.join(user_tot, on='User ID')
              .join(dev, on=['User ID','Device Type'])
              .join(cty, on=['User ID','Country'])
              .join(asn, on=['User ID','ASN']))
    df['f_device']=df['user_device_count']/df['user_total'].clip(lower=1)
    df['f_country']=df['user_country_count']/df['user_total'].clip(lower=1)
    df['f_asn']=df['user_asn_count']/df['user_total'].clip(lower=1)
    df['f_ip_clean']=(~df['Is Attack IP']).astype(int)
    df['f_success']=df['Login Successful'].astype(int)

    trusts, labels=[], []
    for _,row in df.iterrows():
        mus=eval_mfs(row); outset=mamdani_priority(mus); T=defuzz(outset)
        trusts.append(T); labels.append(1 if T<0.5 else 0)
    df['fuzzy_trust']=trusts; df['silver_label']=labels

    keep=['Login Timestamp','User ID','IP Address','Country','Region','City','ASN',
          'User Agent String','Browser Name and Version','OS Name and Version',
          'Device Type','Login Successful','Is Attack IP','Is Account Takeover',
          'f_device','f_country','f_asn','f_ip_clean','f_success','fuzzy_trust','silver_label']
    df[keep].to_csv(out, index=False)
    print(f"Saved {out}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--xlsx', default='data/rba_test.xlsx')
    ap.add_argument('--out',  default='data/rba_fuzzy_labels.csv')
    args=ap.parse_args()
    main(args.xlsx, args.out)
