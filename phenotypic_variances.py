#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:24:51 2019

Calculate distribution and variance of residuals for model 3...
    y ~ intercept + 20 PCs + age + age^2
...stratified by sex.

@author: nbaya
"""

import hail as hl
import datetime as dt
import argparse
import subprocess

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phsource', type=str, required=True, help="phenotype source (phesant, icd10, finngen)")
parser.add_argument('--parsplit', type=int, required=True, help="number of batches to split phsource phenotypes into")
parser.add_argument('--paridx', type=int, required=True, help="batch id number")

args = parser.parse_args()

phsource = args.phsource
parsplit = args.parsplit
paridx = args.paridx

wd = 'gs://nbaya/sex_linreg/residuals/'

if phsource == 'phesant':
    phen_tb_path = 'gs://ukb31063/ukb31063.PHESANT_January_2019.both_sexes.tsv.bgz'
elif phsource == 'finngen':
    phen_tb_path = 'gs://ukb31063/ukb31063.FINNGEN_phenotypes.both_sexes.tsv.bgz'
elif phsource == 'icd10':
    phen_tb_path = 'gs://ukb31063/ukb31063.ICD10_phenotypes.both_sexes.tsv.bgz'
phen_tb_all = hl.import_table(phen_tb_path,missing='',impute=True,types={'s': hl.tstr}, key='s')

cov =  hl.import_table('gs://ukb31063/ukb31063.neale_gwas_covariates.both_sexes.tsv.bgz',
                          key='s', impute=True, types={'s': hl.tstr})
cov = cov.annotate(age = cov.age - cov.aggregate(hl.agg.mean(cov.age))) #center age
cov = cov.annotate(age_squared = cov.age**2)
cov = cov.annotate(age_isFemale = cov.age*cov.isFemale)
cov = cov.annotate(age_squared_isFemale = cov.age_squared*cov.isFemale)
cov = cov.annotate(**{f'isFemale_PC{i}':cov.isFemale*cov[f'PC{i}'] for i in range(1,21)})
cov = cov.rename(dict(zip(list(cov.row),[x.replace('isFemale','sex') for x in list(cov.row)])))
withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True).key_by('f0')
cov = cov.filter(hl.is_defined(withdrawn[cov.s]),keep=False) #Remove withdrawn samples from covariates
phen_tb_all = phen_tb_all.filter(hl.is_defined(cov[phen_tb_all.s]))

linreg = hl.import_table(f'gs://nbaya/sex_linreg/ukb31063.{phsource}_phenotypes.both_sexes.reg3.tsv.bgz',impute=True).to_pandas()
bad_cols = [x for x in linreg.columns.values if ('age_square' in x) or ('isFemale' in x)]
new_cols = [x.replace('square','squared').replace('isFemale','sex') for x in bad_cols]
new_cols = [((x.split('_')[0]+'_'+'_'.join(x.split('_')[1:-1])+'_'+x.split('_')[-1]) if ('age' in x and 'sex' in x) else x).strip('_') for x in new_cols ]
columns = dict(zip(bad_cols, new_cols))
print(columns)
linreg = linreg.rename(columns=columns) #update "bad" column names to have the same style as in the covariates table

def get_residuals(phen, linreg):
    start = dt.datetime.now()
    path_f = wd+f'ukb31063.{phsource}.{phen}.residuals.female.reg3.tsv.bgz'
    path_m = wd+f'ukb31063.{phsource}.{phen}.residuals.male.reg3.tsv.bgz'
    try:
        subprocess.check_output([f'gsutil','ls',path_f]) != None
        subprocess.check_output([f'gsutil','ls',path_m]) != None
        print(f'\n#############\n{phen} already completed!\n#############\n')
    except:
        print(f'\n############\nStarting phenotype {phen}\n############\n')
        phen_tb = phen_tb_all.select(phen).join(cov, how='inner') #join phenotype and covariate table
        phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb[phen]))
        phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
        if phen_tb[phen].dtype == hl.dtype('bool'):
            phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str.replace('\"','')))
        else:
            phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"','')))

        phen_betas = linreg[linreg.phen == phen][[x for x in linreg.columns.values if 'beta' in x]]
        betas = dict(zip(list(phen_betas.columns.values), phen_betas.values.tolist()[0]))
        fields = [x.replace('beta_','') for x in linreg.columns.values if 'beta_' in x]
        phen_tb = phen_tb.annotate(intercept = 1)
        phen_tb = phen_tb.annotate(y_hat = 0)
        phen_tb_f = phen_tb.filter(phen_tb.sex == 1) #female
        phen_tb_m = phen_tb.filter(phen_tb.sex == 0) #male
        phen_tb_f = phen_tb_f.annotate(**{f'y_hat': phen_tb_f.y_hat + phen_tb_f[f]*betas['beta_'+f] for f in fields})
        phen_tb_m = phen_tb_m.annotate(**{f'y_hat': phen_tb_m.y_hat + phen_tb_m[f]*betas['beta_'+f] for f in fields})
        phen_tb_f = phen_tb_f.annotate(res_f = phen_tb_f.phen-phen_tb_f.y_hat)
        phen_tb_m = phen_tb_m.annotate(res_m = phen_tb_m.phen-phen_tb_m.y_hat)
        phen_tb_f.select(phen,'y_hat','res_f').export(path_f)
        phen_tb_m.select(phen,'y_hat','res_m').export(path_m)
    phen_tb_f = hl.import_table(path_f,impute=True)
    phen_tb_m = hl.import_table(path_m,impute=True)
    res_var_f = phen_tb_f.aggregate(hl.agg.stats(phen_tb_f.res_f)).stdev**2 if phen_tb_f.count() > 0 else float('nan')
    res_var_m = phen_tb_m.aggregate(hl.agg.stats(phen_tb_m.res_m)).stdev**2 if phen_tb_m.count() > 0 else float('nan')
    linreg.loc[linreg.phen==phen,'resid_var_f'] = res_var_f
    linreg.loc[linreg.phen==phen,'resid_var_m'] = res_var_m
    print(f'Variance of residual for {phen} females:\t{linreg[linreg.phen==phen].resid_var_f.values[0]}')
    print(f'Variance of residual for {phen} males:\t{linreg[linreg.phen==phen].resid_var_m.values[0]}')
    print(f'\n############\nIteration time for {phen}: {round((dt.datetime.now()-start).seconds/60, 2)} minutes\n############')
    return linreg

phens = linreg.phen.values.tolist()
start_idx = int((paridx-1)*len(phens)/args.parsplit)
stop_idx = int((paridx)*len(phens)/parsplit)
idx = range(start_idx,stop_idx,1) #chunks all phenotypes for phsource into parsplit number of chunks, then runs on the paridx-th chunk

for i in idx:
    phen = phens[i]
    linreg = get_residuals(phen, linreg)
    print(linreg[~linreg.resid_var_f.isna() | ~linreg.resid_var_m.isna()][['phen','resid_var_f','resid_var_m']])
    
hl.Table.from_pandas(linreg[~linreg.resid_var_f.isna() | ~linreg.resid_var_m.isna()][['phen','resid_var_f','resid_var_m']]).export(wd+f'batches/ukb31063.{phsource}.residual_variances.reg3.batch{paridx}.tsv.bgz')
