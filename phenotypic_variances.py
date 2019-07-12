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

phsource = 'phesant'
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

def get_residuals(phen, linreg):
    start = dt.datetime.now()
    print(f'\n############\nStarting {phen}\n############\n')
    phen_tb = phen_tb_all.select(phen).join(cov, how='inner') #join phenotype and covariate table
    phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb[phen]))
    phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
    if phen_tb[phen].dtype == hl.dtype('bool'):
        phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str.replace('\"','')))
    else:
        phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"','')))

    n = phen_tb.count()
    print(f'\n>>> Sample count for phenotype {phen}: {n} <<<')
    
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
    phen_tb_f.select(phen,'y_hat','res_f').export(f'gs://nbaya/sex_linreg/ukb31063.{phen}.residuals.female.reg3.tsv.bgz')
    phen_tb_m.select(phen,'y_hat','res_m').export(f'gs://nbaya/sex_linreg/ukb31063.{phen}.residuals.male.reg3.tsv.bgz')
#    linreg.loc[linreg.phen==phen,'resid_var_f'] = phen_tb_f.aggregate(hl.agg.stats(phen_tb_f.res_f)).stdev**2
#    linreg.loc[linreg.phen==phen,'resid_var_m'] = phen_tb_m.aggregate(hl.agg.stats(phen_tb_m.res_m)).stdev**2
#    print(f'Variance of residual for {phen} females: {linreg[linreg.phen==phen].resid_var_f.values[0]}')
#    print(f'Variance of residual for {phen} males: {linreg[linreg.phen==phen].resid_var_m.values[0]}')
    print(f'\n############\nIteration time for {phen} (n: {n}): {round((dt.datetime.now()-start).seconds/60, 2)} minutes\n############')
    return phen_tb_f,  phen_tb_m, linreg

for phen in linreg.phen.values.tolist():
    phen_tb_f,  phen_tb_m, linreg = get_residuals(phen, linreg)
    print(linreg[~linreg.resid_var_f.isna()].resid_var_f)
    
