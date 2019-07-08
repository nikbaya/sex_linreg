#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:58:10 2018

Phenotypic association of sex for all phenotypes in UK Biobank Round 2.

Three related models of linear regression, with progressively fewer sex-related predictors.

@author: nbaya
"""
import hail as hl
import pandas as pd
import datetime
import argparse
import numpy as np


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phsource', type=str, required=True, help="phenotype source (phesant, icd10, finngen)")
parser.add_argument('--parsplit', type=int, required=True, help="number of batches to split phsource phenotypes into")
parser.add_argument('--paridx', type=int, required=True, help="batch id number")

args = parser.parse_args()

paridx = args.paridx

phsource = args.phsource

wd = 'gs://nbaya/sex_linreg/'

print('\n####################')
print('Running phenotypes from ' + phsource)
print('Number of parsplit batches: ' + str(args.parsplit))
print('Batch paridx: ' + str(args.paridx))
print('####################')

if phsource == 'icd10':
    phen_tb_all = hl.import_table('gs://nbaya/linreg/icd10_phenotypes.both_sexes.tsv.bgz',missing='',impute=True)
elif phsource == 'phesant':
    phen_tb_all = hl.import_table('gs://phenotype_31063/ukb31063.phesant_phenotypes.both_sexes.tsv.bgz',missing='',impute=True,types={'"userId"': hl.tstr}).rename({'"userId"': 's'})
elif phsource == 'finngen':
    phen_tb_all = hl.import_table('gs://ukb31063-mega-gwas/phenotype-files/curated-phenotypes/2018-04-06_ukb-finngen-pheno-for-analysis.tsv',missing='',impute=True,types={'eid':hl.tstr}).rename({'eid':'s'})
phen_tb_all = phen_tb_all.annotate(s = hl.str(phen_tb_all.s))
phen_tb_all = phen_tb_all.key_by('s')

phenlist = [x.replace('\"','') for x in phen_tb_all.row_value][0:]

cov =  hl.import_table('gs://phenotype_31063/ukb31063.gwas_covariates.both_sexes.tsv',
                             key='s', impute=True, types={'s': hl.tstr})
cov = cov.annotate(age = cov.age - cov.aggregate(hl.agg.mean(cov.age))) #center age
cov = cov.annotate(age_squared = cov.age**2)
cov = cov.annotate(age_isFemale = cov.age*cov.isFemale)
cov = cov.annotate(age_squared_isFemale = cov.age_squared*cov.isFemale)

withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))

cov = cov.filter(hl.literal(withdrawn_set).contains(cov['s']),keep=False) #Remove withdrawn samples from covariates

cov_samples = set(cov.s.take(cov.count())) #sample IDs from covariates table
phen_tb_all = phen_tb_all.filter(hl.literal(cov_samples).contains(phen_tb_all['s']),keep=True) #Only keep samples from the newly filtered covariates

def get_cols(cov):
    cols = (['phen','r2_mul','r2_adj']+['beta_{:}'.format(i) for i in cov]+
            ['beta_PC{:}'.format(i) for i in range(1, 21)]+['se_{:}'.format(i) for i in cov],
            ['se_PC{:}'.format(i) for i in range(1, 21)])
    cols = [i for j in cols for i in j]
    return cols

cov1 = ['intercept']
cov2 = cov1+['sex']
cov3 = cov1+['age','age_square']
cov4 = np.asarray(list(set(cov2).union(set(cov3))))[[3,0,1,2]].tolist()
cov5 = cov4 + ['age_sex','age_square_sex']
cov6 = cov5 + ['sex_PC'+str(i) for i in range(1,21)]

covs = [cov1]+[cov2]+[cov3]+[cov4]+[cov5]+[cov6] #get list of covariate lists

cols = [get_cols(cov) for cov in covs] #get column names for different models

dfs = [pd.DataFrame(columns=col) for col in cols] #get list of dataframes for all models

numphens = len(phenlist)

start_idx = int((paridx-1)*numphens/args.parsplit)
stop_idx = int((paridx)*numphens/args.parsplit)
idx = range(start_idx,stop_idx,1) #chunks all phenotypes into parsplit number of chunks, then runs on the paridx-th chunk

for i in idx:
    phen = phenlist[i]
    
    print('############')
    print('Running phenotype '+phen+' (iter '+str(i+1)+')')
    print('iter '+str(idx.index(i)+1)+' of '+str(len(idx))+' for parallel batch '+str(paridx))
    starttime = datetime.datetime.now()
    print('############')
    
    if phsource == 'phesant':
        phen_tb = phen_tb_all.select('"'+phen+'"').join(cov, how='inner') #join phenotype and covariates    
        phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb['"'+phen+'"']))
        phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
        if phen_tb['"'+phen+'"'].dtype == hl.dtype('bool'):
            phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str.replace('\"',''))).drop('"'+phen+'"')
        else:
            phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"',''))).drop('"'+phen+'"')
    elif phsource == 'icd10' or phsource == 'finngen':
        phen_tb = phen_tb_all.select(phen).join(cov, how='inner') #join phenotype and covariates    
        phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb[phen]))
        phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
        if phen_tb[phen].dtype == hl.dtype('bool'):
            phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str)).drop(phen)
        else:
            phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"',''))).drop(phen)
            
    n = phen_tb.count()
    print('\n>>> Sample count for phenotype '+phen+': '+str(n)+' <<<')
    
    for cov_i, cov in enumerate(covs):
        if 'sex' not in cov or phen_tb.filter(phen_tb.isFemale == 1).count() % n != 0: #don't run regression if sex in cov AND trait is sex specific
            cov_list = [phen_tb[x.replace('sex','isFemale')] for x in cov if x is not 'intercept']
            reg = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list))
            stats = [phen,reg.multiple_r_squared,reg.adjusted_r_squared,reg.beta,
                      reg.standard_error]
            stats = [i for j in stats for i in j] #flatten list
            dfs[cov_i].loc[i] = stats #enter regression result in df at index cov_i in dfs, list of dataframes
        
    stoptime = datetime.datetime.now()

    print('\n############')
    print('Iteration time for '+phen+': '+str(round((stoptime-starttime).seconds/60, 2))+' minutes')
    print('############')
          
hl.Table.from_pandas(df1).export(wd+'ukb31063.'+phsource+'_phenotypes.both_sexes.reg1_batch'+str(args.paridx)+'.tsv.bgz',header=True)
hl.Table.from_pandas(df2).export(wd+'ukb31063.'+phsource+'_phenotypes.both_sexes.reg2_batch'+str(args.paridx)+'.tsv.bgz',header=True)
hl.Table.from_pandas(df3).export(wd+'ukb31063.'+phsource+'_phenotypes.both_sexes.reg3_batch'+str(args.paridx)+'.tsv.bgz',header=True)
