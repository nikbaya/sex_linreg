#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:58:10 2018

Phenotypic association of sex for all phenotypes in UK Biobank Round 2.

Nested linear regression models, without genotypes.

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

phsource = args.phsource
parsplit = args.parsplit
paridx = args.paridx

wd = 'gs://nbaya/sex_linreg/'

print('\n####################')
print('Running phenotypes from ' + phsource)
print('Number of parsplit batches: ' + str(parsplit))
print('Batch paridx: ' + str(paridx))
print('####################')

      
if phsource == 'phesant':
    phen_tb_path = 'gs://ukb31063/ukb31063.PHESANT_January_2019.both_sexes.tsv.bgz'
elif phsource == 'finngen':
    phen_tb_path = 'gs://ukb31063/ukb31063.FINNGEN_phenotypes.both_sexes.tsv.bgz'
elif phsource == 'icd10':
    phen_tb_path = 'gs://ukb31063/ukb31063.ICD10_phenotypes.both_sexes.tsv.bgz'

phen_tb_all = hl.import_table(phen_tb_path,missing='',impute=True,types={'s': hl.tstr}, key='s')

phenlist = [x.replace('\"','') for x in phen_tb_all.row_value][0:]

cov_tb =  hl.import_table('gs://ukb31063/ukb31063.neale_gwas_covariates.both_sexes.tsv.bgz',
                          key='s', impute=True, types={'s': hl.tstr})
cov_tb = cov_tb.annotate(age = cov_tb.age - cov_tb.aggregate(hl.agg.mean(cov_tb.age))) #center age
cov_tb = cov_tb.annotate(age_squared = cov_tb.age**2)
cov_tb = cov_tb.annotate(age_isFemale = cov_tb.age*cov_tb.isFemale)
cov_tb = cov_tb.annotate(age_squared_isFemale = cov_tb.age_squared*cov_tb.isFemale)
cov_tb = cov_tb.annotate(**{f'isFemale_PC{i}':cov_tb.isFemale*cov_tb[f'PC{i}'] for i in range(1,21)})
cov_tb.show()

withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True).key_by('f0')
#withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))
#cov_tb = cov_tb.filter(hl.literal(withdrawn_set).contains(cov_tb['s']),keep=False) #Remove withdrawn samples from covariates
cov_tb = cov_tb.filter(hl.is_defined(withdrawn[cov_tb.s]),keep=False) #Remove withdrawn samples from covariates

#cov_samples = set(cov_tb.s.collect()) #sample IDs from covariates table
#phen_tb_all = phen_tb_all.filter(hl.literal(cov_samples).contains(phen_tb_all['s']),keep=True) #Only keep samples from the newly filtered covariates
phen_tb_all = phen_tb_all.filter(hl.is_defined(cov_tb[phen_tb_all.s]))

print(f'Number of individuals: {phen_tb_all.count()}')
#print(phen_tb_all.describe())
#print(cov_tb.describe())

def get_cols(cov):
    cols = (['phen','r2_mul','r2_adj']+['beta_{:}'.format(i) for i in cov]+
            ['se_{:}'.format(i) for i in cov])
    return cols

cov1 = []
cov2 = ['sex']
cov3 = ['age','age_squared'] #already has some phenotypes completed
cov4 = cov2+cov3 
cov5 = cov4 + ['age_sex','age_squared_sex'] # already has some phenotypes completed
cov6 = cov5 + ['sex_PC'+str(i) for i in range(1,21)]

covs = [cov1]+[cov2]+[cov3]+[cov4]+[cov5]+[cov6] #get list of covariate lists
covs = [['intercept']+cov+[f'PC{i}' for i in range(1,21)] for cov in covs] #add PCs to the end of the list of covariates

[print(cov) for cov in covs]

cols = [get_cols(cov) for cov in covs] #get column names for different models

dfs = [pd.DataFrame(columns=col) for col in cols] #get list of dataframes for all models

numphens = len(phenlist)

start_idx = int((paridx-1)*numphens/args.parsplit)
stop_idx = int((paridx)*numphens/parsplit)
idx = range(start_idx,stop_idx,1) #chunks all phenotypes for phsource into parsplit number of chunks, then runs on the paridx-th chunk

for i in idx:
    phen = phenlist[i]
    
    print('\n############')
    print(f'Running phenotype {phen} (iter {i+1})')
    print(f'iter {idx.index(i)+1} of {len(idx)} for parallel batch {paridx}')
    print('############')
    starttime = datetime.datetime.now()
    
    phen_tb = phen_tb_all.select(phen).join(cov_tb, how='inner') #join phenotype and covariate table
    phen_tb = phen_tb.annotate(phen_str = hl.str(phen_tb[phen]))
    phen_tb = phen_tb.filter(phen_tb.phen_str == '',keep=False)
    if phen_tb[phen].dtype == hl.dtype('bool'):
        phen_tb = phen_tb.annotate(phen = hl.bool(phen_tb.phen_str.replace('\"','')))
    else:
        phen_tb = phen_tb.annotate(phen = hl.float64(phen_tb.phen_str.replace('\"','')))
            
    n = phen_tb.count()
    print(f'\n>>> Sample count for phenotype {phen}: {n} <<<')
    
    for cov_i, cov in enumerate(covs):
        cov = cov.copy()
        if cov_i+1 in [3,4,5]: #only run models 3,4,5
            print(f'\nRunning model {cov_i+1} for phen {phen}\ncovs: {cov}\ncols: {cols[cov_i]}')
            if 'sex' not in cov or phen_tb.filter(phen_tb.isFemale == 1).count() % n != 0: #don't run regression if sex in cov AND trait is sex specific
                if 'intercept' in cov:
                    cov.remove('intercept')
                cov_list = [phen_tb[x.replace('sex','isFemale')] for x in cov]
                reg = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list))
                stats = [[phen],[reg.multiple_r_squared,reg.adjusted_r_squared],
                         reg.beta,reg.standard_error]
                stats = [i for j in stats for i in j] #flatten list
                dfs[cov_i].loc[i] = stats #enter regression result in df at index cov_i in dfs, list of dataframes
        
    stoptime = datetime.datetime.now()

    print('\n############'+
          f'\nIteration time for {phen} in {phsource} (n: {n}): {round((stoptime-starttime).seconds/60, 2)} minutes'+
          '\n############')

for df_i, df in enumerate(dfs):
    if df_i+1 in [3,4,5]: #only run models 3,4,5
        try:
            hl.Table.from_pandas(df).export(wd+'batches/'+'ukb31063.'+phsource+f'_phenotypes.both_sexes.reg{df_i+1}_batch'+str(20+args.paridx)+'.tsv.bgz',header=True)
        except ValueError:
            print(f'The dataframe for model {df_i+1} is empty.\n\n{df}')
