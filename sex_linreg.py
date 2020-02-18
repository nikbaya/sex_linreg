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
#import numpy as np
#import random


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phsource', type=str, required=True, help="phenotype source (phesant, icd10, finngen)")
parser.add_argument('--parsplit', type=int, required=True, help="number of batches to split phsource phenotypes into")
parser.add_argument('--paridx', type=int, required=True, help="batch id number")

args = parser.parse_args()

phsource = args.phsource
parsplit = args.parsplit
paridx = args.paridx

wd = 'gs://nbaya/sex_linreg/'

models_to_run = [14,15] # which regression model(s) to run

print('\n####################')
print(f'Running phenotypes from {phsource}')
print(f'Number of parsplit batches: {parsplit}')
print(f'Batch paridx: {paridx}')
print(f'Models to run: {models_to_run}')
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
#add sibling info and family-based analysis indicator
phesant_sibs = hl.import_table('gs://ukb31063/ukb31063.PHESANT_January_2019.both_sexes.tsv.bgz',
                          missing='',key='s', impute=True, types={'s': hl.tstr}).select('1883','1873')    
cov_tb = cov_tb.annotate(sisters = hl.int(phesant_sibs[cov_tb.s]['1883'])) # 1883: "Number of full sisters"
cov_tb = cov_tb.annotate(brothers = hl.int(phesant_sibs[cov_tb.s]['1873'])) # 1873: "Number of full brothers"
cov_tb = cov_tb.annotate(in_fba = 0)
in_fba = hl.import_table('gs://nbaya/sex_linreg/ukb31063_fampairs_EURonly.tsv')
in_fba = set(in_fba.ID1.collect()+in_fba.ID2.collect())
cov_tb = cov_tb.annotate(in_fba = hl.literal(in_fba).contains(cov_tb['s']))
cov_tb = cov_tb.annotate(siblings = cov_tb.brothers + cov_tb.sisters)
cov_tb = cov_tb.annotate(brothers_minus_sisters = cov_tb.brothers - cov_tb.sisters)
cov_tb = cov_tb.annotate(same_sex_sibs = cov_tb.isFemale*cov_tb.sisters + (~cov_tb.isFemale)*cov_tb.brothers)
cov_tb = cov_tb.annotate(opposite_sex_sibs = (~cov_tb.isFemale*cov_tb.sisters) + (cov_tb.isFemale)*cov_tb.brothers)
cov_tb = cov_tb.annotate(same_minus_opposite_sex_sibs = cov_tb.same_sex_sibs-cov_tb.opposite_sex_sibs)
cov_tb.show()

withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True).key_by('f0')
cov_tb = cov_tb.filter(hl.is_defined(withdrawn[cov_tb.s]),keep=False) #Remove withdrawn samples from covariates

#cov_samples = set(cov_tb.s.collect()) #sample IDs from covariates table
#phen_tb_all = phen_tb_all.filter(hl.literal(cov_samples).contains(phen_tb_all['s']),keep=True) #Only keep samples from the newly filtered covariates
phen_tb_all = phen_tb_all.filter(hl.is_defined(cov_tb[phen_tb_all.s]))


print(f'\nNumber of individuals: {phen_tb_all.count()}')
#print(phen_tb_all.describe())
#print(cov_tb.describe())

def get_cols(cov):
    cols = (['phen','r2_mul','r2_adj']+['beta_{:}'.format(i) for i in cov]+
            ['se_{:}'.format(i) for i in cov])
    return cols

# Note: intercept and first 20 PCs are always added
cov1 = []
cov2 = ['sex']                  # sex
cov3 = ['age','age_squared']    # age + age_squared; 
cov4 = cov2+cov3                # sex + age + age_squared
cov5 = ['sex','age','age_sex']  # sex, age, age_sex
cov6 = cov4 + ['age_sex','age_squared_sex'] # sex + age + age_squared + age_sex + age_squared_sex; already has some phenotypes completed
cov7 = cov6 + ['sex_PC'+str(i) for i in range(1,21)] # sex + age + age_squared + age_sex + age_squared_sex + sex * (20 PCs)
cov8 = cov6 + ['sisters','brothers'] # sex + age + age_squared + age_sex + age_squared_sex + sisters + brothers; default GWAS covariates + siblings
cov9 = cov8 + ['in_fba'] # in_fba: "in family-based analsis"
cov10 = cov6 + ['sisters'] # sex + age + age_squared + age_sex + age_squared_sex + sisters; 
cov11 = cov6 + ['brothers'] # sex + age + age_squared + age_sex + age_squared_sex + sisters;
cov12 = cov6 + ['siblings'] 
cov13 = cov6 + ['brothers_minus_sisters']
cov14 = cov6 + ['same_sex_sibs','opposite_sex_sibs']
cov15 = cov6 + ['same_minus_opposite_sex_sibs']


covs = ([cov1]+[cov2]+[cov3]+[cov4]+[cov5]+[cov6]+[cov7]+[cov8]+[cov9]+[cov10]+ #get list of covariate lists; DO NOT EDIT WHEN RUNNING ONLY A SUBSET OF MODELS
        [cov11]+[cov12]+[cov13]+[cov14]+[cov15]) 
covs = [['intercept']+cov+[f'PC{i}' for i in range(1,21)] for cov in covs] #add intercept and 20 PCs to the end of the list of covariates

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
    print(f'Running phenotype {phen} (phen idx {i+1})')
    print(f'iter {idx.index(i)+1} of {len(idx)} for parallel batch {paridx} of {parsplit}')
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
        if cov_i+1 in models_to_run: #only run models in models_to_run
            if 'sex' not in cov or phen_tb.filter(phen_tb.isFemale == 1).count() % n != 0: #don't run regression if sex in cov AND trait is sex specific
                print(f'\n############\nRunning linreg model {cov_i+1} for phen {phen}\n############\n')
                if 'intercept' in cov:
                    cov.remove('intercept')
                cov_list = [phen_tb[(x.replace('sex','isFemale') if 'sibs' not in x else x)] for x in cov] # change all terms with sex or cross terms with sex to isFemale, but ignore the sibling fields
                reg = phen_tb.aggregate(hl.agg.linreg(y=phen_tb.phen, x = [1]+cov_list))
                stats = [[phen],[reg.multiple_r_squared,reg.adjusted_r_squared],
                         reg.beta,reg.standard_error]
                if (reg.beta==None) or (reg.standard_error==None):
                    print(f'\n######## WARNING: phen {phen} linreg stats are None ##########')
                    print(stats)
                    stats = [phen]+[float('NaN')]*(2+2*(len(cov)+1))
                    print(len(stats))
                else:
                    stats = [i for j in stats for i in j] #flatten list
                    dfs[cov_i].loc[i] = stats #enter regression result in df at index cov_i in dfs, list of dataframes
                print(f'\nCompleted running model {cov_i+1} for phen {phen}')
            else:
                print(f'\n#############\nWARNING: Trait is sex-specific. Not able to use sex as a covariate for model {cov_i+1}.\n###########\n')
        
    stoptime = datetime.datetime.now()
    
#    print(dfs)

    print('\n############'+
          f'\nIteration time for {phen} in {phsource} (n: {n}): {round((stoptime-starttime).seconds/60, 2)} minutes'+
          '\n############')

for df_i, df in enumerate(dfs):
    if df_i+1 in models_to_run: #only run models in models_to_run
        try:
            hl.Table.from_pandas(df).export(wd+'batches/'+'ukb31063.'+phsource+f'_phenotypes.both_sexes.reg{df_i+1}_batch'+str(args.paridx)+'.tsv.bgz',header=True)
        except ValueError:
            print(f'The dataframe for model {df_i+1} is empty.\n\n{df}')
