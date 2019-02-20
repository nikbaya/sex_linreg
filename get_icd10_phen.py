#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:35:22 2018

Filters icd10 phenotypes from ukb31063.raw.csv

@author: nbaya
"""

import hail as hl

ht = hl.import_table('gs://phenotype_31063/ukb31063.raw_phenotypes.tsv.bgz')
icd10_cols = [x for x in ht.row if x.startswith('41202')]
ht = ht.select(*(['eid']+icd10_cols))
ht = ht.rename({'eid': 's'})
ht = ht.key_by('s')

# create new column for each sample that is the set of all 3-character ICD10 codes given to that sample
first_col = icd10_cols[0]
ht = ht.annotate(set_of_codes=hl.set([ht[first_col][:3]]))
for col in icd10_cols[1:]:
    ht = ht.annotate(set_of_codes=ht['set_of_codes'].add(ht[col][:3]))
ht = ht.annotate(set_of_codes=ht['set_of_codes'].remove(hl.null(hl.tstr)))     # Remove NAs from the sets
ht = ht.select('set_of_codes')

# collect set of all 3-character ICD10 codes given to any sample
icd10_codes = [code for code in ht.aggregate(hl.agg.explode(lambda x: hl.agg.collect_as_set(x), ht['set_of_codes']))]

# iterate through collected list of 3-character ICD10 codes and add a new True/False field for each to the Hail table
ht = ht.annotate(**{code: ht['set_of_codes'].contains(code) for code in icd10_codes})

# drop set_of_codes field, no longer needed
ht = ht.drop('set_of_codes')


ht.export('gs://nbaya/sex_linreg/icd10_phenotypes.both_sexes.tsv.bgz')
