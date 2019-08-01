import os, sys
import pandas as pd
from collections import Counter, defaultdict

na_values = ["NA", "NONE", "N/A", "NULL", "NAN"]                                    
na_values += [na.lower() for na in na_values]

def load_base(name="meps_base_data.csv"):
    return pd.read_csv(name, encoding="utf8", na_values=na_values)

def load_meds(name="meps_meds.csv"):
    return pd.read_csv(name, encoding="utf8", na_values=na_values)

def drop_unnamed(x_df, cols=["Unnamed: 0"]):
    return x_df.drop(cols, axis=1)

def drop_duplicate_base_ids(x_df, cols=["id"]):
    return x_df.drop_duplicates(subset=cols, keep="last")

def drop_na_base(x_df, cols=["id", "panel", "pooledWeight", "age", "sex", "race"]):
    return x_df.dropna(axis=0, how="any", subset=cols)

def drop_na_meds(x_df):
    return x_df.dropna(axis=0, how="any")

def drop_na_comb(x_df):
    return x_df.dropna(axis=0, how="all",
            subset=[col for col in x_df.columns if col.endswith("Diagnosed")])

def fill_na_base(x_df):
    cols = [col for col in x_df.columns if col.endswith("Diagnosed")]
    x_df[cols] = x_df[cols].fillna("Inapplicable")
    return x_df

def filter_pos_base_age(x_df):
    return x_df.loc[x_df.age > 0]

def load_combined(base_df, meds_df):
    meds_df = meds_df.loc[meds_df.id.isin(set(base_df.id.values))]
    df = meds_df.groupby(["id", "rxName"])["rxQuantity"].agg(["sum"]).reset_index()
    return df.merge(base_df, left_on="id", right_on="id", how="left")

def load_base_numeric(x_df):
    res = pd.DataFrame()
    maps = {"Yes":1, "No":-1, "Inapplicable":0}
    for col in x_df.columns:
        if col.endswith("Diagnosed"):
            res[col] = x_df[col].map(maps)
    return res

df_base = (
    load_base()                              # load meps_base_data.csv
    .pipe(drop_unnamed)                      # drop the first column
    .pipe(drop_duplicate_base_ids)           # drop lines with duplicate ids (actually no such lines in the sample)
    .pipe(drop_na_base)                      # drop data that has invalid id, panel, poopledWeight, age, sex or race
    .pipe(fill_na_base)                      # fill invalid diagnosed data with 'Inapplicable'
)

df_meds = (
    load_meds()                              # load meps_meds.csv
    .pipe(drop_unnamed)                      # drop the first column
    .pipe(drop_na_meds)                      # drop rows that has nan
)

# Test if meds rows with invalid year data has different distriubtions
# df_meds_with_date = df_meds.loc[df_meds.rxStartYear < 0]       

df_comb = (                                   # 1. remove records in meds that have ids that do not exist in the base
    load_combined(df_base, df_meds)           # 2. group the meds data by id and rxName
)                                             # 3. left join the base data

def get_count(x_df):
    return len(x_df)

def most_common_medications(x_df):
    res = {}
    for col in x_df.columns:
        if col.endswith("Diagnosed"):                                
            df_temp = x_df.loc[x_df[col] == "Yes"]                          # get the sample that has been confirmed of each disease
            counts = df_temp.groupby("rxName")["id"].agg(get_count)         # count grouping by rxName
            if not len(counts) > 0: continue
            res[col] = counts.idxmax()                                      # get the most common meds
    return res

most_common = most_common_medications(df_comb)            # calculate the most common medicine for people with each disease
print("Most common medication for each disease:")
for key, val in most_common.items():
    print(f"{key}: {val}")

def meds_population_perc(x_df, diagnosed=["Yes"]):                        # this function calculates the percentage of 
    res = defaultdict(pd.Series)                                          # using a certain medicine with a certain diagnosed property
    for col in x_df.columns:
        if col.endswith("Diagnosed"):
            df_temp = x_df.loc[x_df[col].isin(diagnosed)]
            counts = df_temp.groupby("rxName")["id"].agg(get_count)
            counts /= len(df_temp)
            res[col] = counts.copy()
    return pd.DataFrame(res).fillna(0)

df_meds_pop_yes = meds_population_perc(df_comb)                           # calculate the usage of each medicine for diagnosed people
df_meds_pop_no = meds_population_perc(df_comb, ["No"])                    # calculate the usage of each medicine for healthy people

def calc_pop_diff(df_yes, df_no):
    res = defaultdict(pd.Series)
    for col in df_yes.columns:
        df_temp = pd.DataFrame({"Yes": df_yes[col], "No": df_no[col]}).fillna(0)
        res[col] = df_temp["Yes"] - df_temp["No"]
    return pd.DataFrame(res).fillna(0)

df_meds_pop_diff = calc_pop_diff(df_meds_pop_yes, df_meds_pop_no)         # take the difference between the two usages
print("\nMedications most indicative of each disease:")                   # take the medicine with the biggest difference
for col in df_meds_pop_diff.columns:
    print(f"{col}: {df_meds_pop_diff[col].idxmax()}")

def disease_population(df):
    tot_pop = len(df)
    res = {}
    for col in df.columns:
        if not col.endswith("Diagnosed"):
            continue
        df_temp = df.loc[df[col] == "Yes"]
        count = len(df_temp)
        res[col] = count / tot_pop
    return res

disease_stats = disease_population(df_base)                               # some disease statistics

df_base_numeric = (
    load_base_numeric(df_base)                                            # Yes -> 1.0, No -> -1.0, Inapplicable -> 0
)

print("Disease correlation:")                                             # correlations among the diseases
print(df_base_numeric.corr())


# Feature Extraction
df_asthma = df_base.loc[(df_base.age > 0) & (df_base.asthmaDiagnosed.isin(["Yes", "No"])),      # select patients diagnosed or excluded of asthma
        ["id", "sex", "pooledWeight", "age", "race", "asthmaDiagnosed"]]

from sklearn.preprocessing import LabelBinarizer
import numpy as np

race_encoder = LabelBinarizer()
race_1hot = race_encoder.fit_transform(df_asthma.race)                                           # 1hot conversion of the race, not needed for TF2

race_1hot = np.c_[df_asthma["id"].values, race_1hot]
df_asthma = df_asthma.merge(pd.DataFrame(race_1hot, 
    columns=["id"]+list(race_encoder.classes_)), on=["id"], how="left")

target = df_asthma.asthmaDiagnosed.map({"Yes": 1, "No": 0})
df_asthma["target"] = target


# Normalize rxQuantity
df_meds_by_id = df_meds.groupby(["id", "rxName", "rxForm"])["rxQuantity"].agg(["sum"])           # use aggregated rxQuantity (assumption)
df_meds_by_id.reset_index(inplace=True)
df_meds_by_id.rename(columns={"sum": "rxQuantity"}, inplace=True)

df_meds_by_name = df_meds_by_id.groupby(["rxName", "rxForm"])["rxQuantity"].agg(["count", "max", "min"])          # normalize rxQuantity by rxForm
df_meds_by_name.reset_index(inplace=True)
df_meds_by_id = df_meds_by_id.merge(df_meds_by_name, on=["rxName", "rxForm"], how="left")
df_meds_by_id.loc[df_meds_by_id["max"] == df_meds_by_id["min"], "normQuantity"] = 0.5
df_meds_by_id.loc[df_meds_by_id["max"] > df_meds_by_id["min"], "normQuantity"] \
        = (df_meds_by_id["rxQuantity"] - df_meds_by_id["min"]).div(df_meds_by_id["max"] - df_meds_by_id["min"])

topN = 30                  
most_common_meds = list(df_meds_pop_diff.asthmaDiagnosed.sort_values().tail(topN).index)         # select top 30 most common medicines as features

for med in most_common_meds:
    df_temp = df_meds_by_id.loc[df_meds_by_id.rxName == med, ["id", "normQuantity"]]
    df_asthma = df_asthma.merge(df_temp, on=["id"], how="left")
    df_asthma.rename(columns={"normQuantity": med}, inplace=True)
    df_asthma[med].fillna(0, inplace=True)

df_asthma["age"] = np.log(df_asthma["age"]) / np.log(10.) / 2.                                   # normalize the age
df_asthma["pooledWeight"] = np.log(df_asthma["pooledWeight"]) / 10.                              # normalize the weight

import re
new_columns = [re.sub(r'\W+', '_', col.strip()) for col in df_asthma.columns]
df_asthma.rename(columns=dict(zip(df_asthma.columns, new_columns)), inplace=True)                # remove special characters from the feature names

race_columns = [re.sub(r'\W+', '_', col.strip()) for col in list(race_encoder.classes_)]
meds_columns = [re.sub(r'\W+', '_', col.strip()) for col in most_common_meds]

feature_columns_tf = ["pooledWeight", "age", "race", "sex"] + meds_columns
cols = ["id"] + feature_columns_tf + ["target"]
df_asthma[cols].to_csv("dataset_tf.csv", index=False)                                            # dump the train/test dataset for TF2

feature_columns = ["pooledWeight", "age", "sex"] + race_columns + meds_columns
cols = ["id"] + feature_columns + ["target"]
df_asthma["sex"] = df_asthma.sex.map({"Male": 1, "Female": 0})
df_asthma[cols].to_csv("dataset.csv", index=False)                                               # dump the train/test dataset

