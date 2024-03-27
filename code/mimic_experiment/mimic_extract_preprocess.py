# %%
import pandas as pd
import numpy as np

# DATA_FILEPATH     = '/Volumes/LaCie/datasets/Physionet/mimic_extract/all_hourly_data.h5'
DATA_FILEPATH     = './data/mimic/all_hourly_data.h5'

GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
SEED              = 1
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']

np.random.seed(SEED)

data_full_lvl2 = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
statics        = pd.read_hdf(DATA_FILEPATH, 'patients')

# %%
data_full_lvl2.head()

# %%
len(set(data_full_lvl2.columns.get_level_values(0)))

# %%
statics.head()

# %%
def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


# %% This was not mentioned in Suriyakumar paper: extra preprocess to match cohort selection in Suriyakumar paper
# drop the NaN for diagnosis_at_admission (1 person)
statics = statics[statics['diagnosis_at_admission'].notnull()]
# filter rows to los_icu > 36 hours
statics = statics[statics['los_icu'] > 1.5]
statics['los_icu'].describe()

# %%
Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
Ys['los_3'] = Ys['los_icu'] > 3
Ys['los_7'] = Ys['los_icu'] > 7
# Ys.drop(columns=['los_icu'], inplace=True)
Ys.astype(float)

df = data_full_lvl2
lvl2 = df[
    (df.index.get_level_values('icustay_id').isin(set(Ys.index.get_level_values('icustay_id')))) &
    (df.index.get_level_values('hours_in') < WINDOW_SIZE) ]  

train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
lvl2_subj_idx, Ys_subj_idx = [df.index.get_level_values('subject_id') for df in (lvl2, Ys)]
lvl2_subjects = set(lvl2_subj_idx)
assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"

# shuffle the dataset
np.random.seed(SEED)
subjects, N = np.random.permutation(list(lvl2_subjects)), len(lvl2_subjects)
N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
train_subj = subjects[:N_train]
dev_subj   = subjects[N_train:N_train + N_dev]
test_subj  = subjects[N_train+N_dev:]

[(lvl2_train, lvl2_dev, lvl2_test), (Ys_train, Ys_dev, Ys_test)] = [
    [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
    for df in (lvl2,  Ys)
]


# %%
## Save targets: 
# Ys.to_csv('data/mimic/Ys_mimic_extract.csv')
Ys_train.to_csv('data/mimic/Ys_train.csv')
Ys_dev.to_csv('data/mimic/Ys_dev.csv')
Ys_test.to_csv('data/mimic/Ys_test.csv')

## read in the cohort selection
# Ysr = pd.read_csv('data/mimic/Ys_mimic_extract.csv', index_col=[0,1,2])
# Ysr.head()

# %%
# normalize the dataset w/ z-score

idx = pd.IndexSlice
dfz_train, dfz_dev, dfz_test = lvl2_train.copy(), lvl2_dev.copy(), lvl2_test.copy()

lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:,'mean']].mean(axis=0), lvl2_train.loc[:, idx[:,'mean']].std(axis=0)
dfz_train.loc[:, idx[:,'mean']] = (dfz_train.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds

lvl2_means, lvl2_stds = lvl2_dev.loc[:, idx[:,'mean']].mean(axis=0), lvl2_dev.loc[:, idx[:,'mean']].std(axis=0)
dfz_dev.loc[:, idx[:,'mean']] = (dfz_dev.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds

lvl2_means, lvl2_stds = lvl2_test.loc[:, idx[:,'mean']].mean(axis=0), lvl2_test.loc[:, idx[:,'mean']].std(axis=0)
dfz_test.loc[:, idx[:,'mean']] = (dfz_test.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds

# %% l infinity normalization of the dataset
df_train, df_dev, df_test = lvl2_train.copy(), lvl2_dev.copy(), lvl2_test.copy()
lvl2_absmax = lvl2_train.loc[:, idx[:,'mean']].abs().max(axis=0)
df_train.loc[:, idx[:,'mean']] = (df_train.loc[:, idx[:,'mean']] )/lvl2_absmax

lvl2_absmax = lvl2_dev.loc[:, idx[:,'mean']].abs().max(axis=0)
df_dev.loc[:, idx[:,'mean']] = (df_dev.loc[:, idx[:,'mean']] )/lvl2_absmax

lvl2_absmax = lvl2_test.loc[:, idx[:,'mean']].abs().max(axis=0)
df_test.loc[:, idx[:,'mean']] = (df_test.loc[:, idx[:,'mean']] )/lvl2_absmax


# %%#compute rate of missingness 
df = pd.concat([lvl2_train, lvl2_dev, lvl2_test], axis = 0)
df = df.loc[:, idx[:, 'mean']]
print(f' rate of missingness before imputation is {df.isna().sum().sum()} / {df.shape[0] * df.shape[1]} = {df.isna().sum().sum() / (df.shape[0] * df.shape[1])}')

# %%
df_train.loc[:, idx[:, 'mean']].head()

# %%
# impute missing values
df_train, df_dev, df_test, dfz_train, dfz_dev, dfz_test = [
    simple_imputer(df) for df in (df_train, df_dev, df_test, dfz_train, dfz_dev, dfz_test)
    ]


# %%
for df in df_train, df_dev, df_test: assert not df.isnull().any().any()

# %% [markdown]
# ### save preprocess data as h5 file

# %%
# save the data at this point.
# in practice we'll probably want to pivot it after reading it in. 
# for GRU-D likely multiply the mask by the mean (killing off the imputed values) 
# likely will ignore the time_since columns. 
print('Saving data.....')
dfz_train.to_hdf('data/mimic/lvl2_z_normalized.h5', key='train')
dfz_dev.to_hdf('data/mimic/lvl2_z_normalized.h5', key='dev')
dfz_test.to_hdf('data/mimic/lvl2_z_normalized.h5', key='test')
print('Written data/mimic/lvl2_z_normalized.h5')

df_train.to_hdf('data/mimic/lvl2_l_inf_normalized.h5', key='train')
df_dev.to_hdf('data/mimic/lvl2_l_inf_normalized.h5', key='dev')
df_test.to_hdf('data/mimic/lvl2_l_inf_normalized.h5', key='test')
print('Written data/mimic/lvl2_l_inf_normalized.h5')


# %% [markdown]
# ## Examples of using saved data for ML: 

# %%
with pd.HDFStore('data/mimic/lvl2_l_inf_normalized.h5') as hdf:
    print(hdf.keys())
    print(hdf['train'].shape)
    
# re-load data example: 
df_train2 = pd.read_hdf('data/mimic/lvl2_l_inf_normalized.h5', key = 'train')
df_train2.head()

# %%
# pivot on the hourly measurements to make three levels
df3_train, df3_dev, df3_test = [
    df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) \
        for df in (df_train, df_dev, df_test )]
df3_train.head()

# %%
df3_train.loc[:, idx[:, 'mean']].head()

# %%
# CS, check shapes

print(df_train.shape)
print(df_dev.shape)
print(df_test.shape)
print()
print(df3_train.shape)
print(df3_dev.shape)
print(df3_test.shape)
print()
print(367512 * 312)
print(15313 * 7488)


