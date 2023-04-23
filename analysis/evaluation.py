from pathlib import Path
import pandas as pd
from glob import glob
from tqdm import tqdm

from .constants import SITE

def get_files(csv_dir):
    return sorted(csv_dir.glob(('*_' + SITE + '_analysis.csv')))

def get_basic_df(csv_dir):
    csv_files = get_files(csv_dir)
    df = pd.read_csv(csv_files[0])
    new_df = df.get(['Preprocess', 'CNN', 'DR', 'Clustering', 'Pred labels', 'Pred species'])
    return new_df

def get_single_column_dfs(csv_dir, column_name:str):
    csv_files = get_files(csv_dir)
    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        dfs.append(df[column_name])
    return dfs

def get_mean_values(dfs):
    divisor = len(dfs)
    sum_column = sum(dfs) / divisor
    return sum_column

def get_micro_f1_mean(csv_dir): # is essentially 'accuracy' in a multi-class scenario
    column_name = 'Micro F1-Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Micro F1-Score'
    return column, new_c_name

def get_macro_f1_mean(csv_dir):
    column_name = 'Macro F1-Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Macro F1-Score'
    return column, new_c_name

def get_f_star_mean(csv_dir):
    column_name = 'F-Star Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean F-Star Score'
    return column, new_c_name

def get_cohen_kappa_mean(csv_dir):
    column_name = 'Cohens Kappa'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Cohens Kappa'
    return column, new_c_name

def get_mcc_mean(csv_dir):
    column_name = 'Matthews correlation coefficient'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Matthews correlation coefficient'
    return column, new_c_name

def get_nmi_mean(csv_dir):
    column_name = 'NMI'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean NMI'
    return column, new_c_name

def get_complete_mean_df(csv_dir):
    new_df = get_basic_df(csv_dir)
    micro_mean, micro_name = get_micro_f1_mean(csv_dir)
    new_df[micro_name] = micro_mean
    macro_mean, macro_name = get_macro_f1_mean(csv_dir)
    new_df[macro_name] = macro_mean
    f_star_mean, f_star_name = get_f_star_mean(csv_dir)
    new_df[f_star_name] = f_star_mean
    cohen_kappa_mean, cohen_kappa_name = get_cohen_kappa_mean(csv_dir)
    new_df[cohen_kappa_name] = cohen_kappa_mean
    mcc_mean, mcc_name = get_mcc_mean(csv_dir)
    new_df[mcc_name] = mcc_mean
    nmi_mean, nmi_name = get_nmi_mean(csv_dir)
    new_df[nmi_name] = nmi_mean
    return new_df