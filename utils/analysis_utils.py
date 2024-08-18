"""
Author: Kyungeun Lim
Date: 2021-04-05

This module provides a collection of derivative functions for data analysis using Pandas and other scientific computing libraries.
"""

# Standard library imports
import os
import time
from io import BytesIO
from functools import reduce
from itertools import combinations
from typing import List, Union

# Third-party imports for analysis
import numpy as np
import pandas as pd
import scipy

# import pymc3 as pm  # Uncomment if needed

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Google Cloud Platform (GCP) imports
from google.cloud import bigquery, storage
import google.cloud


def check_data(df:pd.DataFrame, unique_id:str) -> None:
    """
    Input: Pandas dataframe
    
    Return: None, prints details including data type, missing values, and number of unique values
    """
    print('-------- 1. Datatypes --------')
    print(df.dtypes)
    print('\n')
    print('-------- 2. Missing Values --------')
    print(create_missing_values_table(df))
    print('\n')
    print('-------- 3. Unique Values --------')
    print(print_col_uniques(df))
    print('\n')
    print('-------- 4. Checking Duplicates --------')
    df_dup = create_duplicated_df(df, unique_id)
    
    if df_dup.shape[0] > 0 :
        print(f"There are {df_dup.shape[0]} duplicates found with {unique_id}. Check the raw data!!")
    else:
        print(f"No duplicates found with {unique_id}")

        
def convert_to_datetime_type(df:pd.DataFrame, time_cols:List[str]) -> None:
    """ 
    Context: Sometimes pandas read time column with an object/str type
    
    Args: 
        1. df: Pandas dataframe
        2. time_cols: Columns with object type which need to be converted to datetime
          
    Return: None, just convert existing dataframe time cols to datetime type
    """
    for col in time_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col])
            
    print(f" --- Convert the data type from object to datetime for {time_cols}")
    print('\n')
       
    
def create_bucket_by_thr_col(df: pd.DataFrame, 
                             thr_col: str, 
                             thrs: List[Union[int, float]]) -> None:
    """
    Args:
        1. Pandas DataFrame
        2. Column to create bucket based on the thresholds
        3. Thresholds to create bucket

    Return: Pandas DataFrame contains f"{thr_col}_bucket" and f"{thr_col}_bucket_num"
    """

    if thrs[-1] < df[thr_col].max():
        raise ValueError(f"thrs[-1](={thrs[-1]}) should be larger than df[{thr_col}].max()(={df[thr_col].max()})")

    conditions = []
    vals_str = []

    for i, thr in enumerate(thrs):
        if i == 0:
            conditions.append((df[thr_col] <= thr).astype(bool))
            vals_str.append(f"<={thr}")
        else:
            conditions.append(((df[thr_col] > thrs[i-1]) & (df[thr_col] <= thr)).astype(bool))
            vals_str.append(f"{thrs[i-1]}-{thr}")

    df.loc[:, f"{thr_col}_bucket"] = np.select(conditions, vals_str, default = "")
    df.loc[:, f"{thr_col}_bucket_num"] = np.select(conditions, thrs, default = np.nan)    
    
    
def create_count_frac_and_uncertainty_cols(df:pd.DataFrame, 
                                           ratio_col:str, 
                                           numerator_col:str, 
                                           denominator_col:str) -> None:
    """
    Args:
        1. df: Pandas dataframes
        2. ratio_col: new column name for ratio
        3. numerator_col: numerator_col name
        4. denominator_col: numerator_col name
        
    Return: None, create two new columns in the existing pandas dataframe
    """
    df[ratio_col] = df[numerator_col]/df[denominator_col]
    df.loc[~np.isfinite(df[ratio_col]), ratio_col] = np.nan
    ## assuming the uncertainty (std) on the N counting is sqrt(N)
    df[f"{ratio_col}_err"] = df[ratio_col]*np.power((1./np.sqrt(df[numerator_col]))**2 + (1./np.sqrt(df[denominator_col]))**2, 0.5)    
    

def create_df_from_bq(query_txt:str,
                      project_number:str,
                      project_id:str) -> pd.DataFrame:
    """
    Args:
        1. query_txt: SQL query as str  
        2. project_number: Project number
        3. project_id: it can be used instead of project_number inside Client
    
    Return: Pandas dataframe
    """
    #client = bigquery.Client(project = project_number)
    client = bigquery.Client(project = project_id)
    start_time = time.time()
    df = client.query(query_txt).to_dataframe()
    print(f" --- Read from BigQuery time: {np.round(time.time() - start_time, 3)} s") 
    print(f" --- Create a pandas dataframe with {df.shape} shape")
    
    # standardize the columns
    standardize_columns(df)
    print(f" --- Standardize column names")
    return df


def create_df_from_gcs_file(blob_name:str, 
                            file_type = "xlsx",
                            excel_sheet_num = 0,
                            excel_skiprows_num = 0,
                            excel_header_nums = 0,
                            b_stand_cols = True,                            
                            project_number:str,
                            bucket_name:str, 
                            ) -> pd.DataFrame:
    """
    Args: 
        1. blob_name: rest of the path including filename in the bucket
        2. file_type: input file type, it can be "xls(x)", "csv", "json" etc
        3. excel_sheet_num: value for "sheet_name" for `read_excel` method
        4. excel_skiprows_num: value for "skiprows" for `read_excel` method
        5. excel_header_nums: value for "header" for `read_excel` method
        6. project_number: gcp project number        
        7. bucket_name: gcp bucket name

        
    Return: Pandas Dataframe
    
    Notes:
        - 2022-11-18: xls requires to install with %pip install xlrd in the command cell
        - As of 2022-08-17, it only supports xlsx but we can easily add ways to read
    other file types such as csv, json, etc
    """
    client = storage.Client(project = project_number)
    start_time = time.time()    
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    content = blob.download_as_string()
    
    if file_type in ("xlsx", "xls"):
        df = pd.read_excel(BytesIO(content), 
                           sheet_name = excel_sheet_num,
                           skiprows = excel_skiprows_num,
                           header = excel_header_nums)
        
    elif file_type == "csv":
        df = pd.read_csv(BytesIO(content))
        
    print(f" --- Read from GCS time: {np.round(time.time() - start_time, 3)} s") 
    print(f" --- Create a Pandas dataframe with {df.shape} shape from '{file_type}' file ")
    
    # standardize the columns
    if b_stand_cols and excel_header_nums == 0:
        standardize_columns(df)
        print(f" --- Standardize column names")
            
    return df


def create_duplicated_df(df:pd.DataFrame, 
                         col:str,
                         b_show = False) -> pd.DataFrame:
    """
    Args:
        1. df: Pandas dataframe
        2. col: Column name to see the duplicates, currently we're seeing only 1 feature, rather than multiple features
        3. b_show: Option to show all the column values with duplicates
        
    Return: Pandas dataframe
    """
    
    df_duplicated_arr = df[df.duplicated(col) == True][col].unique()
    print(f"Total number of duplicated {col} is {len(df_duplicated_arr)}")
    
    if b_show:
        print(df_duplicated_arr)

    df_duplicates = df[df[col].isin(df_duplicated_arr)==True].sort_values(by = col)

    return df_duplicates


def create_gb_aggs_df(df:pd.DataFrame,
                      gb_cols:List[str],
                      count_cols = None,
                      nunique_cols = None,
                      mean_cols = None,
                      mean_std_cols = None,
                      min_max_cols = None,
                      sum_cols = None,
                      merge_how = 'outer') -> pd.DataFrame:
    """
    Context: Sometimes we want to perform different aggregation for different columns and see them together.
    
    Args:
        1. df: Pandas dataframe
        2. gb_cols: Columns used for group by/aggregation
        3. count_cols: Columns used to obtain count 
        4. nunique_cols: Columns we're interested in knowing nunique
        5. mean_cols: Columns we're intrested in knowing average (sometimes having std is too crowded)
        6. mean_std_cols: Columns we're interested in knowing average/mean and std
        7. min_max: Columns we're interested in knowing min and max
        8. sum_cols: Columns we're interested in knowing sum
        
    Return: Pandas dataframe all the above merged
    """
    
    dfs = []
    agg_str = []
    if count_cols:
        df_gb_count = create_gb_df(df, gb_cols, count_cols, ['count'])
        dfs.append(df_gb_count)
        agg_str.append('count')
        
    if nunique_cols:
        df_gb_nunique = create_gb_df(df, gb_cols, nunique_cols, ['nunique'])
        dfs.append(df_gb_nunique)
        agg_str.append('nunique')
        
    if mean_cols:
        df_gb_nunique = create_gb_df(df, gb_cols, mean_cols, ['mean'])
        dfs.append(df_gb_nunique)
        agg_str.append('mean')        
        
    if mean_std_cols:
        df_gb_mean_std = create_gb_df(df, gb_cols, mean_std_cols, ['mean','std'])
        dfs.append(df_gb_mean_std)
        agg_str.append('mean-std')
        
    if min_max_cols:
        df_gb_min_max = create_gb_df(df, gb_cols, min_max_cols, ['min','max'])
        dfs.append(df_gb_min_max)
        agg_str.append('min-max')
        
    if sum_cols:
        df_gb_sum = create_gb_df(df, gb_cols, sum_cols, ['sum'])
        dfs.append(df_gb_sum)
        agg_str.append('sum')
        
    df_merged = create_merged_df(dfs, gb_cols, merge_how)
    print(f"{agg_str} were performed, and hence {len(dfs)} dfs were merged.")
    
    return df_merged


def create_gb_df(df:pd.DataFrame, 
                 groupby_cols:List[str], 
                 feature_cols:List[str], 
                 agg_cols = ['count','mean','median','sum'], 
                 dropna_opt = True) -> pd.DataFrame:
    """
    Args:
        1. df: Pandas dataframe
        2. groupby_cols: Columns to use for group by/aggregation
        3. feature_cols: Columns to have summary stat with aggregation, 
            Note that f should be numeric to be able to compute summary stat
        4. agg_cols: Aggregation/summary stat types = ['count', 'mean', 'std', 'median', 'sum', 'unique','nunique'] etc
        5. dropna_opt: we do not include null vals by default
        
    Return: Pandas dataframe
    """

    df_groupby = df.groupby(groupby_cols, as_index = False, dropna = dropna_opt)[feature_cols].agg(agg_cols)
    df_show = df_groupby
    df_show.columns = [f'{col[0]}_{col[1]}' for col in df_show.columns]

    df_return = df_show.reset_index()

    # Add precentage when we want to see the counts or nuniques
    if (agg_cols == ['count'] or agg_cols == ['nunique'] or agg_cols == ['mean']) and len(feature_cols) == 1:
        # compute percentages for "count" or "nunique"
        # 1. count percentage
        if agg_cols == ['count']:
            print(f'{feature_cols[0]}_{agg_cols[0]}')
            df_return['perc [%]'] = np.round(df_return[f'{feature_cols[0]}_{agg_cols[0]}']/
                                             df_return[f'{feature_cols[0]}_count'].sum()*100,2)
        # 2. unique percentage
        ## note that denominator is number of unique users, not the simple sum
        elif agg_cols ==['nunique']:
            print(f'{feature_cols[0]}_{agg_cols[0]}')
            df_return['perc [%]'] = np.round(df_return[f'{feature_cols[0]}_nunique']/
                                     df_return[f'{feature_cols[0]}_nunique'].sum()*100,2)
#                                     df[feature_cols[0]].nunique()*100,2)

        # return the results ordered by feature col
        df_return = df_return.sort_values(by=f'{feature_cols[0]}_{agg_cols[0]}', ascending = False)

    return df_return


def create_merged_df(dfs:List[pd.DataFrame], 
                     merge_on:List[str], 
                     merge_how ='outer') -> pd.DataFrame:
    """
    Args:
        1. dfs: List with pandas dataframes: e.g., [df1, df2, etc]
        2. merge_on: columns to merge/join, 
        3. merge_how: merging/joining type: 'left', 'inner', 'outer', etc
        
    Return: Pandas dataframe with all the individual dataframes merged
    """
    df_merged = reduce(lambda left,right: pd.merge(left, right, on = merge_on, how = merge_how), dfs)

    return df_merged


def create_missing_values_table(df:pd.DataFrame) -> pd.DataFrame:
    """
    Input: Pandas dataframe
    
    Return: Pandas dataframe along with print dataframe detail
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Counts of NaN', 1 : '% of NaN'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of NaN', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values(NaN).")
    
    return mis_val_table_ren_columns        


def create_ratio_col(df: pd.DataFrame, 
                     new_col: str, 
                     numerator_col: str, 
                     denominator_col: str, 
                     uncertainty_cols: tuple = None) -> None:
    """
    Args:
        1. df: Pandas dataframe
        2. new_col: column name for ratio
        3. numerator_col: numerator column name
        4. denominator_col: denominator column name
        5. uncertainty_cols: Tuple containing the uncertainty for the numerator and denominator respectively.
        
    Return: None, just create a new column in the existing pandas dataframe
    """
    
    ## below gets error with ```value is trying to be set on a copy of a slice from a DataFrame
    ## .Try using .loc[row_indexer,col_indexer] = value instead ```
    ratio = df[numerator_col] / df[denominator_col]
    df.loc[:, new_col] = ratio.where(np.isfinite(ratio), np.nan)
    
    #df[new_col] = df[numerator_col]/df[denominator_col]
    #df.loc[~np.isfinite(df[new_col]), new_col] = np.nan
    
    if uncertainty_cols:
        numerator_uncertainty_col, denominator_uncertainty_col = uncertainty_cols
        
    # approximate uncertainty (standard deviation) with sqrt(n) assuming counting is Poisson/Gaussian Distribution 
    # with mean n and success (failure) counting.
    # ref. Measurement of radiation detection, knoll. 3rd ed. p. 84
    else:
        numerator_uncertainty_col = f"{numerator_col}_uncertainty"
        denominator_uncertainty_col = f"{denominator_col}_uncertainty"
        df.loc[:, numerator_uncertainty_col] = np.sqrt(df[numerator_col])
        df.loc[:, denominator_uncertainty_col] = np.sqrt(df[denominator_col])
    
    uncertainty_ratio = np.sqrt((df[numerator_uncertainty_col]/df[numerator_col])**2 
                                + (df[denominator_uncertainty_col]/df[denominator_col])**2) * df[new_col]
    df.loc[:, f'{new_col}_uncertainty'] = uncertainty_ratio.where(np.isfinite(uncertainty_ratio), np.nan)
    
    if not uncertainty_cols:
        df.drop(columns = [numerator_uncertainty_col, denominator_uncertainty_col], inplace=True)

        
def create_time_diff_col(df:pd.DataFrame, 
                         time1_col:str, 
                         time2_col:str,
                         new_col_name = 'timediff',
                         time_astype = 'timedelta64[m]') -> None:
    
    """
    Args:
        1. df: Pandas dataframe
        2. time1_col: time column to compute the difference
        3. time2_col: time column to compute the difference
            * time2_col is substracted from time1_col
        4. new_col_name: column name of time difference col
        5. time_astype: Determines the unit of time difference, 
            default value is minute
            
    Return: None, just add new time diff column in the existing pandas dataframe
    """
    df['time1_tmp']= pd.to_datetime(df[time1_col])
    df['time2_tmp']= pd.to_datetime(df[time2_col])
 
  # create a column
    if time_astype =='timedelta64[D]':
     #        df[new_col_name] = (df['time1_tmp']-df['time2_tmp']).astype(time_astype) + 1
        df[new_col_name] = (df['time1_tmp']-df['time2_tmp']).astype(time_astype)
        
    else:
        df[new_col_name] = (df['time1_tmp']-df['time2_tmp']).astype(time_astype)
 
    
    df.drop(columns=['time1_tmp','time2_tmp'], inplace = True)
    
    
def draw_cat_vars_dist(df:pd.DataFrame, 
                       cat_cols:List[str],
                       fig_size = (12,8),
                       b_save = False,
                       savefig_path = "", 
                       savefig_name = 'cat_var_dist.png',
                       savefig_dpi = 300) -> None:
    
    """
    Context: It is useful to see the unique val distribution of categorical features for visualization
    
    Args: 
        1. df: Pandas dataframe
        2. cat_cols: Columns with limited unique vals which we want to see the distributions
        3. fig_size: Figure size in inches
        4. b_save: Option to save a fig
        5. savefig_path: Figure path, probably need to set as ds bucket folder
        6. savefig_name: Figure name
        7. savefig_dpi: DPI for the figure
          
    Return: None, optionally save a figure in png format with b_save == True
    """
    
    n_rows = int(len(cat_cols)/2) + len(cat_cols)%2

    fig, ax2d = plt.subplots(n_rows, 2, squeeze=False, figsize = fig_size)
    for i, ft in enumerate(cat_cols):
        _ = df.groupby(ft).size().plot(kind='barh', ax=ax2d[i//2, i%2], title=ft)

    ## viz options
#    plt.xticks(rotation=90,fontsize=20)
#    plt.xticks(fontsize=20)
#    plt.yticks(fontsize=20)
    plt.tight_layout()        

    if b_save:
        savefig_fullname = savefig_path + savefig_name
        print(savefig_fullname)
        fig.savefig(savefig_fullname, dpi = savefig_dpi)

        
def draw_cor_heatmap(df:pd.DataFrame, 
                     cor_cols:List[str], 
                     fig_size=(10,8), 
                     corr_method = 'pearson', cmap_option="PiYG",
                     tick_fontsize = 12, fmt_option = '.1g', 
                     b_save = False, 
                     save_name = 'default.png') -> None:
    """
    Args: 
        1. df: Pandas dataframe
        2. cor_cols: Columns to check the correlation
        3. fig_size: Default figure size
        4. corr_method: Method to use for correlation
        5. cmap_option: Correlation map color style
        6. tick_fontsize: Tick Fontsize
        7. fmt_option: Display number precison
        8. b_save: Option to save the correlation plot as a figure
        9. save_name: Figure name
        
    Return: None, options to save correlation plot as a figure
    """

    plt.style.use('default')
    plt.rcParams["font.family"] = "Times New Roman"
    corr = df[cor_cols].corr(method=corr_method)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=fig_size)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
#    plt.tight_layout()
    sns.heatmap(corr, mask=mask, cmap=cmap_option, vmax=.3, center=0, ax =ax,\
                annot = True, fmt=fmt_option, cbar_kws={"shrink": .5})
    if b_save:
        plt.savefig(save_name, dpi = 300)
        plt.close()
        
    # going back to default style for the following cells
    sns.set_style('darkgrid', {'font.family':'Times New Roman'})
    sns.set_palette("deep")
    plt.rcParams['figure.figsize'] = 6.0, 4.0
#    plt.rcParams['figure.dpi'] = 300        
        
        
def draw_histplots(df:pd.DataFrame, 
                   numeric_cols:List[str], 
                   fig_size = [20,4], nbins = 10, b_logy = False, 
                   b_save = False, 
                   savefig_path = "",
                   savefig_name = 'num_var_hist.png', savefig_dpi = 300) -> None:
    """
    Context: It is useful to see the 1D distribution of numeric features for visualization
    
    Args: 
        1. df: Pandas dataframe
        2. numeric_cols: Columns with numeric type which we want to see the distributions
        3. fig_size: Figure size in inches
        4. nbins: Number of bins for the histgoram
        5. b_logy: Option to show the dist plot with logy
        6. b_save: Option to save a fig
        7. savefig_path: Figure path, probably need to set as ds bucket folder
        8. savefig_name: Figure name
        9. savefig_dpi: DPI for the figure
          
    Return: None, optionally save figure in png format with b_save == True
    """
    
    fig, axes = plt.subplots(1, len(numeric_cols))
    fig.set_size_inches(fig_size)

    if len(numeric_cols) == 1:
        sns.histplot(df[numeric_cols[0]], bins = nbins)
        if b_logy == True:
            plt.yscale('log')
            
    else:
        for i in range(len(numeric_cols)):
            sns.histplot(df[numeric_cols[i]], bins = nbins, ax=axes[i])
            if b_logy == True:
#        axes[i].set_title(axes_set_titles[i])
               axes[i].set_yscale('log')
    
    ## viz option
    plt.tight_layout()
    
    if b_save:
        
        savefig_fullname = savefig_path + savefig_name
        print(savefig_fullname)
        fig.savefig(savefig_fullname,dpi = savefig_dpi)


def load_bq_table_from_df(df:pd.DataFrame,
                          ds_table_name:str,
                          project_name:str,
                         ) -> None:
    """
    Context: Alternative to upload pandas df save as csv and upload it to gcs bucket
    Reference: https://stackoverflow.com/questions/63200201/create-a-bigquery-table-from-pandas-dataframe-without-specifying-schema-explici
    
    Args: 
        1. df: Pandas dataframe to save
        2. ds_table_name: Dataset and Table Name in format "{dataset}.{table_name}"
        3. project_name: Project Name
          
    Return: None, Load/Create bq table
    """
    
    # Load client
    client = bigquery.Client(project= project_name)

    # Define table name, in format dataset.table_name
    table = ds_table_name

    # Load data to BQ
    job = client.load_table_from_dataframe(df, table)        
        
    
def print_col_uniques(df:pd.DataFrame, 
                      b_print_col_unique = False, 
                      n_unique_min = 0, 
                      n_unique_max = 100000) -> None:
    """
    Args: 
        1. df: Pandas dataframe
        2. b_print_col_unique: default options to print out unique values
        3. b_unique_min: threshold range min to print unique values of a col
        4. b_unique_max: threshold range max to print unique values of a col        
    
    Return: None, just print statement
    """
    df_cols = np.sort(df.columns.values)
    
    for i in range(len(df_cols)):
        if((df[df_cols[i]].nunique() <= n_unique_max) &
           (df[df_cols[i]].nunique() > n_unique_min)):
            if not b_print_col_unique:
                print(i, df_cols[i],df[df_cols[i]].nunique())
            # below line makes sense only n_unique_max is small enough
            elif b_print_col_unique:
                print(i, df_cols[i],df[df_cols[i]].nunique(),df[df_cols[i]].unique())
            elif(df[df_cols[i]].nunique() > n_unique_max):
                print(df_cols[i], 'has more than', n_unique_max, 'uniques!!', df[df_cols[i]].nunique())
                
                
def print_cor_val(df:pd.DataFrame,
                  cols_to_check:List[str],
                  cor_thr_min = 0.,
                  cor_thr_max = 1.,
                  cor_method = "pearson") -> None:
    """
    Args: 
        1. df: Pandas dataframe
        2. cols_to_check: Columns to check the correlation
        3. cor_thr_min: Lower bound to check the correlation
        4. cor_thr_max: Upper bound to check the correlation
        5. cor_method: Correlation Method, default is set as "Pearson"
    
    Return: None, just print the correlation between two columns whose absolute value
        is between cor_thr_min and cor_thr_max
    """     
    
    comb_num = combinations(cols_to_check, 2)
    for c in comb_num:
        # correlation value
        cor_val = df[c[0]].corr(df[c[1]], method=cor_method)
        # p-value for the correlation
        df_clean = df.dropna(subset=[c[0],c[1]])
        if cor_method == 'pearson':
            cor_val_p_val = scipy.stats.pearsonr(df_clean[c[0]],df_clean[c[1]])[1]
        elif cor_method == 'spearman':
            cor_val_p_val = scipy.stats.spearmanr(df_clean[c[0]],df_clean[c[1]])[1]
        elif cor_method == 'kendall':
            cor_val_p_val = scipy.stats.kendalltau(df_clean[c[0]],df_clean[c[1]])[1]
        else:
            raise ValueError(
                  "method must be either 'pearson', "
                  "'spearman', 'kendall', or a callable, "
                  f"'{method}' was supplied"
              )
        # if the threshold values are within the correct range
        if np.abs(cor_val)>cor_thr_min and np.abs(cor_val)<cor_thr_max:
            print(f'cor{c}', np.round(cor_val,3), ', p-val:', np.round(cor_val_p_val,6))    
            
    
def standardize_columns(df:pd.DataFrame) -> None:
    """
    Input: Pandas dataframe
    
    Return: None, just update input pandas dataframe
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.lstrip()
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.replace('\n', '_', regex=True)    
    df.columns = df.columns.str.replace(' ', '_', regex=True)
    df.columns = df.columns.str.replace('_-_', '_', regex=True)
    df.columns = df.columns.str.replace('\(', '', regex=True)
    df.columns = df.columns.str.replace('\)', '', regex=True)
    df.columns = df.columns.str.replace('%', 'perc', regex=True)
    df.columns = df.columns.str.replace('\?', '', regex=True)
    df.columns = df.columns.str.replace('!', '', regex=True)
    df.columns = df.columns.str.replace('\.', '', regex=True)
    df.columns = df.columns.str.replace('&', '', regex=True)
    df.columns = df.columns.str.replace('__', '_', regex=True)
    # BQ doesn't like - and prefer _
    df.columns = df.columns.str.replace('-', '_', regex=True)
    df.columns = df.columns.str.replace('/', '_', regex=True)    
    df.columns = df.columns.str.replace('#', 'num', regex=True)
    df.columns = df.columns.str.replace('1st', 'first', regex=True) # patsy doesn't like start with number
    
def standardize_column_values(df:pd.DataFrame,
                              st_cols:List[str]) -> None:
    """
    Args:
        1. df: Pandas dataframe
        2. st_cols: Columns to standardize vals
        
    Return: None, just update input dataframe column vals
    """
    for col in st_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.lstrip()
        df[col] = df[col].str.rstrip()
        df[col] = df[col].str.replace("'", "", regex=True)
        df[col] = df[col].str.replace("-", " ", regex=True)
        df[col] = df[col].str.replace(" ", "_", regex=True)
        df[col] = df[col].str.replace("/", "_", regex=True)
        df[col] = df[col].str.replace('(', '', regex=True)
        df[col] = df[col].str.replace(')', '', regex=True)
        df[col] = df[col].str.replace('_&_', '_and_', regex=True)
        print(f"--- Standardized {col}")

    
def upload_file_to_gcs(local_filename:str,
                       blob_name:str, 
                       bucket_name:str,
                      ) -> None:
    """
    Args:
        1. local_filename: file name to send to gcp bucket
        2. blob_name: rest of the path including filename in the bucket
        3. bucket_name
        
    Return: None
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)  # This defines the path where the file will be stored in the bucket
    blob.upload_from_filename(filename = local_filename)    
