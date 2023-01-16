# importing libraries
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from pycaret import classification
import functions as func
import constants as const

# importing in_time file into dataframe
time_df = func.import_ops(file_name='in_time', is_time=True, val_name='in_worktime')

# importing out_time file into dataframe
out_time_df = func.import_ops(file_name='out_time', is_time=True, val_name='out_worktime')
out_time_df.head()

# merging time dataframes for work-time stats computation
time_df = time_df.merge(out_time_df)
print(time_df.shape)
time_df.head()

# creating workhrs based on delta between out and in worktimes
time_df['workhrs'] = time_df['out_worktime'] - time_df['in_worktime']
time_df['workhrs'] = time_df['workhrs'].dt.total_seconds()/(60*60)
time_df.head()

# creating stat cols to aggregate workhrs for each employee
time_stats_df = time_df.groupby('EmployeeID').agg({'workhrs':['mean','median','std','max']})

# creating proxy for leave days based on count of NaN workhrs
time_df.fillna(0, inplace=True)
zero_cnt_df = time_df[time_df['workhrs']==0].groupby('EmployeeID').agg({'workhrs':['count']})
time_stats_df = time_stats_df.join(zero_cnt_df)

# final time feature engineered dataset
time_stats_df.columns = time_stats_df.columns.droplevel()
time_stats_df.reset_index(inplace=True)
time_stats_df.columns=['EmployeeID','time_mean','time_median','time_stddev','time_max','time_leave']
print(time_stats_df.shape)
time_stats_df.head()

# importing general_data file into dataframe
general_df = func.import_ops(file_name='general_data')

# importing employee_survey_data file into dataframe
emp_survey_df = func.import_ops(file_name='employee_survey_data')

# importing manager_survey_data file into dataframe
mgr_survey_df = func.import_ops(file_name='manager_survey_data')

# creating list of dataframes to merge
df_list = [emp_survey_df, general_df, mgr_survey_df, time_stats_df]

for df in df_list:
    df.set_index('EmployeeID', inplace=True)

# merging all individual dataframes into one master dataframe
merged_df = df_list[0].join(df_list[1:]).reset_index()
print(merged_df.shape)
merged_df.head()

# creating feature column to represent under and over work comparing actual worktime to standard
merged_df['time_actual_vs_std'] = round((merged_df['time_mean'] - merged_df['StandardHours']),3)
print(merged_df.shape)
merged_df.head()

# excel export of merged master dataframe
merged_df.to_excel(const.INPUT_PATH + 'merged_df.xlsx', index=False)

# importing data_dictionary to confirm availibility of all variables in merged master dataframe
data_dict_df = pd.read_excel(const.INPUT_PATH + 'data_dictionary.xlsx', usecols='A')
data_dict_df.dropna(inplace=True)
print(data_dict_df.shape)
data_dict_df.head()

# identifying mismatched columns between master dataframe and data dictionary
# Employee num is just reworded, while Relationshipsatisfaction is completely missing
list(set(data_dict_df['Variable'].values).difference(merged_df.columns))

# checking columns with missing values
missing_values_df = pd.DataFrame(merged_df.isna().sum()).reset_index()
missing_values_df.columns = ['col', 'num_missing']
missing_values_df['pct_missing'] = missing_values_df['num_missing']*100/len(merged_df)
missing_values_df[missing_values_df['num_missing']>0].round(3)

# identifying columns with single unique values
merged_df_nonvariance = merged_df.nunique()[merged_df.nunique()==1]
non_variance_cols_list = merged_df_nonvariance.index
print(f'non_variance_cols:\n{merged_df_nonvariance}')

# applying conditional logic based imputation to feature with nulls
print(f'Num missing in NumCompaniesWorked column before conditional logic: {merged_df["NumCompaniesWorked"].isna().sum()}')
merged_df['NumCompaniesWorked'] = merged_df.apply(func.num_companies, axis=1)
print(f'Num missing in NumCompaniesWorked column after conditional logic: {merged_df["NumCompaniesWorked"].isna().sum()}')

# applying conditional logic based imputation to feature with nulls
print(f'Num missing in TotalWorkingYears column before conditional logic: {merged_df["TotalWorkingYears"].isna().sum()}')
merged_df['TotalWorkingYears'] = merged_df.apply(func.working_yrs, axis=1)
print(f'Num missing in TotalWorkingYears column after conditional logic: {merged_df["TotalWorkingYears"].isna().sum()}')

# dropping columns without variance and nulls from merged dataframe
merged_df.drop(columns=non_variance_cols_list, inplace=True)
merged_df.dropna(inplace=True)
print(merged_df.shape)
merged_df.head()

# excel export of merged master dataframe - data processing completed - to model
merged_df.to_excel(const.OUTPUT_PATH + 'merged_df_FeatureEngineered.xlsx', index=False)

# checking for imbalance in response variable- attrition
attrition_count_df = merged_df.groupby('Attrition').agg({'EmployeeID':'count'}).reset_index()

fig = px.pie(attrition_count_df, values='EmployeeID', names='Attrition',
            width=500, height=400, title='fig 1. attrition data distribution')
fig.update_layout(legend={'title': 'Attrition'}, title={'x': 0.5, 'y': 0.8})
fig.show()

# train-test split with stratify applied to approximate the same % of samples of each target class
train, test = train_test_split(merged_df, test_size=0.10, random_state=0, 
                               stratify=merged_df['Attrition'])

print(f'train: {train.shape}')
print(f'test: {test.shape}')
test.head()

# confirming test had similar sampling as parent dataframe
attrition_count_test_df = test.groupby('Attrition').agg({'EmployeeID':'count'}).reset_index()

fig = px.pie(attrition_count_test_df, values='EmployeeID', names='Attrition',
            width=500, height=400, title='test attrition data distribution: post stratified split')
fig.update_layout(legend={'title': 'Attrition'}, title={'x': 0.5, 'y': 0.8})
fig.show()

# defining ordinal features with intrinsic natural order
ordinal_feature_list = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                       'JobSatisfaction', 'PerformanceRating', 'WorkLifeBalance']

train[ordinal_feature_list] = train[ordinal_feature_list].astype(int).astype(str)

# Sorting the ordinal features in correct order
ordinal_features_dict = {
    col: sorted([x for x in train[col].unique() if x==x]) 
    for col in ordinal_feature_list
}
print(ordinal_features_dict)

train.drop(columns=['EmployeeID'], inplace=True)
test.drop(columns=['EmployeeID'], inplace=True)
print(len(train.columns))
train.columns

# save a checkpoint
train.to_pickle(const.OUTPUT_PATH + 'train.pkl')
test.to_pickle(const.OUTPUT_PATH + 'test.pkl')

best_model = func.model_train(train, ordinal_features_dict, experiment_name='baseline_model')

et = classification.create_model('et')

# train confusion matrix
classification.plot_model(best_model, plot='confusion_matrix', use_train_data=True)