# Libraries
import pandas as pd
from scipy import stats
from pycaret import classification

def import_ops(file_name, is_time=False, val_name=''):
    # defining helper function to perform repeated import+related processing operations
    df =  pd.read_csv(f'C:\\Users\\kunal\\OneDrive - Washington State University (email.wsu.edu)\\Desktop\\Data Science\\data\\inputs\\{file_name}.csv')
    print(f'original: {df.shape}')
    
    if is_time:
        df.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
        df = pd.melt(df, id_vars=['EmployeeID'], var_name='workdate', value_name=val_name)
        df[val_name] = pd.to_datetime(df[val_name])
        print(f'transformed: {df.shape}')
    
    print(f'num_unique_employees: {df["EmployeeID"].nunique()}')
    print(df.head())
    return df

def num_companies(row):
    # defining function to impute nulls in companies worked by referencing total working years and years at current company
    if row['NumCompaniesWorked'] != row['NumCompaniesWorked']:
        # executes if row['NumCompaniesWorked'] is NaN
        if row['TotalWorkingYears'] == row['YearsAtCompany']:
            return 0
        
    return row['NumCompaniesWorked'] 

def working_yrs(row):
    # defining function to impute nulls in total working years by referencing number of companies worked and years at current company
    if row['TotalWorkingYears'] != row['TotalWorkingYears']:
        if row['NumCompaniesWorked']==0:
            return row['YearsAtCompany']
        
    return row['TotalWorkingYears']

def cont_stat_sig_test(df,col):
    # defining function for normality + statistical significance check
    # Step 1: normality test
    stat, p_value = stats.normaltest(df[col].values)
    
    # Step 2: filter series    
    active = df[df['Attrition']=='No'][col].values
    resign = df[df['Attrition']=='Yes'][col].values
    
    # Step 3: determining appropriate statistical significance test based on normality
    if p_value>0.01:
        distribution = 'normal distribution'
        test = 'Anova test'
        stat, p_value = stats.f_oneway(active, resign)
    
    else:
        distribution = 'non-normal distribution'
        test = 'KS test'
        stat, p_value = stats.ks_2samp(active, resign)
    
    # Step 4: conclude on statistical significance
    if p_value>0.05:
        sig = 'not statistically significant'
    else:
        sig = 'statistically significant'

    print(f'{col} has {distribution}: running {test}. result: {sig}')
    return p_value

def model_train(train, ordinal_features_dict, experiment_name, save=True):
    # initializing pycaret setup 2nd time post feature selection
    clf = classification.setup(data=train, target='Attrition', 
                               ordinal_features=ordinal_features_dict,
                               session_id=123, log_experiment=True,
                               experiment_name=experiment_name)
    
    # Compare all models
    best_model = classification.compare_models(sort='F1', turbo=False)
    
    # Save the best model
    if save:
        classification.save_model(best_model, f'C:\\Users\\kunal\\OneDrive - Washington State University (email.wsu.edu)\\Desktop\\Data Science\\models\\{experiment_name}')
        
    return best_model