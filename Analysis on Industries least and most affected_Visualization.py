#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import dependencies.
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from datetime import datetime

import seaborn as sns
import matplotlib.ticker as ticker


# # Three Industries Most Affected

# #### Manufacturing

# In[2]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\all_emp_manufacturing.csv')
df_emp_man = pd.read_csv(file_path)

# Display DataFrame.
df_emp_man.head(10)


# #### Education and Health Services

# In[5]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\Cleaned Data\\all_emp_edu.csv')
df_emp_edu = pd.read_csv(file_path)

# Display DataFrame.
df_emp_edu.head(10)


# #### Hospitality

# In[6]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\all_emp_hospitality.csv')
df_emp_hos = pd.read_csv(file_path)

# Display DataFrame.
df_emp_hos.head(10)


# #### Merge Datasets (3 Industries Most Affected)

# In[8]:


# Merge Datasets.
three_industries_alpha = pd.merge(pd.merge(df_emp_man,df_emp_edu,on='observation_date'),df_emp_hos,on='observation_date')

# Display merged DataFrame.
three_industries_alpha.head(10)


# In[9]:


# Rename columns.
three_industries_alpha = three_industries_alpha.rename(columns={"MANEMP": "Manufacturing", "USEHS": "Education and Health Services", "USLAH": "Hospitality"})

# Display DataFrame.
three_industries_alpha.head(10)


# In[18]:


# Use the graph style fivethirtyeight.
style.use('fivethirtyeight')

ax = three_industries_alpha.plot(figsize=(24, 14), x="observation_date", y=["Manufacturing", "Education and Health Services", "Hospitality"])
ax.set_title("Current Employment Statistics (Establishment Survey)")
ax.set_xlabel("Year")
ax.set_ylabel("Thousands of Persons (Seasonally Adjusted)")

# Setting the number of ticks.
plt.locator_params(axis='both', nbins=12)

# Save figure. 
plt.savefig("C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Images", dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)


# # Three Industries Least Affected

# #### Finance

# In[12]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\all_emp_fin.csv')
df_emp_fin = pd.read_csv(file_path)

# Display DataFrame.
df_emp_fin.head(10)


# #### IT

# In[13]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\all_emp_info.csv')
df_emp_info = pd.read_csv(file_path)

# Display DataFrame.
df_emp_info.head(10)


# #### Professional & Business

# In[14]:


# Load the data.
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\all_emp_prof_bus.csv')
df_emp_prof_bus = pd.read_csv(file_path)

# Display DataFrame.
df_emp_prof_bus.head(10)


# #### Merge Datasets (3 Industries Least Affected)

# In[15]:


# Merge Datasets.
three_industries_beta = pd.merge(pd.merge(df_emp_fin,df_emp_info,on='observation_date'),df_emp_prof_bus,on='observation_date')
three_industries_beta.head(10)


# In[16]:


# Rename columns.
three_industries_beta = three_industries_beta.rename(columns={"USFIRE": "Finance", "USINFO": "IT", "USPBS": "Professional & Business"})

# Display DataFrame. 
three_industries_beta.head(10)


# In[19]:


# Use the graph style fivethirtyeight.
style.use('fivethirtyeight')

ax = three_industries_beta.plot(figsize=(24, 14), x="observation_date", y=["Finance", "IT", "Professional & Business"])
ax.set_title("Current Employment Statistics (Establishment Survey)")
ax.set_xlabel("Year")
ax.set_ylabel("Thousands of Persons (Seasonally Adjusted)")

# Setting the number of ticks.
plt.locator_params(axis='both', nbins=12)

# Save figure. 
plt.savefig("C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Images", dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None)

