#!/usr/bin/env python
# coding: utf-8

# In[86]:


#Import dependencies
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import folium


# In[21]:


# Load the data
file_path = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\Cleaned_Attrition.csv')
df = pd.read_csv(file_path)


# In[22]:


# Preview data
df.head(10)


# In[23]:


#Check data types
df.dtypes


# In[24]:


# Calculate statistical data
df.describe()


# In[25]:


#Codify Attrition (yes or no)
df['Attrition'] = df['Attrition'].factorize(['No','Yes'])[0]
df.head()


# In[48]:


plt.figure(figsize = (15, 7))
plt.style.use('seaborn-v0_8-white')
plt.subplot(331)
label = LabelEncoder()
df['EducationField'] = label.fit_transform(df['EducationField'])
sns.countplot(data=df, x=df['EducationField'],)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(332)
sns.countplot(data=df, x=df['JobInvolvement'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(333)
sns.countplot(data=df, x=df.JobSatisfaction)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(334)
sns.countplot(data=df, x=df.EnvironmentSatisfaction)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(335)
sns.countplot(data=df,x=df.RelationshipSatisfaction)
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(336)
sns.countplot(data=df,x=df.WorkLifeBalance)
fig = plt.gcf()
fig.set_size_inches(10,10)


# In[11]:


# Attrition Pie Chart by Gender
df.groupby('Gender')['Attrition'].agg(np.mean).sort_values(ascending=False)


# In[12]:


#Males are more likely to experience attrition in comparison to females
exp_val=[0.170068,0.147959]
exp_label=['Male','Female']
palette_color = sns.color_palette('muted')
plt.title('Attrition Based on Gender', fontsize=15)
plt.pie(exp_val,labels=exp_label,radius=1,colors=palette_color,autopct='%0.2f%%');


# In[13]:


# Attrition Pie Chart by Overtime
df.groupby('OverTime')['Attrition'].agg(np.mean).sort_values(ascending=False)


# In[14]:


#Overtime contributes to attrition
exp_val=[0.305288,0.104364]
exp_label=['Yes','No']
palette_color = sns.color_palette('muted')
plt.title('Attrition Based on Over Time', fontsize=15)
plt.pie(exp_val,labels=exp_label,radius=1,colors=palette_color,autopct='%0.2f%%');


# In[15]:


# Attrition Pie Chart by Travel Frequency
df.groupby('BusinessTravel')['Attrition'].agg(np.mean).sort_values(ascending=False)


# In[16]:


#Travel Frequency contributes to attrition
exp_val=[0.249097,0.149569,.08]
exp_label=['Travel Frequently','Travel Rarely','No Travel']
plt.title('Attrition Based on Travel Frequency', fontsize=15)
palette_color = sns.color_palette('muted')
plt.pie(exp_val,labels=exp_label,radius=1,colors=palette_color,autopct='%0.2f%%');


# In[17]:


# Attrition Pie Chart by WorkLifeBalance
df.groupby('WorkLifeBalance')['Attrition'].agg(np.mean).sort_values(ascending=False)


# In[18]:


#Poor Work Life Balance contributes to attrition
exp_val=[0.3125,0.168605,0.142217,0.176471]
exp_label=['Bad','Good','Better', 'Best']
palette_color = sns.color_palette('muted')
plt.title('Attrition Based on Work/Life Balance', fontsize=15)
plt.pie(exp_val,labels=exp_label,radius=1,colors=palette_color,autopct='%0.2f%%');


# In[194]:


#Load the data
file_path2 = Path('C:\\Users\\15714\\Downloads\\Great_Resignation_Analysis-main\\Great_Resignation_Analysis-main\\Cleaned Data\\cleaned_geographic_data.csv')
gd = pd.read_csv(file_path2)
gd.shape

gd.head()


# In[195]:


#setting the map
url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
state_geo = f"{url}/us-states.json"

m = folium.Map(location = [40, -95],width = 950, 
                   height = 550,
                   zoom_start = 4,tiles = 'openstreetmap' )
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.LayerControl().add_to(m)

m


# In[196]:


#data refining
gd=gd.dropna(subset=['Longitude'])

gd=gd.dropna(subset=['Latitude'])


# In[197]:


#overall plotting of the data
for lat, lng in zip(gd['Latitude'], gd['Longitude']):    
    station = folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            color='red',
            fill=True,
            fill_color='yellow',
            fill_opacity=0.5)   
    # add the circle marker to the map
    station.add_to(m)
    
m


# In[198]:


t1=gd.loc[(gd['Description']=="Government and government enterprises State and local Local government") &(gd["2016"] > 50000)] 
t1.head()


# In[199]:


m = folium.Map(location = [40, -95],width = 950, 
                   height = 550,
                   zoom_start = 4,tiles = 'openstreetmap' )
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.LayerControl().add_to(m)

for lat, lng in zip(t1['Latitude'], t1['Longitude']):    
    station = folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            color='red',
            fill=True,
            fill_color='yellow',
            fill_opacity=0.5)   
    # add the circle marker to the map
    station.add_to(m)
    
m


# In[200]:


t2=gd.loc[(gd['Description']=="Nonfarm employment Private nonfarm employment Health care and social assistance") &(gd["2016"] > 50000) & (gd["2017"] > 50000) & (gd["2018"] > 50000) & (gd["2019"] > 50000) & (gd["2020"] > 50000) ] 
t2.head()


# In[201]:


m = folium.Map(location = [40, -95],width = 950, 
                   height = 550,
                   zoom_start = 4,tiles = 'openstreetmap' )
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.LayerControl().add_to(m)

for lat, lng in zip(t2['Latitude'], t2['Longitude']):    
    station = folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            color='red',
            fill=True,
            fill_color='yellow',
            fill_opacity=0.5)   
    # add the circle marker to the map
    station.add_to(m)
    
m

