#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[2]:


covid_df =pd.read_csv('D:\Dataset\Covid/covid_19_india.csv')


# In[3]:


covid_df.head(10)


# In[4]:


covid_df.info()


# In[5]:


vaccine_df = pd.read_csv('D:\Dataset\Covid/covid_vaccine_statewise_India.csv')


# In[6]:


vaccine_df.head(7)


# In[7]:


covid_df.drop(["Sno", "Time", "ConfirmedIndianNational", "ConfirmedForeignNational"], inplace = True, axis = 1)


# In[8]:


covid_df.head()


# In[9]:


covid_df.head()


# In[10]:


covid_df['Active_cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# In[11]:


statewise = pd.pivot_table(covid_df, values = ["Confirmed", "Deaths", "Cured"], index = "State/UnionTerritory", aggfunc = max)

statewise["Recovery Rate"] = statewise["Cured"]*100/statewise["Confirmed"]

statewise["Mortality Rate"] = statewise["Deaths"]*100/statewise["Confirmed"]

statewise = statewise.sort_values(by = "Confirmed", ascending = False)

statewise.style.background_gradient(cmap = "rainbow")


# In[12]:


#  Top 10 active cases states

top_10_active_cases = covid_df.groupby(by = 'State/UnionTerritory').max()[['Active_cases', 'Date']].sort_values(by = ['Active_cases'], ascending = False).reset_index()

fig = plt.figure(figsize = (16, 9))

plt.title("Top 10 states with most active cases in India", size = 25)

ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = "Active_cases", x = "State/UnionTerritory", linewidth = 2, edgecolor = 'black')

plt.xlabel("states")
plt.ylabel("Total active cases")
plt.show()


# In[13]:


# Top states with Highest deaths

top_10_deaths = covid_df.groupby(by='State/UnionTerritory').max()[['Deaths', 'Date']].sort_values(by='Deaths', ascending=False).reset_index()

fig = plt.figure(figsize = (17,5))

plt.title("Top 10 states with most Deaths", size = 25)

ax = sns.barplot(data = top_10_deaths.iloc[:12], y = "Deaths", x = "State/UnionTerritory", linewidth = 2, edgecolor = "black")

plt.xlabel("states")
plt.ylabel("Total Death cases")
plt.show()


# In[14]:


fig = plt.figure(figsize=(12, 6))

ax = sns.lineplot(data=covid_df[covid_df['State/UnionTerritory'].isin(['Maharashtra', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Uttar Pradesh'])], x='Date', y='Active_cases', hue='State/UnionTerritory')

ax.set_title("Top 5 Affected states in India", size=15)


# In[15]:


vaccine_df.head()


# In[16]:


vaccine_df.rename(columns = {'updated on' : 'Vaccine_Date'}, inplace = True)


# In[17]:


vaccine_df.head(10)


# In[18]:


vaccination = vaccine_df.drop(columns = ['Sputnik V (Doses Administered)', 'AEFI', '18-44 Years (Doses Administered)', '45-60 Years (Doses Administered)', '60+ Years (Doses Administered)'], axis=1)


# In[19]:


vaccination.head()


# In[20]:


vaccine = vaccine_df[vaccine_df.State!= 'India']
vaccine


# In[21]:


vaccine.rename(columns = {"Total Individuals Vaccinated": 'Total'}, inplace = True)
vaccine.head()


# In[22]:


# most vaccinated States

max_vac = vaccine.groupby('State')['Total'].sum().to_frame('Total')
max_vac = max_vac.sort_values('Total', ascending = False)[:5]
max_vac


# In[23]:


fig = plt.figure(figsize = (9,4))

plt.title("Top 5 Vaccinated States In India", size = 25)

x = sns.barplot(data = max_vac.iloc[:10], y = max_vac.Total, x = max_vac.index, linewidth = 2, edgecolor = 'black')

plt.xlabel("states")
plt.ylabel("Vaccination")
plt.show()


# In[ ]:




