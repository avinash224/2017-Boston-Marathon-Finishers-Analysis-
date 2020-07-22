#!/usr/bin/env python
# coding: utf-8

# # 2017 Boston Marathon Finishers Analysis 
# 

# For this analysis, I have taken the Boston marathon 2017 dataset from Kaggle. It contains the name, age, gender, country, city, and state (where available), times at 9 different stages of the race, expected time, finish time and pace, overall place, gender place, and division place.
# objective:
# 
# 1. Check the distribution of participants from different countries, states, and cities?
# 2. How is the age distribution for the different age groups in both men Male and Female Category?
# 3. What are the most popular first and last names of the people who participated?
# 4. Find people who finished the second half of the marathon earlier than the first half?
# 5. Check if there is any person who has a similar time for the first and the second half of the marathon?
# 6. What are the time records for every 5km? starting from 0k to 40k?
# 7. Check for the division winners?

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime 
import time


# In[93]:


import os 
os.getcwd()


# In[94]:


os.chdir("/Users/avinashtripathi/Downloads")


# In[95]:


df = pd.read_csv("marathon_results_2017.csv", index_col='Bib')


# In[96]:


df.head()


# Make all times, time deltas 

# In[97]:


df.iloc[:,9:21] = df.iloc[:,9:21].apply(pd.to_timedelta)


# In[131]:


df.head()


# In[99]:


df.describe()


# In[100]:


df.info()


# # let's see finishers by age 

# In[101]:


g = sns.countplot('Age', data= df, palette='coolwarm')
g.figure.set_size_inches(13,8)
g.set_title("Number of participants per age group" )


# # let's see finshers by age and gender

# In[102]:


g = sns.countplot('Age', data=df, palette = "coolwarm", hue='M/F')
g.figure.set_size_inches(16,8)
g.set_title("count of people based on different genders and age")


# In[103]:


#Box plot by age and gender 


# In[104]:


g = sns.boxplot("M/F", "Age", data=df)
g.figure.set_size_inches(13,8)
g.set_title("distibution of finish time per age group")


# In[105]:


g = sns.lmplot(x='Overall', y='Age', data=df, hue='M/F', size=10)


# # seperate the first and the last name 

# In[106]:


s = df['Name'].apply(lambda x: x.split(', '))
df['First Name'] = s.apply(lambda x: x[1])
df['Last Name'] = s.apply(lambda x: x[0])
df.drop('Name', axis=1, inplace=True)


# # Most popular country - finsihers (top 20)

# In[107]:


s = df.groupby('Country').count()['City'].sort_values(ascending=False).head(20)
g = sns.barplot(s.index, s, palette="bright")
g.figure.set_size_inches(13,8)
g.set_title("Most popular Country")


# since US and canada looks much greater than other countries, hence we drop them and analyse the rest of the countries

# In[108]:


s = df.groupby('Country').count()['City'].sort_values(ascending=False).head(22)[2:]
g = sns.barplot(s.index, s, palette='rainbow')
g.figure.set_size_inches(13,8)
g.set_title("Most popular Country (after US)")


# # Most popular city (top 20)

# In[109]:


s = df[df['City'].notnull()].groupby('City').count()['Country'].sort_values(ascending=False).head(20)
g = sns.barplot(s.index, s, palette='BuGn_r')
g.figure.set_size_inches(18,8)
g.set_title("Most popular City")


# # Most popular state (top 20)

# In[110]:


s = df[df['Country'] == 'USA'].groupby('State').count()['Country'].sort_values(ascending=False).head(20)
g = sns.barplot(s.index, s, palette="Oranges")
g.figure.set_size_inches(13,8)
g.set_title("Most popular State")


# In[111]:


s = df.groupby('First Name').count()['Last Name'].sort_values(ascending=False).head(20)
g = sns.barplot(s.index, s, palette='Greens')
g.figure.set_size_inches(13,8)
g.set_title("Most popular Name")


# In[112]:


s = df.groupby('Last Name').count()['First Name'].sort_values(ascending=False).head(20)
g = sns.barplot(s.index, s, palette='Blues')
g.figure.set_size_inches(13,8)
g.set_title("Most popular Last Name")


# # Distribution of official times by age
# Official Time is transformed into a float to chart times.

# In[129]:


g = sns.jointplot( x=df['Official Time'].apply(lambda x: x.total_seconds()/3600), y=df['Age'], stat_func=None, kind='hex', color="r", size=10)


# # Boxplot of finishing times by age

# In[128]:


g = sns.boxplot(df['Age'], df['Official Time'].apply(lambda x: x.total_seconds()/3600), palette="coolwarm")
g.figure.set_size_inches(13,8)
g.set_title("Distribution of finish times per Age group")


# # Analysis of first half vs second half 

# In[113]:


df["Half_2"] = df['Official Time'] - df['Half']


# calculate the difference between first half and second half

# In[114]:


df['2nd_Split'] = (df['Half_2']-df["Half"])


# In[115]:


df['2nd_Split']= df['2nd_Split'].apply(lambda x: x.total_seconds()/60)


# In[116]:


sns.lmplot(data=df, y='2nd_Split', x='Overall', size=10, markers='.')


# The distinct line of points over 150 and up is caused by the records that don't have a 
# half marathon registered (except for the very last that is over 250).

# find the extreme cases of positive and negative split 

# In[117]:


print(df[df['2nd_Split'] == df['2nd_Split'].max()][['5K', '10K','15K', '20K', '25K', '30K', '35K', '40K']])
print(df[df['2nd_Split'] == df['2nd_Split'].max()])


# In[118]:


print(df[df['2nd_Split'] == df['2nd_Split'].min()][['5K', '10K', '15K', '20K', '25K', '30K', '35K', '40K', 'Division']])
print(df[df['2nd_Split'] == df['2nd_Split'].min()])


# find how many negative splits - faster second half than first half. (particularly difficult in boston due to newton hills)

# In[119]:


df[df['2nd_Split'] < 0].sort_values(by='2nd_Split')


# In[120]:


(len(df[df['2nd_Split'] < 0].sort_values(by='2nd_Split'))/len(df))*100


# only 3.07% of the finishers have a negative split 

# In[121]:


df[df['2nd_Split']== 0].sort_values(by='2nd_Split')


# only 2 finishers have identical split times (very very rare )

# # Calculations of pace at different stages of the race.

# #Fastest pace recorded in every 5K segment
# since there are some missing times at some of the 5K markers, we calculate the difference of times between markers and then we find all the non negative ones. Then we find the min and print a nice fivek_pace.

# In[122]:


def fivek_pace(t):
    minute, second = divmod(t.seconds, 60)
    print('%02d:%02d' % (minute, second))


# In[126]:


fivek_pace((df['5K'][df['5K']!='0']/3.1).min())
fivek_pace(((df['10K'] -df['5K'])[(df['10K'] -df['5K'])>'0']/3.1).min())
fivek_pace(((df['15K'] -df['10K'])[(df['15K'] -df['10K'])>'0']/3.1).min())
fivek_pace(((df['20K'] -df['15K'])[(df['20K'] -df['15K'])>'0']/3.1).min())
fivek_pace(((df['25K'] -df['20K'])[(df['25K'] -df['20K'])>'0']/3.1).min())
fivek_pace(((df['30K'] -df['25K'])[(df['30K'] -df['25K'])>'0']/3.1).min())
fivek_pace(((df['35K'] -df['30K'])[(df['35K'] -df['30K'])>'0']/3.1).min())
fivek_pace(((df['40K'] -df['35K'])[(df['40K'] -df['35K'])>'0']/3.1).min())


# # Division winners 

# In[127]:


win = df[df['Division'] == 1].fillna('')
win


# In[ ]:




