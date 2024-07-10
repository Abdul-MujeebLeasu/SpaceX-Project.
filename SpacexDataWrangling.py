import pandas as pd
import numpy as np

# I use this to load in the previous dataset saved
df=pd.read_csv("dataset_cleaned.csv")
df.head(10)

df.isnull().sum()/len(df)*100
# Rechecking the data types to check if they are correct
print(df.dtypes)

# Applies value_counts to count the data values for each unique Launch pad in the dataframe
df['LaunchSite'].value_counts()

# Counts hte values up for each unique orbital distance.
print(df['Orbit'].value_counts())

# landing_outcomes = values on Outcome column
landing_outcomes = df['Outcome'].value_counts()
# Iterates through the unique landing outcomes values and enumerates them with keys
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)

# Sets the unsuccessful landings into a set called bad_outcomes. This is used for further analysis
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

# Iterates through the dataframe outcome values. If the outcome is in the bad_outcomes set it appends the list with the value 0
# Else it appends the class list with the value 1
landing_class = []
for i in (df['Outcome'].values):
    if i in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)

print(landing_class)

# Adds the class column into the dataframe
df['Class']=landing_class
df[['Class']].head(8)

print(df["Class"].mean()) # Checks the mean values in the dataset
print(df['Class'].value_counts())
df.to_csv('dataset_cleaned2.csv')