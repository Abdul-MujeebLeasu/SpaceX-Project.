import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("dataset_cleaned2.csv")
# Quick Checkup on the data
print(df.head(5))

# Plotting PayloadMass vs FlightNumber to display any potential relationship between the variables
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()

# Plotting FlightNumber vs LaunchSite to display any potential relationship between the variables
sns.catplot(x='FlightNumber', y='LaunchSite', hue='Class', data=df, aspect=1)
plt.xlabel('Flight Number')
plt.ylabel('Launch Site')
plt.title('Flight Number vs Launch Site')
plt.show()

# Plotting PayloadMass vs LaunchSite to display any potential relationship between the variables
sns.scatterplot(data=df, x='PayloadMass', y='LaunchSite', hue='Class', legend=False)
plt.xlabel('Payload Mass (Kg)')
plt.ylabel('Launch Site')
plt.title('Relationship between PayloadMass and Launchsite')

# Visualise the success per orbital distance. There isn't any particular pattern
data = df.groupby('Orbit')['Class'].mean()
data
sns.barplot(data, color='red', )

# Plotting FlightNumber vs Orbit to display any potential relationship between the variables
sns.scatterplot(data=df, x='FlightNumber', y='Orbit', hue='Class', legend=False, marker='x')

# Plotting Payload vs Orbit to display any potential relationship between the variable
sns.scatterplot(data=df, x='PayloadMass', y='Orbit', marker='*', legend=False, hue='Class', s=40)
plt.title('PayloadMass against The Orbit Distance')

# Extract all the years from date. This is used to plot a line graph with the independant variable being years. 
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year


# Group the data using the function then plot on a line graph. Clear Upwards trend in successrate of launches
data = df.groupby(Extract_year(df['Date']))['Class'].mean()
sns.lineplot(data)
plt.title('Yearly Success Rate Trends')
plt.show()

#Specific columns
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

# get_dummies will be used to turn all our categorical data into numberical. This helps as we can now using for further analysis
# Such as Machine Learning etc.
features_one_hot = pd.get_dummies(features, prefix='OneHotEncoder')
features_one_hot.head()
features_one_hot.shape
features_one_hot.replace(False, 0, inplace=True)
features_one_hot.replace(True, 1, inplace=True)

# Turning all our data into float values as now it only contains numerical data
features_one_hot.astype('float64')

# Resaving it into another data set for further analysis
features_one_hot.to_csv('dataset_cleaned3.csv', index=False)