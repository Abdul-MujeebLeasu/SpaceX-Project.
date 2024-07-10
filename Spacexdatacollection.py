import requests
import pandas as pd
import numpy as np
import datetime

# Supressed the terminal from displaying all the columns and indexes.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def getBoosterVersion(data):
    for x in data['rocket']:
       if x:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
        

def getLaunchSite(data):
    for x in data['launchpad']:
       if x:
         response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
         Longitude.append(response['longitude'])
         Latitude.append(response['latitude'])
         LaunchSite.append(response['name'])


def getPayloadData(data):
    for load in data['payloads']:
       if load:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])



def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

# Presetting some functions that i will use for analysis

spacex_url="https://api.spacexdata.com/v4/launches/past"
# Retrieving the url

response = requests.get(spacex_url)
# Gets the url request

static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
# Retrieving the JSON file

df = pd.json_normalize(response.json())
# Using pandas to extract the file

# Taking a subset of the dataframe to keep only the features that are required
df = df[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# Removing extra unnecessary rows
df = df[df['cores'].map(len)==1]
df = df[df['payloads'].map(len)==1]

# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
df['cores'] = df['cores'].map(lambda x : x[0])
df['payloads'] = df['payloads'].map(lambda x : x[0])

# Extracting the date into a proper datetime format
df['date'] = pd.to_datetime(df['date_utc']).dt.date

# Restricting launch dates
data = df[df['date'] <= datetime.date(2020, 11, 13)]

# Creating lists for each of the columns by creating Global Variables
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

# Using the predefined functions to obtain data from there respective JSON file
getBoosterVersion(df)
getLaunchSite(df)
getPayloadData(df)
getCoreData(df)

# Combining the columns in the dataset into a dictionery
launch_dict = {'FlightNumber': list(df['flight_number']),
'Date': list(df['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}

# Turning the dictionary into a dataframe
launch_data = pd.DataFrame(launch_dict)

# Removing all the values with Falcon 1 as we only want Falcon9 values in our further analysis
mask = launch_data['BoosterVersion'] == 'Falcon 1'
data_falconno1 = launch_data[~mask]

# Reset the flight number column by adding a list of values between 1 and the length of the dataframe
data_falconno1.loc[:,'FlightNumber'] = list(range(1, data_falconno1.shape[0]+1))

# Calculates the mean of the Payload Mass column in the data frame
mean = data_falconno1['PayloadMass'].mean()

# Replacing the null values with the new 
data_falconno1['PayloadMass'].replace(np.NaN, mean, inplace=True)
# Recheck if the new 'NULL' have been replaced
print(data_falconno1.isnull().sum())

# Export to a .CSV (Comma Seperated Values) spreadsheet 
data_falconno1.to_csv('dataset_cleaned.csv', index=False)