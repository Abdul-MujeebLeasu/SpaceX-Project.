import sys

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd

def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    

# Presetting some dictioneries that i will use for analysis
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

response = requests.get(static_url)
content = response.text

soup = BeautifulSoup(content, 'html5lib')

print(soup.title)
# To get the title of the Webpage

html_tables = soup.find_all('table')

first_launch_table = html_tables[2]
print(first_launch_table.find_all('th')[0:10])
# Obtains header values

column_names = []


first_launch_table.find_all('th')

for row in first_launch_table.find_all('th'):
    header = extract_column_from_header(row)
    if header is not None and len(header) > 0:
        column_names.append(header)
# Using my pre defined dictionery to iterate through all header values
# Using a conditional statement to remove the headers with no values


print(column_names)
#Prints the list of column names


launch_dict= dict.fromkeys(column_names)
# Turns all the column names into dictionery keys

launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column that won't be used in further analysis
del launch_dict['Date and time ( )']

# Empty each dict value
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns needed
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]


print(launch_dict)
# Printing the launch_dict dictionery to produce the structure.

extracted_row = 0
#Extract each table using iteration and then putting the values into there respective dictioneries.
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        #checks to see if first table heading is as number corresponding to launch a number 
        if rows.th:
            if rows.th.string:
                flight_number=rows.th.string.strip()
                flag=flight_number.isdigit()
        else:
            flag=False

        row=rows.find_all('td')
        #if it is number save cells in a dictonary 
        if flag:
            extracted_row += 1

            launch_dict['Flight No.'].extend([flight_number])
            datatimelist=date_time(row[0])
        
            date = datatimelist[0].strip(',')
            launch_dict['Date'].extend([date])
            #print(date)
            
            time = datatimelist[1]
            launch_dict['Time'].extend([time])
            #print(time)
              
            bv=booster_version(row[1])
            if not(bv):
                bv=row[1].a.string
            launch_dict['Version Booster'].extend([bv])
            print(bv)
            
            launch_site = row[2].a.string
            launch_dict['Launch site'].extend([launch_site])
            #print(launch_site)
            
            payload = row[3].a.string
            launch_dict['Payload'].extend([payload])
            #print(payload)
            
            payload_mass = get_mass(row[4])
            launch_dict['Payload mass'].extend([payload_mass])
            #print(payload)

            orbit = row[5].a.string
            launch_dict['Orbit'].extend([orbit])
            #print(orbit)
            
            customer = row[6]
            launch_dict['Customer'].extend([customer])
            #print(customer)
            
            launch_outcome = list(row[7].strings)[0]
            launch_dict['Launch outcome'].extend([launch_outcome])
            #print(launch_outcome)
            
            booster_landing = landing_status(row[8])
            launch_dict['Booster landing'].extend([booster_landing])
            #print(booster_landing)
            

