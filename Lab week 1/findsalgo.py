import csv 
import pandas as pd
import numpy as np
# csv file name 
filename = "data.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
  
# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
    # extracting field names through first row 
    fields = next(csvreader) 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
  
    # get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num-1)) 

# printing the field names 
print('Field names are:' + ', '.join(field for field in fields)) 
  
#  printing first 5 rows 
print('\nThe rows are:\n') 
for row in rows[:4]: 
    # parsing each column of a row 
    for col in row: 
        print("%10s"%col, end =" ")
    print('\n')
    Total no. of rows: 4
Field names are:Weather, Temprature, Humidity, Goes

The rows are:

     Sunny       Warm       Mild        Yes 

     Rainy       Cold       Mild         No 

     Sunny   Moderate     Normal        Yes 

     Sunny       Cold       High        Yes 

In [9]:
data = pd.read_csv("data.csv")
print("The entered data is \n")
print(data,"\n")
d = np.array(data)[:,:-1]
print("\n The attributes are: \n",d)
target = np.array(data)[:,-1]
print("\n The target is: ",target)
def training(c,t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break          
    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
    return specific_hypothesis
print("\n The final hypothesis is:",training(d,target))
The entered data is 

  Weather Temprature Humidity Goes
0   Sunny       Warm     Mild  Yes
1   Rainy       Cold     Mild   No
2   Sunny   Moderate   Normal  Yes
3   Sunny       Cold     High  Yes 


 The attributes are: 
 [['Sunny' 'Warm' 'Mild']
 ['Rainy' 'Cold' 'Mild']
 ['Sunny' 'Moderate' 'Normal']
 ['Sunny' 'Cold' 'High']]

 The target is:  ['Yes' 'No' 'Yes' 'Yes']

 The final hypothesis is: ['Sunny' '?' '?']
