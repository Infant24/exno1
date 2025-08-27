# Exno:1

Data Cleaning Process

# AIM

To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation

Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm

STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
df=pd.read_csv("C:/Users/admin/Downloads/SAMPLEIDS.csv")
print(df)
```
<img width="1424" height="478" alt="image" src="https://github.com/user-attachments/assets/70d2e1ff-fd53-42a5-93d3-38153c7ead96" />
<img width="1428" height="437" alt="image" src="https://github.com/user-attachments/assets/c5770ea3-4dfd-461d-be86-5bd979ad6803" />
<img width="1437" height="371" alt="image" src="https://github.com/user-attachments/assets/1ca4cdec-b58d-44ad-8d01-7ca242c85c5a" />

df.head()
<img width="1430" height="216" alt="image" src="https://github.com/user-attachments/assets/02ac6f77-3ec8-4db5-9a81-75c083571367" />

df.tail()
<img width="1429" height="236" alt="image" src="https://github.com/user-attachments/assets/3f035121-a56a-4fb8-8853-e36eb22a1028" />

df.isnull()
<img width="1429" height="763" alt="image" src="https://github.com/user-attachments/assets/6111527a-1637-4ea7-95e8-9632f7785eaf" />

df.isnull().sum()
<img width="1424" height="359" alt="image" src="https://github.com/user-attachments/assets/a1887582-deee-40c5-9294-20e69f64d2c8" />

df.isnull().any()
<img width="1423" height="363" alt="image" src="https://github.com/user-attachments/assets/417bb37b-dcbb-4b7a-b082-3d278449a35b" />

df.dropna()
<img width="1429" height="537" alt="image" src="https://github.com/user-attachments/assets/69fc1378-7ebf-4bb9-8b0e-56f489ad8f80" />

df.dropna(axis=0)
<img width="1427" height="537" alt="image" src="https://github.com/user-attachments/assets/c7f93e11-f0a0-44b6-bc51-8d9a4b0dbfeb" />

df.dropna(axis=1)
<img width="1432" height="786" alt="image" src="https://github.com/user-attachments/assets/731d405b-6c8a-4f29-828e-f5bae832fdb7" />

df.fillna(0)
<img width="1426" height="779" alt="image" src="https://github.com/user-attachments/assets/17a21610-1424-43b7-a3f4-5ef55f2823e1" />

df.fillna(method = 'ffill')
<img width="1431" height="775" alt="image" src="https://github.com/user-attachments/assets/f5b8fc43-c904-4407-86c9-87cc54ceb80f" />

df.fillna(method = 'bfill')
<img width="1427" height="782" alt="image" src="https://github.com/user-attachments/assets/019ba19f-6be6-4332-ac58-7a1b122bb4ae" />

df.fillna({'GENDER':'MALE', 'NAME':'SAM', 'ADDRESS':'POONAMALE', 'M1':'10', 'M2':'22','M3':'99', 'M4':'24', 'TOTAL':'100', 'AVG':'125'})
<img width="1432" height="806" alt="image" src="https://github.com/user-attachments/assets/3bcf5565-7d84-46bf-968b-8cd1960216f1" />

ir = pd.read_csv("C:/Users/admin/Downloads/iris.csv")
<img width="1430" height="435" alt="image" src="https://github.com/user-attachments/assets/a6093600-40cf-459f-a1f0-c19b19580c78" />

ir.describe()
<img width="1433" height="376" alt="image" src="https://github.com/user-attachments/assets/010a83e9-ae30-4e42-8db1-c9c031896e23" />

import seaborn as sns
sns.boxplot(x='sepal_width', data=ir)
<img width="1433" height="650" alt="image" src="https://github.com/user-attachments/assets/0697e42a-7170-4777-b4c8-c34fb8c4d5eb" />

```
q1=ir.sepal_width.quantile(0.25)
q3=ir.sepal_width.quantile(0.75)
iqr=q3-q1
print(iqr)
```
<img width="1428" height="117" alt="image" src="https://github.com/user-attachments/assets/de031171-5871-4be2-9ab7-78d6d652bc58" />

rid=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
print(rid['sepal_width'])
<img width="1437" height="183" alt="image" src="https://github.com/user-attachments/assets/8c77bee3-a557-46f3-809a-70a7c0bb4f2d" />

```
delid=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
print(delid)
```
<img width="1436" height="360" alt="image" src="https://github.com/user-attachments/assets/f7388d48-b73d-4381-a1c6-c8370583851b" />

sns.boxplot(x='sepal_width',data=delid)
<img width="1437" height="644" alt="image" src="https://github.com/user-attachments/assets/dd5d1e44-fb84-41e8-8746-4510d93f0262" />

```
import numpy as np
import scipy.stats as stats
z = np.abs(stats.zscore(ir['sepal_width']))
print(z)
```
<img width="1442" height="327" alt="image" src="https://github.com/user-attachments/assets/60bed96f-b184-47f8-8764-91f244143b83" />

```
ir1 = ir[z<3]
print(ir1)
```
<img width="1422" height="373" alt="image" src="https://github.com/user-attachments/assets/f08e9661-c301-492f-9c01-b41e8ca2b8e1" />

# Result

The data cleaning process is successfully excecuted.
