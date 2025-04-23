## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/f53fc745-0cd8-4780-9622-48087cd8c749)
```
from sklearn.preprocessing import OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/1b1ff4e4-99cf-4929-b32f-fd82311ef7c6)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/38afb72c-1988-4a8a-ae0e-572034005d21)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/581166d4-f8b0-4d21-895b-31e51e0ff2ea)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/2637f859-2cc6-4e48-b722-c22c1254cf6f)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/7434b15a-059d-4075-a302-443fc5275b8f)
```
from category_encoders import BinaryEncoder
df5=pd.read_csv("data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df5['Ord_2'])
dfb=pd.concat([df5,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/bd40ab10-e41e-413d-86a5-3987da4944a4)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
dft=df.copy()
new=te.fit_transform(X=dft["City"],y=dft["Target"])
dft=pd.concat([dft,new],axis=1)
dft
```
![image](https://github.com/user-attachments/assets/20c96b64-802e-4575-ad84-03b68a3f393c)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/b5d9dfc7-cd5e-479c-8914-9f551937b579)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/21ac6799-7e99-4069-8f8e-091ac5e27a43)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/5b527a89-3501-479a-b70e-1253cf58bd7b)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7563b120-131f-4348-b404-3c85cff8fb41)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/603adc82-3f5a-418b-8fae-f28fa0966da7)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/986503c4-e4ca-459b-a2ec-a8ef900451d5)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c6dadcfb-7160-4ed1-b0a5-74ccb0b89e41)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/b48dc32e-e129-4e9c-93ea-8c9f4502ce77)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/1b1ffe21-aea3-45bb-a465-42a03350fb99)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6a71ede4-fc2b-4837-b4ce-4bc019730224)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/2c9c2bdd-8834-4571-a0f9-3c610f0aa271)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ef8bbaa9-6ed5-4592-9554-272269f8a639)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce76c5f2-28df-4357-a403-9b1ee7e9b783)

# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
