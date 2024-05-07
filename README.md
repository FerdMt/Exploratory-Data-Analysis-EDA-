# Exploratory-Data-Analysis-EDA-
Classification Data with Exploratory Data Analysis (EDA)
#Package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from jcopml.plot import plot_missing_value
from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.tuning import grid_search_params as gsp

pd.set_option("display.max_columns", None)

#Read CSV/Excel
df = pd.read_excel("get_otp_10k_data.xlsx", index_col="merchant_id")
df.drop(columns=["id","phone","trx_id","created_at","verified_at"], inplace=True)
df.head()

#Plot missing value for find null value in data
df = pd.read_excel("get_otp_10k_data.xlsx", index_col="merchant_id")
df.drop(columns=["id","phone","trx_id","created_at","verified_at"], inplace=True)
df.head()

#Using package pandas for find count value
df.price.value_counts()

#Using package pandas for operator value 
df.operator.value_counts()

#Binning Status to status id
df["status_id"]=(df.status)
df['status_id'].replace(["DELIVERED", "SENT", "REJECTED", "ERROR"],[0, 1, 2, 3], inplace=True)
df.head()

#Distribution Plot product id to status id
plt.figure(figsize=(7,6))
sns.distplot(df.product_id[df.status_id==0], bins=[0, 1, 2, 3, 4, 5],color="r", label="DELIVERED")
sns.distplot(df.product_id[df.status_id==1], bins=[0, 1, 2, 3, 4, 5],color="g", label="SENT")
sns.distplot(df.product_id[df.status_id==2], bins=[0, 1, 2, 3, 4, 5],color="c", label="REJECTED")
sns.distplot(df.product_id[df.status_id==3], bins=[0, 1, 2, 3, 4, 5],color="k", label="ERROR")
plt.legend();

#Distribution Plot provider id to status id
plt.figure(figsize=(7,6))
sns.distplot(df.provider_id[df.status_id==0], bins=[0, 1, 2, 3, 4, 5],color="r", label="DELIVERED")
sns.distplot(df.provider_id[df.status_id==1], bins=[0, 1, 2, 3, 4, 5],color="g", label="SENT")
sns.distplot(df.provider_id[df.status_id==2], bins=[0, 1, 2, 3, 4, 5],color="c", label="REJECTED")
sns.distplot(df.provider_id[df.status_id==3], bins=[0, 1, 2, 3, 4, 5],color="k", label="ERROR")
plt.legend();

##Distribution Plot price id to status id
plt.figure(figsize=(7,6))
sns.distplot(df.price[df.status_id==0], bins=[10, 100, 200, 300, 400, 500],color="r", label="DELIVERED")
sns.distplot(df.price[df.status_id==1], bins=[10, 100, 200, 300, 400, 500],color="g", label="SENT")
sns.distplot(df.price[df.status_id==2], bins=[10, 100, 200, 300, 400, 500],color="c", label="REJECTED")
sns.distplot(df.price[df.status_id==3], bins=[10, 100, 200, 300, 400, 500],color="k", label="ERROR")
plt.legend();

##Distribution Plot produck id, provider, price, status id 
# Create a list of numeric columns
numeric_columns = ["product_id", "provider_id", "price", "status_id"]

# Create the figure and subplots
fig, axes = plt.subplots(nrows=len(numeric_columns), figsize=(10, 6))

# Plot the distribution of each numeric column
for i, column in enumerate(numeric_columns):
    sns.distplot(df[column], ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")

# Show the plot
plt.tight_layout()
plt.show()

#crosstab package using pandas status provider_id
pd.crosstab(df.status, df.provider_id,)

#crosstab package using pandas status product
pd.crosstab(df.status, df.product_id,)

##crosstab package using pandas status price
pd.crosstab(df.status, df.price,)

# prompt: data set splitting
X = df.drop(columns="status_id")
y = df.status_id

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


preprocessor= ColumnTransformer([
    ('numeric', num_pipe(scaling="minmax"),["provider_id", "product_id","price"]),
    ('categoric', cat_pipe(encoder="onehot"), ["operator", "status"])
])

#preprocessor Data
preprocessor= ColumnTransformer([
    ('numeric', num_pipe(scaling="minmax", impute="median"), ["provider_id", "product_id"]),
    ('categoric', cat_pipe(encoder="onehot", impute="most_frequent"), ["operator", "message", "price","status"])
])
# traning
pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", KNeighborsClassifier())
])

model = GridSearchCV(pipeline, gsp.knn_params, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
