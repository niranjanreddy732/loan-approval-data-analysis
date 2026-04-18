# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("C:/Users/niran/OneDrive/Desktop/python/loanapproval.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# ==============================
# 3. BASIC INFO
# ==============================
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# ==============================
# 4. DATA CLEANING (SAFE)
# ==============================
df = df.dropna(how='all')

# Fill missing values forward
df.fillna(method='ffill', inplace=True)

# ==============================
# 5. HANDLE NUMERIC DATA
# ==============================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) == 0:
    print("❌ No numeric columns found!")
    exit()

num_col = numeric_cols[0]
print("\nUsing column:", num_col)

# ==============================
# 6. HISTOGRAM
# ==============================
plt.figure()
plt.hist(df[num_col], bins=10)
plt.title("Distribution of " + num_col)
plt.xlabel(num_col)
plt.ylabel("Frequency")
plt.show()

# ==============================
# 7. BAR CHART
# ==============================
top10 = df[num_col].value_counts().head(10)

plt.figure()
top10.plot(kind='bar')
plt.title("Top Values of " + num_col)
plt.xlabel(num_col)
plt.ylabel("Count")
plt.show()

# ==============================
# 8. PIE CHART
# ==============================
plt.figure()
top10.plot(kind='pie', autopct='%1.1f%%')
plt.title("Distribution Share")
plt.ylabel("")
plt.show()

# ==============================
# 9. BOXPLOT
# ==============================
plt.figure()
sns.boxplot(x=df[num_col])
plt.title("Boxplot (Outlier Detection)")
plt.show()

# ==============================
# 10. CORRELATION HEATMAP
# ==============================
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 1:
    plt.figure()
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Matrix")
    plt.show()

# ==============================
# 11. SCATTER PLOT
# ==============================
plt.figure()
plt.scatter(df.index, df[num_col])
plt.title("Scatter Plot")
plt.xlabel("Index")
plt.ylabel(num_col)
plt.show()

# ==============================
# 12. REGRESSION LINE
# ==============================
plt.figure()
sns.regplot(x=df.index, y=df[num_col],
            line_kws={"color": "red"})
plt.title("Regression Line")
plt.xlabel("Index")
plt.ylabel(num_col)
plt.show()

# ==============================
# 13. CATEGORIZATION
# ==============================
def categorize(val):
    if val < df[num_col].quantile(0.33):
        return "Low"
    elif val < df[num_col].quantile(0.66):
        return "Medium"
    else:
        return "High"

df['Category'] = df[num_col].apply(categorize)

category_count = df['Category'].value_counts()

plt.figure()
plt.bar(category_count.index, category_count.values)
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# ==============================
# 14. MACHINE LEARNING
# ==============================
X = df.index.values.reshape(-1, 1)
y = df[num_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ==============================
# 15. FINAL OUTPUT
# ==============================
print("\n✅ PROJECT EXECUTED SUCCESSFULLY!")