import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('titanic-spaceship/train.csv')

df_copy = df.copy()

# Drop a passengerId Column
df_copy.drop('PassengerId', axis=1, inplace=True)

# Drop all rows with NaN values in any column, drop do because sum of null value is below a five percent
df_copy.dropna(inplace=True)

print(df_copy.info())

# See how the distribution of Transported with histogram
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Transported', data=df_copy)
# plt.show()
# The result is , transported and not transported are almost equal
