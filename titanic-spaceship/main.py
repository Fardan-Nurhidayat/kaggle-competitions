import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('titanic-spaceship/train.csv')
label_encoder = LabelEncoder()
df_copy = df.copy()

# Drop a passengerId Column
df_copy.drop('PassengerId', axis=1, inplace=True)

# Drop all rows with NaN values in any column, drop do because sum of null value is below a five percent
df_copy.dropna(inplace=True)

# Check is any a duplicate data
# print(df_copy.duplicated().sum())
# The result is 0 , so there is no duplicate data

# Check about the table
# print(df_copy.info())

# See how the distribution of Transported with histogram
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Transported', data=df_copy)
# The result is , transported and not transported are almost equal , so don't worry about inbalance

# sns.countplot(x='HomePlanet', data=df_copy , hue="Transported")
# Hasil : secara jumlah , banyak dari bumi , disusul europe lalu mars
# Dari Bumi banyak yang selamat / tidak ter teleport, sekitar yang selamat 55% dan 45% yang tidak selamat
# Dari Europe banyak yang terteleport , sekitar 65% ter teleport , dan 35% tidak
# Dari Mars , sekitar 52% ter teleport dan sekitar 48% tidak ter teleport

# sns.countplot(x="VIP", data=df_copy , hue="Transported")
# Secara data yang pennumpang yang VIP sedikit, tapi VIP memiliki jumlah terteleport yang rendah di banding ter teleport
# Lalu yang bukan VIP, jumlah yang terteleport lebih banyak

# sns.countplot(x="Destination" , data=df_copy , hue="Transported")
# Destination paling banyak ialah TRAPPIST-1e
# Dari TRAPPIST-1e , terteleport 40% dan tidak ter teleport 60%
# Ke 55 Cancri e , terteleport 60% dan tidak ter teleport 40%
# Ke PSO J318.5-22 , terteleport 51% dan tidak ter teleport 49%

# sns.countplot(x="CryoSleep" , hue="Transported" , data=df_copy)
# Bisa di pastikan cryosleep adalah fitur paling berpengaruh
# Ketika penumpang tidak memilih untuk cryosleep , maka peluang selamat (tidak terteleport) lebih besar
# Ketika penumpang memilih untuk cryosleep , maka peluang tidak selamat (terteleport) lebih besar

# sns.countplot(x="Cabin" , hue="Transported" , data=df_copy)

numerical_columns = ["Age", "RoomService" , "FoodCourt" , "ShoppingMall" , "Spa" , "VRDeck"]

# sns.heatmap(
#     df_copy.corr(numeric_only=True),
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     linewidths=0.1,
#     cbar_kws=dict(shrink=0.5)
# )

# Setelah dilihat heatmap , tidak ada numerical_columns yang memiliki keterkaitan yang signifikan

# sns.histplot(
#     data=df_copy,
#     x="Age",
#     hue="Transported",
#     kde=True,
#     multiple="stack",
# )
# Age memiliki pengaruh yang cukup baik
# Penumpang yang berumur antara 10 18 cenderung terteleport , sementara di range 20 ke atas cenderung tidak terteleport

# plt.show()
# print(df_copy.head())

# Proses pemetaan Cabin

df_copy[["Deck" , "CabinNum" , "Side"]] = df_copy["Cabin"].str.split("/" , expand=True)
df_copy.drop(columns=["Cabin"] , inplace=True)
# print(pd.crosstab(df_copy["Deck"] , df_copy["Transported"] , normalize="index"))
# print(pd.crosstab(df_copy["CabinNum"] , df_copy["Transported"] , normalize="index"))
# print(pd.crosstab(df_copy["Side"] , df_copy["Transported"] , normalize="index"))
# Ternyata deskripsi setiap cabin ada pengaruh yang signifikan

# Section Feature Engineering

# Mengubah tipe data CabinNum menjadi number
df_copy["CabinNum"] = df_copy["CabinNum"].astype("int32")

# Drop Name columns
df_copy.drop(columns=["Name" ] , inplace=True)

# Feature Engineering
# Create a TotalSpend column
expenses_columns = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]
df_copy["TotalSpend"] = df_copy[expenses_columns].sum(axis=1)

X = df_copy.drop(columns=["Transported"])
y = df_copy["Transported"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = one_hot_encoder.fit_transform(X_train["HomePlanet" , "Destination" , "Deck" , "Side"])

categorical_columns = [
    "HomePlanet",
    "Destination",
    "Deck",
    "Side",
]

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = encoder.fit_transform(X_train[categorical_columns])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X_train.index
)

X_train = X_train.drop(columns=categorical_columns)
X_train = pd.concat([X_train, encoded_df], axis=1)
X_train["VIP"] = X_train["VIP"].astype("int32")
X_train["CryoSleep"] = X_train["CryoSleep"].astype("int32")

encoded_test = encoder.transform(X_test[categorical_columns])
encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=encoder.get_feature_names_out(categorical_columns),
    index=X_test.index
)

X_test = X_test.drop(columns=categorical_columns)
X_test = pd.concat([X_test, encoded_test_df], axis=1)
X_test["VIP"] = X_test["VIP"].astype("int32")
X_test["CryoSleep"] = X_test["CryoSleep"].astype("int32")

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
print(random_forest.score(X_test, y_test))