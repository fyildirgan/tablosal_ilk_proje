import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = sns.load_dataset("penguins")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()
# print(df.head())
# print(df.info())
# print(df.shape)
# sayisal değerler için:
# print(df.describe())

# tüm değerler için:
# print(df.describe(include='all'))
# print(df.corr())
# eksik değerleri görmek için
# rint(df.isna().sum() / df.count() * 100)
# tablo olusturmak istersem
# nan_table = df.isna().sum() / df.count() * 100
nan_percentage = df.isna().sum() / df.count() * 100
nan_count = df.isna().sum()
nan_table = pd.concat([nan_count, nan_percentage], axis=1)
nan_table.columns = ['Count', 'Percentage']
# print(nan_table)

# from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
df.iloc[:, :] = imputer.fit_transform(df)
df.isna().sum()
# print(df.head())
#df = df.drop(labels=['sex'], axis=1)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender'] = le.fit_transform(df['sex'])
df = df.drop(labels=['sex'], axis=1)
#print(df.head())
#print(df['gender'].value_counts())

species_count = df['species'].value_counts().reset_index()
#sns.barplot(data=species_count, x='index', y='species')
#plt.show()
#print(species_count)
#df[df['species'] == 'Adelie']['body_mass_g']
sns.kdeplot(df[df['species'] == 'Adelie']['body_mass_g'])
sns.kdeplot(df[df['species'] == 'Gentoo']['body_mass_g'])
sns.kdeplot(df[df['species'] == 'Chinstrap']['body_mass_g'])
#plt.show()


#for col in df.columns[2:-1]:
#    print(col)
#FOR DONGUSU ILE YAZABILIRIZ.
#for col in df.columns[2:-1]:
#    for spec in df['species'].unique():
#        sns.kdeplot(df[df['species'] == spec][col], shade=True, label=spec)
#        plt.legend()
#    plt.show()

#Bu islemlerin daha gelismis hali
sns.pairplot(df, hue='species', size=3, diag_kind='hist')
plt.show()


