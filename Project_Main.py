import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# DOS_element_compounds = pd.read_excel('dos - project.xlsx', skiprows=[0, 1, 2, 3])
# DOS_bulk_moduli = pd.read_excel('dos - project.xlsx', sheet_name=1)
DOS_elemental = pd.read_excel('dos - project.xlsx', header=0, sheet_name='Elemental DOS')
# print(DOS_elemental)
# shape: 10000 * 23

# pandas:dataframe(excel)- iloc -- array
Y = DOS_elemental.iloc[:, 0].values
X = DOS_elemental.iloc[:, 1:].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train)
# Center the data
X_train = X_train - np.mean(X_train, axis=0)

pca = PCA(n_components=18)
pca.fit(X_train)
# This doesn't agree with SVD or EIG calculation???
Sigma_PCA = pca.singular_values_

print('Sigma_PCA is' + str(Sigma_PCA))

percent_pca = 100*pca.explained_variance_ratio_

print('percent_PCA is' + str(percent_pca))

# for i in Sigma_PCA:
#     percent_PCA = i/sum(Sigma_PCA)
#     print('percent_PCA is' + str(percent_PCA))

plt.figure()
plt.plot(Sigma_PCA)
plt.xlabel('$i$')
plt.ylabel('$\sigma_i$')
plt.grid()
plt.savefig('sigma_PCA')

pca = PCA(n_components=1)
X_proj_PCA = pca.fit_transform(X_train)
print(X_proj_PCA.shape)







