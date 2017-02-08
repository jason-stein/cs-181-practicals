from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

# importing old for gap values
# importing old for gap values
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# store gap values
Y_train = df_train.gap.values
# row where testing examples start
test_idx = df_train.shape[0]
# delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
# delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

# DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_smiles = df_all[['smiles']]
smiles = df_smiles['smiles'].tolist()

cols = []
for i in xrange(1024):
	cols.append('feat' + str(i+1))

df = pd.DataFrame(columns=(cols))

for i, smile in enumerate(smiles):
	print i
	m = rdkit.Chem.MolFromSmiles(smiles[0])
	fp = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
	vals = list(fp.ToBitString())
	df.loc[i]  = vals

df.to_csv('1024data.csv')