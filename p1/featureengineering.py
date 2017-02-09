from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

nbits = 512

print "reading in csv"
# importing old for gap values
# importing old for gap values
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print "manipulating into one smiles list"
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
smiles = df_all['smiles']

total = len(smiles)

fpts = [[0 for _ in range(nbits)] for _ in range(total)]

print "making fpts"

for i, smile in enumerate(smiles):
	if i % 100000 == 0:
		print i
	m = Chem.MolFromSmiles(smile)
	fpts[i] = map(lambda x: int(x), AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits))

print "making DataFrame"

df = pd.DataFrame(fpts, columns=['feat'+str(i+1) for i in range(nbits)])

print "storing"

df.to_csv(str(nbits)+'data.csv')

# Time on Wendy's computer: 30 min