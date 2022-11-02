import numpy as np 
np.random.seed(26)
import os, torch
basedir = os.path.abspath(os.path.dirname(__file__))
os.chdir(basedir)

file_path = os.path.abspath(os.path.join(basedir, 'literature_data'))
data_path = os.path.abspath(os.path.join(basedir, 'new_data'))

# 1. drug_disease
drug_disease_co = torch.Tensor(np.loadtxt(os.path.join(file_path, 'drugdis_new.txt')))
drug_disease = torch.Tensor(np.loadtxt(os.path.join(data_path, 'mat_drug_disease.txt')))
print(drug_disease_co.shape, drug_disease.shape)
# assert 1==0

drug_disease_co_weight = torch.where(drug_disease == 1, torch.sqrt(torch.sigmoid(drug_disease_co + 0)), torch.full_like(drug_disease, 1)).numpy()
drug_disease_co_label = torch.where(drug_disease == 1, torch.sigmoid(drug_disease_co + torch.log(torch.tensor(3))), torch.full_like(drug_disease, 0)).numpy()  #TODO
# drug_disease_co_label_rand = np.random.uniform(low=0.75, high=1.0, size=drug_disease_co_label.shape) * (drug_disease.numpy())
np.save("new_data/drug_disease_co_weight.npy", drug_disease_co_weight, allow_pickle=True)
np.save("new_data/drug_disease_co_label.npy", drug_disease_co_label, allow_pickle=True)
# np.save("data/drug_disease_co_label_rand.npy", drug_disease_co_label_rand, allow_pickle=True)
print(drug_disease[0][0], drug_disease_co[0][0], drug_disease_co_label[0][0])
print(drug_disease_co.max())

# 2. drug_protein
drug_protein_co = torch.Tensor(np.loadtxt(os.path.join(file_path, 'drugpro_new.txt')))
drug_protein = torch.Tensor(np.loadtxt(os.path.join(data_path, 'mat_drug_protein.txt')))
print(drug_protein_co.shape, drug_protein.shape)

drug_protein_co_weight = torch.where(drug_protein == 1, torch.sqrt(torch.sigmoid(drug_protein_co + 0)), torch.full_like(drug_protein, 1)).numpy()
drug_protein_co_label = torch.where(drug_protein == 1, torch.sigmoid(drug_protein_co + torch.log(torch.tensor(3))), torch.full_like(drug_protein, 0)).numpy()
# drug_protein_co_label_rand = np.random.uniform(low=0.75, high=1.0, size=drug_protein_co_label.shape) * (drug_protein.numpy())
np.save("new_data/drug_protein_co_weight.npy", drug_protein_co_weight, allow_pickle=True)
np.save("new_data/drug_protein_co_label.npy", drug_protein_co_label, allow_pickle=True)
# np.save("data/drug_protein_co_label_rand.npy", drug_protein_co_label_rand, allow_pickle=True)

# 3. protein_disease
protein_disease_co = torch.Tensor(np.loadtxt(os.path.join(file_path, 'prodis_new.txt')))
protein_disease = torch.Tensor(np.loadtxt(os.path.join(data_path, 'mat_protein_disease.txt')))
print(protein_disease_co.shape, protein_disease.shape)

protein_disease_co_weight = torch.where(protein_disease == 1, torch.sqrt(torch.sigmoid(protein_disease_co + 0)), torch.full_like(protein_disease, 1)).numpy()
protein_disease_co_label = torch.where(protein_disease == 1, torch.sigmoid(protein_disease_co + torch.log(torch.tensor(3))), torch.full_like(protein_disease, 0)).numpy()
# protein_disease_co_label_rand = np.random.uniform(low=0.75, high=1.0, size=protein_disease_co_label.shape) * (protein_disease.numpy())
np.save("new_data/protein_disease_co_weight.npy", protein_disease_co_weight, allow_pickle=True)
np.save("new_data/protein_disease_co_label.npy", protein_disease_co_label, allow_pickle=True)
# # np.save("data/protein_disease_co_label_rand.npy", protein_disease_co_label_rand, allow_pickle=True)
