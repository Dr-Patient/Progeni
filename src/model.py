import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.trunc_normal_(m.weight.data, std=0.1, a=-0.2, b=+0.2)
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.1)

class Progeni_og(nn.Module):
    def __init__(self, args, num_drug, num_disease, num_protein, num_sideeffect):
        super(Progeni_og, self).__init__()
        self.args = args 
        self.criterion = MSELoss(reduction='sum')
        # f0(v)
        self.drug_embedding = nn.Parameter(torch.Tensor(num_drug, args.d))
        self.protein_embedding = nn.Parameter(torch.Tensor(num_protein, args.d))
        self.disease_embedding = nn.Parameter(torch.Tensor(num_disease, args.d))
        self.sideeffect_embedding = nn.Parameter(torch.Tensor(num_sideeffect, args.d))
        
        # Wr
        self.drug_drug_wr = nn.Linear(args.d, args.d)
        self.drug_chemical_wr = nn.Linear(args.d, args.d)
        self.drug_disease_wr = nn.Linear(args.d, args.d)
        self.drug_sideeffect_wr = nn.Linear(args.d, args.d)
        self.drug_protein_wr = nn.Linear(args.d, args.d)

        self.protein_protein_wr = nn.Linear(args.d, args.d)
        self.protein_sequence_wr = nn.Linear(args.d, args.d)
        self.protein_disease_wr = nn.Linear(args.d, args.d)
        self.protein_drug_wr = nn.Linear(args.d, args.d)

        self.disease_drug_wr = nn.Linear(args.d, args.d)
        self.disease_protein_wr = nn.Linear(args.d, args.d)

        self.sideeffect_drug_wr = nn.Linear(args.d, args.d)

        # W1
        self.W1 = nn.Linear(2*args.d, args.d)
        # projection G and H
        self.drug_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_drug_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_chemical_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_sideeffect_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_sideeffect_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_protein_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.protein_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_sequence_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.reset_parameters()
        self.apply(initialize_weights)
        
    def reset_parameters(self):
        nn.init.trunc_normal_(self.drug_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_drug_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_chemical_G, std=0.1, a=-0.2, b=+0.2)  
        nn.init.trunc_normal_(self.drug_sideeffect_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_sideeffect_H, std=0.1, a=-0.2, b=+0.2) 
        nn.init.trunc_normal_(self.drug_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_protein_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_sequence_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.disease_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.sideeffect_embedding, std=0.1, a=-0.2, b=+0.2)
    
    def forward(self, drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, protein_disease_normalize, 
    disease_drug_normalize, disease_protein_normalize, sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, 
    drug_protein, protein_disease_mask):
        # passing 1 times (can be easily extended to multiple passes)
        drug_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(drug_drug_normalize, self.drug_drug_wr(self.drug_embedding)) + \
            torch.matmul(drug_chemical_normalize, self.drug_chemical_wr(self.drug_embedding)) + \
            torch.matmul(drug_disease_normalize, self.drug_disease_wr(self.disease_embedding)) + \
            torch.matmul(drug_sideeffect_normalize, self.drug_sideeffect_wr(self.sideeffect_embedding)) + \
            torch.matmul(drug_protein_normalize, self.drug_protein_wr(self.protein_embedding)), 
            self.drug_embedding
        ], dim=1))), p=2.0, dim=1)

        protein_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(protein_protein_normalize, self.protein_protein_wr(self.protein_embedding)) + \
            torch.matmul(protein_sequence_normalize, self.protein_sequence_wr(self.protein_embedding)) + \
            torch.matmul(protein_disease_normalize, self.protein_disease_wr(self.disease_embedding)) + \
            torch.matmul(protein_drug_normalize, self.protein_drug_wr(self.drug_embedding)), 
            self.protein_embedding
        ], dim=1))), p=2.0, dim=1)

        disease_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(disease_drug_normalize, self.disease_drug_wr(self.drug_embedding)) + \
            torch.matmul(disease_protein_normalize, self.disease_protein_wr(self.protein_embedding)),
            self.disease_embedding
        ], dim=1))), p=2.0, dim=1)

        sideeffect_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(sideeffect_drug_normalize, self.sideeffect_drug_wr(self.drug_embedding)),
            self.sideeffect_embedding
        ], dim=1))), p=2.0, dim=1)
        # print(protein_vector1, disease_vector1, drug_vector1, sideeffect_vector1)
        # reconstructing networks
        drug_drug_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_drug_G, self.drug_drug_G.T, drug_vector1.T])
        drug_drug_reconstruct_loss = self.criterion(drug_drug_reconstruct, drug_drug)
        
        drug_chemical_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_chemical_G, self.drug_chemical_G.T, drug_vector1.T])
        drug_chemical_reconstruct_loss = self.criterion(drug_chemical_reconstruct, drug_chemical)

        drug_disease_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_disease_G, self.drug_disease_H.T, disease_vector1.T])
        drug_disease_reconstruct_loss = self.criterion(drug_disease_reconstruct, drug_disease)

        drug_sideeffect_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_sideeffect_G, self.drug_sideeffect_H.T, sideeffect_vector1.T])
        drug_sideeffect_reconstruct_loss = self.criterion(drug_sideeffect_reconstruct, drug_sideeffect)

        protein_protein_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_protein_G, self.protein_protein_G.T, protein_vector1.T])
        protein_protein_reconstruct_loss = self.criterion(protein_protein_reconstruct, protein_protein)

        protein_sequence_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_sequence_G, self.protein_sequence_G.T, protein_vector1.T])
        protein_sequence_reconstruct_loss = self.criterion(protein_sequence_reconstruct, protein_sequence)
        
        protein_disease_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_disease_G, self.protein_disease_H.T, disease_vector1.T])
        tmp = torch.mul(protein_disease_mask, (protein_disease_reconstruct - protein_disease))
        protein_disease_reconstruct_loss = torch.sum(tmp.pow(2))
        

        drug_protein_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_protein_G, self.drug_protein_H.T, protein_vector1.T])
        drug_protein_reconstruct_loss = self.criterion(drug_protein_reconstruct, drug_protein)
        

        # l2-regularization
        l2_loss = torch.linalg.matrix_norm(self.drug_embedding)**2 +\
                torch.linalg.matrix_norm(self.protein_embedding)**2 +\
                torch.linalg.matrix_norm(self.disease_embedding)**2 +\
                torch.linalg.matrix_norm(self.sideeffect_embedding)**2 +\
                torch.linalg.matrix_norm(self.drug_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_chemical_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_sequence_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.disease_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.disease_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.sideeffect_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.W1.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_G)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_H)**2 +\
                torch.linalg.matrix_norm(self.drug_drug_G)**2 +\
                torch.linalg.matrix_norm(self.drug_chemical_G)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_G)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_H)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_G)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_H)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_G)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_H)**2 +\
                torch.linalg.matrix_norm(self.protein_protein_G)**2 +\
                torch.linalg.matrix_norm(self.protein_sequence_G)**2
                
                   

    
        total_loss = protein_disease_reconstruct_loss + 1.0 * (drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss +\
            drug_disease_reconstruct_loss + drug_sideeffect_reconstruct_loss + protein_protein_reconstruct_loss +\
            protein_sequence_reconstruct_loss + drug_protein_reconstruct_loss) + l2_loss * self.args.l2_factor
        
        return total_loss, protein_disease_reconstruct_loss, protein_disease_reconstruct.detach().cpu().numpy()

class Progeni(nn.Module):
    def __init__(self, args, num_drug, num_disease, num_protein, num_sideeffect):
        super(Progeni, self).__init__()
        self.args = args 
        self.criterion = MSELoss(reduction='sum')
        # f0(v)
        self.drug_embedding = nn.Parameter(torch.Tensor(num_drug, args.d))
        self.protein_embedding = nn.Parameter(torch.Tensor(num_protein, args.d))
        self.disease_embedding = nn.Parameter(torch.Tensor(num_disease, args.d))
        self.sideeffect_embedding = nn.Parameter(torch.Tensor(num_sideeffect, args.d))
        
        # Wr
        self.drug_drug_wr = nn.Linear(args.d, args.d)
        self.drug_chemical_wr = nn.Linear(args.d, args.d)
        self.drug_disease_wr = nn.Linear(args.d, args.d)
        self.drug_sideeffect_wr = nn.Linear(args.d, args.d)
        self.drug_protein_wr = nn.Linear(args.d, args.d)

        self.protein_protein_wr = nn.Linear(args.d, args.d)
        self.protein_sequence_wr = nn.Linear(args.d, args.d)
        self.protein_disease_wr = nn.Linear(args.d, args.d)
        self.protein_drug_wr = nn.Linear(args.d, args.d)

        self.disease_drug_wr = nn.Linear(args.d, args.d)
        self.disease_protein_wr = nn.Linear(args.d, args.d)

        self.sideeffect_drug_wr = nn.Linear(args.d, args.d)

        # W1
        self.W1 = nn.Linear(2*args.d, args.d)
        # projection G and H
        self.drug_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_drug_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_chemical_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_sideeffect_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_sideeffect_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.drug_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.drug_protein_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_disease_G = nn.Parameter(torch.Tensor(args.d, args.k))
        self.protein_disease_H = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_protein_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.protein_sequence_G = nn.Parameter(torch.Tensor(args.d, args.k))

        self.reset_parameters()
        self.apply(initialize_weights)
        
    def reset_parameters(self):
        nn.init.trunc_normal_(self.drug_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_drug_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_chemical_G, std=0.1, a=-0.2, b=+0.2)  
        nn.init.trunc_normal_(self.drug_sideeffect_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_sideeffect_H, std=0.1, a=-0.2, b=+0.2) 
        nn.init.trunc_normal_(self.drug_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_protein_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_disease_H, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_protein_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_sequence_G, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.drug_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.protein_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.disease_embedding, std=0.1, a=-0.2, b=+0.2)
        nn.init.trunc_normal_(self.sideeffect_embedding, std=0.1, a=-0.2, b=+0.2)
    
    def forward(self, drug_drug_normalize, drug_chemical_normalize, drug_disease_normalize, drug_sideeffect_normalize, protein_protein_normalize, protein_sequence_normalize, protein_disease_normalize, 
    disease_drug_normalize, disease_protein_normalize, sideeffect_drug_normalize, drug_protein_normalize, protein_drug_normalize, drug_drug, drug_chemical, drug_disease, drug_sideeffect, protein_protein, protein_sequence, protein_disease, 
    drug_protein, protein_disease_mask, drug_protein_co_weight, drug_disease_co_weight, protein_disease_co_weight, drug_protein_co_label, drug_disease_co_label, protein_disease_co_label):
        # passing 1 times (can be easily extended to multiple passes)
        # print(drug_chemical_normalize.shape)
        # print(self.drug_drug_wr(self.drug_embedding).shape)
        drug_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(drug_drug_normalize, self.drug_drug_wr(self.drug_embedding)) + \
            torch.matmul(drug_chemical_normalize, self.drug_chemical_wr(self.drug_embedding)) + \
            torch.matmul(drug_disease_normalize, self.drug_disease_wr(self.disease_embedding)) + \
            torch.matmul(drug_sideeffect_normalize, self.drug_sideeffect_wr(self.sideeffect_embedding)) + \
            torch.matmul(drug_protein_normalize, self.drug_protein_wr(self.protein_embedding)), 
            self.drug_embedding
        ], dim=1))), p=2.0, dim=1)

        protein_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(protein_protein_normalize, self.protein_protein_wr(self.protein_embedding)) + \
            torch.matmul(protein_sequence_normalize, self.protein_sequence_wr(self.protein_embedding)) + \
            torch.matmul(protein_disease_normalize, self.protein_disease_wr(self.disease_embedding)) + \
            torch.matmul(protein_drug_normalize, self.protein_drug_wr(self.drug_embedding)), 
            self.protein_embedding
        ], dim=1))), p=2.0, dim=1)

        disease_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(disease_drug_normalize, self.disease_drug_wr(self.drug_embedding)) + \
            torch.matmul(disease_protein_normalize, self.disease_protein_wr(self.protein_embedding)),
            self.disease_embedding
        ], dim=1))), p=2.0, dim=1)

        sideeffect_vector1 = F.normalize(F.relu(self.W1(torch.cat([
            torch.matmul(sideeffect_drug_normalize, self.sideeffect_drug_wr(self.drug_embedding)),
            self.sideeffect_embedding
        ], dim=1))), p=2.0, dim=1)
        # print(protein_vector1, disease_vector1, drug_vector1, sideeffect_vector1)
        # reconstructing networks
        drug_drug_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_drug_G, self.drug_drug_G.T, drug_vector1.T])
        drug_drug_reconstruct_loss = self.criterion(drug_drug_reconstruct, drug_drug)
        
        drug_chemical_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_chemical_G, self.drug_chemical_G.T, drug_vector1.T])
        drug_chemical_reconstruct_loss = self.criterion(drug_chemical_reconstruct, drug_chemical)

        drug_disease_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_disease_G, self.drug_disease_H.T, disease_vector1.T])
        weighted_diff_drug_disease = torch.mul((drug_disease_reconstruct - drug_disease_co_label), drug_disease_co_weight)
        drug_disease_reconstruct_loss = torch.sum(weighted_diff_drug_disease.pow(2))

        drug_sideeffect_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_sideeffect_G, self.drug_sideeffect_H.T, sideeffect_vector1.T])
        drug_sideeffect_reconstruct_loss = self.criterion(drug_sideeffect_reconstruct, drug_sideeffect)

        protein_protein_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_protein_G, self.protein_protein_G.T, protein_vector1.T])
        protein_protein_reconstruct_loss = self.criterion(protein_protein_reconstruct, protein_protein)

        protein_sequence_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_sequence_G, self.protein_sequence_G.T, protein_vector1.T])
        protein_sequence_reconstruct_loss = self.criterion(protein_sequence_reconstruct, protein_sequence)
        
        protein_disease_reconstruct = torch.linalg.multi_dot([protein_vector1, self.protein_disease_G, self.protein_disease_H.T, disease_vector1.T])
        tmp = torch.mul(protein_disease_mask, (protein_disease_reconstruct - protein_disease_co_label) * protein_disease_co_weight)
        protein_disease_reconstruct_loss = torch.sum(tmp.pow(2))
        

        drug_protein_reconstruct = torch.linalg.multi_dot([drug_vector1, self.drug_protein_G, self.drug_protein_H.T, protein_vector1.T])
        weighted_diff_drug_protein = torch.mul((drug_protein_reconstruct - drug_protein_co_label), drug_protein_co_weight)
        drug_protein_reconstruct_loss = torch.sum(weighted_diff_drug_protein.pow(2))
        

        # l2-regularization
        l2_loss = torch.linalg.matrix_norm(self.drug_embedding)**2 +\
                torch.linalg.matrix_norm(self.protein_embedding)**2 +\
                torch.linalg.matrix_norm(self.disease_embedding)**2 +\
                torch.linalg.matrix_norm(self.sideeffect_embedding)**2 +\
                torch.linalg.matrix_norm(self.drug_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_chemical_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_sequence_wr.weight + 1e-12)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.protein_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.disease_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.disease_protein_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.sideeffect_drug_wr.weight)**2 +\
                torch.linalg.matrix_norm(self.W1.weight)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_G)**2 +\
                torch.linalg.matrix_norm(self.drug_disease_H)**2 +\
                torch.linalg.matrix_norm(self.drug_drug_G)**2 +\
                torch.linalg.matrix_norm(self.drug_chemical_G)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_G)**2 +\
                torch.linalg.matrix_norm(self.drug_sideeffect_H)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_G)**2 +\
                torch.linalg.matrix_norm(self.drug_protein_H)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_G)**2 +\
                torch.linalg.matrix_norm(self.protein_disease_H)**2 +\
                torch.linalg.matrix_norm(self.protein_protein_G)**2 +\
                torch.linalg.matrix_norm(self.protein_sequence_G)**2
                

        total_loss = protein_disease_reconstruct_loss + 1.0 * (drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss +\
            drug_disease_reconstruct_loss + drug_sideeffect_reconstruct_loss + protein_protein_reconstruct_loss +\
            protein_sequence_reconstruct_loss + drug_protein_reconstruct_loss) + l2_loss * self.args.l2_factor
        
        return total_loss, protein_disease_reconstruct_loss, protein_disease_reconstruct.detach().cpu().numpy()