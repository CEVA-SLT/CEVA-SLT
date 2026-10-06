import torch
import torch.nn as nn
import torch.nn.functional as F


class RecoverFeatureContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
        print(f"   RecoverFeatureContrastiveLoss 初始化:")
        print(f"   - 温度系数: {temperature}")
        print(f"   - 对比方式: 全局序列级")
        print(f"   - 正样本: 同样本的prior和posterior")
        print(f"   - 负样本: batch内其他样本的posterior")
    
    def _masked_pool(self, features, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        
        masked_features = features * mask_expanded  
        sum_features = masked_features.sum(dim=1)   
        
        valid_lengths = mask_expanded.sum(dim=1).clamp(min=1e-8)  
        pooled = sum_features / valid_lengths  
        
        return pooled
    
    def forward(self, prior_recover, posterior_recover, mask):
        
        batch_size = prior_recover.shape[0]
        

        prior_pooled = self._masked_pool(prior_recover, mask)     
        posterior_pooled = self._masked_pool(posterior_recover, mask) 
        
        prior_norm = F.normalize(prior_pooled, p=2, dim=-1)       
        posterior_norm = F.normalize(posterior_pooled, p=2, dim=-1) 
        
      
        similarity_matrix = torch.matmul(prior_norm, posterior_norm.T)
        

        similarity_matrix = similarity_matrix / self.temperature
        
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        

        loss = F.cross_entropy(similarity_matrix, labels)
        

        with torch.no_grad():
            pos_sim = torch.diagonal(similarity_matrix).mean()
            
            mask_diag = torch.eye(batch_size, device=similarity_matrix.device).bool()
            neg_sim = similarity_matrix.masked_select(~mask_diag).mean()

        return {
            'loss': loss,
            'pos_sim': pos_sim.item(),
            'neg_sim': neg_sim.item(),
        }
