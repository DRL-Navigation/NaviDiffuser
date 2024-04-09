import torch, math
import torch.nn as nn


class StateEmbed(nn.Module):
    def __init__(self, dim=512):
        super(StateEmbed, self).__init__()
        self.mlp_vector = nn.Linear(3, dim)
        self.conv1d1 =  torch.nn.Conv1d(1, 32, 5, 2, "valid")
        self.conv1d2 =  torch.nn.Conv1d(32, 32, 3, 2, "valid")
        self.mlp_laser = nn.Sequential(
            nn.Linear(7616, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
        )
    
    def _encode_laser(self, laser):
        x = self.conv1d1(laser.unsqueeze(1))
        x = self.conv1d2(x)
        return self.mlp_laser(x.reshape(x.shape[0], -1))   

    def forward(self, vector, laser):
        batch_size, seq_length = laser.shape[0], laser.shape[1]
        vector = vector.reshape((batch_size*seq_length,)+vector.shape[2:])
        laser = laser.reshape((batch_size*seq_length,)+laser.shape[2:])
        vector = self.mlp_vector(vector).reshape(batch_size, seq_length, -1)
        laser = self._encode_laser(laser).reshape(batch_size, seq_length, -1)
        return [vector, laser]
    
class ActionEmbed(nn.Module):
    def __init__(self, dim=512):
        super(ActionEmbed, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim)
        )

    def forward(self, action):
        batch_size, seq_length = action.shape[0], action.shape[1]
        action = action.reshape((batch_size*seq_length,)+action.shape[2:])
        action = self.embed(action).reshape(batch_size, seq_length, -1)
        return action

class RewardEmbed(nn.Module):
    def __init__(self, dim=512):
        super(RewardEmbed, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(3, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim)
        )

    def forward(self, reward):
        batch_size, seq_length = reward.shape[0], reward.shape[1]
        reward = reward.reshape((batch_size*seq_length,)+reward.shape[2:])
        reward = self.embed(reward).reshape(batch_size, seq_length, -1)
        return reward
        
class ConditionEmbed(nn.Module):
    def __init__(self, dim=512):
        assert dim % 8 == 0
        super(ConditionEmbed, self).__init__()
        self.mlp_c = nn.Sequential(
            nn.Linear(3, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, (dim//8)*5)
        )
        self.embed_label = nn.Embedding(num_embeddings=3, embedding_dim=dim//8)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, condition, condition_label):
        batch_size = condition.shape[0]
        c = self.mlp_c(condition)
        l = self.embed_label(condition_label.to(dtype=torch.int64)).reshape(batch_size, -1)
        return self.mlp(torch.cat([c, l], dim=1))
    
class TimeEmbed(nn.Module):
    def __init__(self, dim=512):
        super(TimeEmbed, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim)
        )

    def forward(self, time):
        return self.embed(time)

def sinusoidal_positional_embeddings(seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    pos_embedding = torch.zeros(seq_len, dim)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding

class EncoderEmbed(nn.Module):
    def __init__(self, dim=512):
        super(EncoderEmbed, self).__init__()
        self.token_dim = dim
        self.state_embed = StateEmbed(dim)
        self.action_embed = ActionEmbed(dim)
        self.reward_embed = RewardEmbed(dim)
        self.pe = sinusoidal_positional_embeddings(80, dim)
        self.pe.requires_grad = False

    def forward(self, vector, laser, action, reward, history_mask):
        batch_size, seq_length = action.shape[0], action.shape[1]
        state_tokens = self.state_embed(vector, laser)
        action_token = self.action_embed(action)
        reward_token = self.reward_embed(reward)
        tokens = torch.stack(state_tokens+[action_token, reward_token], dim=1).permute(0, 2, 1, 3).reshape(batch_size, seq_length*(2+1+1), self.token_dim)
        mask = torch.stack([history_mask]*(2+1+1), dim=1).permute(0, 2, 1).reshape(batch_size, seq_length*(2+1+1))
        tokens, _ = tokens.split([seq_length*(2+1+1)-2, 2], dim=1)
        mask, _ = mask.split([seq_length*(2+1+1)-2, 2], dim=1)
        pe = self.pe[:tokens.shape[1]].unsqueeze(0).repeat(tokens.shape[0], 1, 1).to(dtype=torch.float32, device=tokens.device)
        return tokens+pe, mask
    
class DecoderEmbed(nn.Module):
    def __init__(self, dim=512):
        super(DecoderEmbed, self).__init__()
        self.token_dim = dim
        self.action_embed = ActionEmbed(dim)
        self.time_embed = TimeEmbed(dim)
        self.condition_embed = ConditionEmbed(dim)
        self.pe = sinusoidal_positional_embeddings(20, dim)
        self.pe.requires_grad = False

    def forward(self, action, time, condition, condition_label, condition_mask):
        batch_size, seq_length = action.shape[0], action.shape[1]
        action_token = self.action_embed(action)
        time_token = self.time_embed(time).unsqueeze(1)
        condition_token = self.condition_embed(condition, condition_label).unsqueeze(1).repeat(1, seq_length, 1)
        action_token = time_token * action_token
        tokens = torch.stack([condition_token, action_token], dim=1).permute(0, 2, 1, 3).reshape(batch_size, seq_length*(1+1), self.token_dim)
        tokens = torch.cat([time_token, tokens], dim=1)
        condition_mask = condition_mask.unsqueeze(1).repeat(1, seq_length)
        action_mask = torch.ones((batch_size, seq_length), device=condition_mask.device, dtype=condition_mask.dtype)
        mask = torch.stack([condition_mask, action_mask], dim=1).permute(0, 2, 1).reshape(batch_size, seq_length*(1+1))
        time_mask = torch.ones((batch_size, 1), device=condition_mask.device, dtype=condition_mask.dtype)
        mask = torch.cat([time_mask, mask], dim=1)
        pe = self.pe[:tokens.shape[1]].unsqueeze(0).repeat(tokens.shape[0], 1, 1).to(dtype=torch.float32, device=tokens.device)
        return tokens+pe, mask

class PredictionHead(nn.Module):
    def __init__(self, dim=512):
        super(PredictionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 2)
        )

    def forward(self, tokens):
        _, tokens = tokens.split([1, tokens.shape[1]-1], dim=1)
        batch_size, seq_length = tokens.shape[0], tokens.shape[1]//(1+1)
        tokens = tokens.reshape(batch_size, seq_length, 1+1, -1).permute(0, 2, 1, 3)
        token = tokens.split(1, dim=1)[1].squeeze(1)
        token = self.head(token).reshape(batch_size, seq_length, -1)
        return token