import torch
import torch.nn as nn
import torch.nn.init


# 自我实现的LayerNorm
class CustomNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        u = x.mean(dim=-1, keepdim=True)
        s = (x - u).pow(2)
        s = s.mean(dim=-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class CrossAttentionGenerator(nn.Module):
    def __init__(self,
                 map_width=8,  # 7
                 embed_dim=16, # 256
                 proj_dim=128, # 256
                 num_heads=4): # 4
        super().__init__()
        self.map_width = map_width
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads   # 多头注意力 最后使用权重矩阵归一？
        self.scale = (proj_dim // self.num_heads) ** (-0.5)  # 这是做什么用的
        # position embedding origin
        self.pos_embed_origin = nn.Parameter(torch.zeros(1, self.map_width ** 2, embed_dim))
        self.pos_embed_da = nn.Parameter(torch.zeros(1, self.map_width ** 2, embed_dim))
        # 对x_origin进行norm
        self.norm_origin = CustomNorm(embed_dim)
        # 对x_origin进行query化，维度映射为proj_dim
        self.qkv_origin = nn.Linear(embed_dim, proj_dim * 3, bias=False)
        # 对x_da进行norm
        self.norm_da = CustomNorm(embed_dim)
        # 对x_da进行k,v化，维度映射为proj_dim
        self.qkv_da = nn.Linear(embed_dim, proj_dim * 3, bias=False)
        # # flatten
        # self.mlp = nn.Linear((map_width ** 2) * proj_dim * 2, 1024)
        # mix prototypes
        self.mix_conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), bias=False)
        # # 参数初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.constant_(m.bias, 0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x_origin, x_da, x_prototypes):
        """
        Args:
            x_origin: resampling_num * 1024
            x_da: (resampling_num * self.da_num) * 1024

        Returns:
        """
        # sampling_size, _, _, _ = x_origin.shape
        x_origin = x_origin.permute(0, 2, 3, 1)
        x_da = x_da.permute(0, 2, 3, 1)
        x_prototypes = x_prototypes.permute(0, 2, 3, 1)

        x_prototypes = x_prototypes.reshape(-1, self.map_width ** 2, self.embed_dim)
        x_prototypes = x_prototypes.unsqueeze(1)

        x_origin = x_origin.reshape(-1, self.map_width ** 2, self.embed_dim)
        x_origin = x_origin + self.pos_embed_origin
        # print(x_origin.size())
        x_origin = self.norm_origin(x_origin)

        # 此时的x_da :((resampling_num * self.da_num), (self.map_width ** 2), self.embed_dim)
        x_da = x_da.reshape(-1, self.map_width ** 2, self.embed_dim)
        x_da = x_da + self.pos_embed_da
        x_da = self.norm_da(x_da)

        B, N, C = x_origin.shape
        qkv_origin = self.qkv_origin(x_origin).reshape(B, N, 3, self.num_heads,
                                                       self.proj_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_origin, k_origin, v_origin = qkv_origin[0], qkv_origin[1], qkv_origin[2]
        qkv_da = self.qkv_da(x_da).reshape(B, N, 3, self.num_heads, self.proj_dim // self.num_heads).permute(2, 0, 3, 1,
                                                                                                             4)
        q_da, k_da, v_da = qkv_da[0], qkv_da[1], qkv_da[2]
        # 互换query
        cross_attn_origin = torch.matmul(q_origin, k_da.transpose(-2, -1)) * self.scale
        cross_attn_da = torch.matmul(q_da, k_origin.transpose(-2, -1)) * self.scale
        cross_attn_origin = cross_attn_origin.softmax(dim=-1)
        cross_attn_da = cross_attn_da.softmax(dim=-1)
        da_v = torch.matmul(cross_attn_origin, v_da).transpose(1, 2).reshape(B, N, self.proj_dim)
        origin_v = torch.matmul(cross_attn_da, v_origin).transpose(1, 2).reshape(B, N, self.proj_dim)
        da_v = da_v.unsqueeze(1)
        origin_v = origin_v.unsqueeze(1)
        da_features = torch.cat((origin_v, da_v, x_prototypes), dim=1)
        da_features = self.mix_conv(da_features)
        da_features = da_features.squeeze(1)
        da_features = da_features.permute(0, 2, 1)
        # B * 256 * 7 * 7
        da_features = da_features.reshape(B, self.embed_dim, self.map_width, self.map_width)

        return da_features


if __name__ == '__main__':
    # cn = CustomNorm(4)
    # x = torch.randn((0, 4, 3, 3))
    # # x = x.permute(0, 2, 3, 1)
    # # x = x.reshape(-1, 9, 4)
    # y = torch.randn((0, 4, 3, 3))
    # z = torch.randn((0, 4, 3, 3))
    # ca = CrossAttentionGenerator(map_width=3, embed_dim=4, proj_dim=4, num_heads=4)
    # t = ca(x, y, z)
    # print(t)
    ca = CrossAttentionGenerator(map_width=2, embed_dim=4, proj_dim=4, num_heads=4)
    ca.init_weights()
    x = torch.zeros((1, 4, 2, 2))
    y = torch.randn((1, 4, 2, 2))
    pro = torch.zeros((1, 4, 2, 2))
    da = ca(y, x, pro)
    print(y)
    print(da)

    # for m in ca.modules():
    #     if isinstance(m, nn.Linear):
    #         print(m)
