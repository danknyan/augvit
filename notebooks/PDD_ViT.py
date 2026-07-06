#vit modules

#imports
import torch
from torchvision.transforms import v2
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# data transforms
num_classes = 23

# transforms 
train_flip_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

val_transforms = nn.Identity()

train_mixup_tranforms = v2.Compose([
    v2.MixUp(num_classes = num_classes, alpha=0.2),
])

train_flipmix_tranforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.MixUp(num_classes = num_classes, alpha=0.2),
])

og_transforms = v2.Compose([
    #v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get probability of the ground truth class
        pt = torch.exp(-ce_loss) 
        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
# model architecture
# attention module
class Attention(nn.Module):
    def __init__(self, dim_em, heads, head_dim, attn_dropout):
        super(Attention, self).__init__()
        self.dim_em = dim_em
        self.dim_hidden = head_dim  * heads
        self.heads = heads
        self.scale = head_dim  ** -0.5
        self.dropout=nn.Dropout(attn_dropout)
        self.attn = nn.Softmax(dim=-1)

        self.w_qkv = nn.Linear(dim_em, self.dim_hidden * 3, bias=False)
        
        self.fc_out = nn.Linear(self.dim_hidden, dim_em)
    
    def forward(self, x):
        # Query, Key, Value
        qkv = self.w_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.fc_out(out)

        return out

#transformer encoder block
class Encoder(nn.Module):
    def __init__(self, input_dim, ffn_inner_dim , heads, attn_head_dim, attn_dropout, ffn_dropout):
        super(Encoder, self).__init__()
        self.attention = Attention(input_dim, heads, attn_head_dim,attn_dropout)
        self.norm1 =nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, ffn_inner_dim )
        self.activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.fc2 = nn.Linear(ffn_inner_dim , input_dim)


    def forward(self, x):
 
        # Attention sublayer — Pre-LN
        x = x + self.attention(self.norm1(x))
        # FFN sublayer — Pre-LN
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.ffn_dropout(x)
        x = self.fc2(x)
        x = residual + x
        
        return x
        
# transformer block compilers
class Transformer(nn.Module):
    def __init__(self, em_dim, ffn_inner_dim , depth, n_heads, attn_head_dim,attn_dropout,ffn_dropout):
        super(Transformer, self).__init__()
        self.dim_em = em_dim
        self.norm = nn.LayerNorm(em_dim)
        self.trans_blocks = nn.ModuleList(
            [Encoder(em_dim, ffn_inner_dim , n_heads, attn_head_dim,attn_dropout=attn_dropout,ffn_dropout=ffn_dropout) for _ in range(depth)]
        )

    def forward(self, x):
        for trans_block in self.trans_blocks:
            x = trans_block(x)
        return self.norm(x)
    
#final MLP classifier head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, head_hidden_dim , num_classes, head_dropout):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, head_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc_out = nn.Linear(head_hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x
    

# depreciated convolutional patch embedding layer
# saved for record keeping
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, stride, hidden_dim, in_channels = 3):
        super(PatchEmbedding, self).__init__()
        self.cnn = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=stride)
        
    def forward(self, x):
        x = self.cnn(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2) 
        return x

# vision transformer class, the star of the show
class VisionTransformer(nn.Module):
    def __init__(self, 
                patch_size,  
                embedding_dim, 
                ffn_inner_dim, 
                depth,
                attn_head_dim,  
                heads, 
                head_hidden_dim,
                head_dropout,
                attn_dropout, 
                ffn_dropout,
                emb_dropout,
                num_classes = 23,
                in_channels=3,
                img_size=224, 
                ):
        super(VisionTransformer, self).__init__()
        image_height, image_width = img_size, img_size
        patch_height, patch_width = patch_size, patch_size
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        
        
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        n_cls_token = 1
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + n_cls_token, embedding_dim))
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(embedding_dim,
                                    ffn_inner_dim, 
                                    depth, 
                                    heads,
                                    attn_head_dim, 
                                    attn_dropout, 
                                    ffn_dropout)

        self.mlp_head = ClassificationHead(embedding_dim, 
                                           head_hidden_dim, num_classes, head_dropout)

    def forward(self, input):
        batch_size = input.size(0)

        x = self.patch_embed(input)  

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.emb_dropout(x)

        x = self.transformer(x) 
   
        cls_output = x[:, 0] 

        return self.mlp_head(cls_output)  
    

