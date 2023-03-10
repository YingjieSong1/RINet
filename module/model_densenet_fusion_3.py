import torchvision.models as models
import torch
import torch.nn as nn


class spacial_channel_enhanced_attention(nn.Module):
    def __init__(self,dim,out_dim=None,r=2):
        super(spacial_channel_enhanced_attention, self).__init__()
        self.input_dim=dim
        self.qkv_dim=int(dim/r)
        if out_dim==None:
            out_dim=self.input_dim
        self.q1_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.k1_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.q2_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.k2_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.q3_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.k3_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(in_channels = dim, out_channels = self.qkv_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels = self.qkv_dim, out_channels = dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels = self.qkv_dim, out_channels = out_dim, kernel_size=1, padding=0, bias = False),
            nn.ReLU(inplace=True),
        )

    def spacial_norm(self,attn,H,W):
        assert len(attn.shape)==3
        B,N,_=attn.shape
        attn = attn.softmax(dim=-1)     #[B,N,N]
        attn=torch.sum(attn,dim=1,keepdim=True)#[[B,1,N]
        attn_max=torch.max(attn,dim=-1,keepdim=True).values.repeat(1,1,N)
        attn_min=torch.min(attn,dim=-1,keepdim=True).values.repeat(1,1,N)
        attn=(attn-attn_min)/(attn_max-attn_min)
        attn=attn.repeat(1,self.input_dim,1).reshape(B,self.input_dim, H,W)#B,1,N]->[B,C,N]->[B,C,H,W]
        return attn
    def channel_norm(self,attn):
        assert len(attn.shape)==3
        B,C,_=attn.shape
        attn = attn.softmax(dim=-1)     #[B,C,C]
        attn=torch.sum(attn,dim=1,keepdim=True)#[[B,1,C]
        attn_max=torch.max(attn,dim=-1,keepdim=True).values.repeat(1,1,C)
        attn_min=torch.min(attn,dim=-1,keepdim=True).values.repeat(1,1,C)
        attn=(attn-attn_min)/(attn_max-attn_min)#[[B,1,C]
        return attn[:,0,:]

    def forward(self, x):
        B, C, H,W = x.shape
        N=W*H
        x_input=x.clone()

        k1=self.k1_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        q1=self.q1_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        k2=self.k2_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        q2=self.q2_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        k3=self.k3_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        q3=self.q3_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]
        v=self.v_conv(x).view(B,self.qkv_dim,N).permute(0,2,1) #[B, C, W,H]->[B,qkv_dim,N]->[B,N,qkv_dim]

        attn = (q1 @ k1.transpose(-2, -1)) #[B,N,qkv_dim]->[B,N,N]
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0,2,1).contiguous().reshape(B,self.qkv_dim, H,W)#[B,N,qkv_dim]->[B,qkv_dim,N]->[B,qkv_dim,H,W]

        spacial_attn = (q2 @ k2.transpose(-2, -1))  #[B,N,qkv_dim]->[B,N,N]
        spacial_attn=self.spacial_norm(spacial_attn,H,W)#[B,C,H,W]
        
        channel_att= (q3.transpose(-2, -1) @ k3) #[B,qkv_dim,N]@[B,N,qkv_dim]->[B,qkv_dim,qkv_dim]
        channel_att=self.channel_norm(channel_att).unsqueeze(-1).unsqueeze(-1)#[B,qkv_dim]->[B,qkv_dim,1,1]
        channel_att=self.channel_conv(channel_att).repeat(1,1,H,W)#[B,qkv_dim,1,1]->[B,C,H,W]

        enhance_att=spacial_attn+channel_att

        x=self.output_conv(x)
        x=(x+x_input)*enhance_att

        return x


class Dense_Model(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1,type=None):
        super(Dense_Model, self).__init__()
        self.type=type
        self.dense = models.densenet161(pretrained=bool(load_weight)).features

        for param in self.dense.parameters():
            param.requires_grad = train_enc

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_layer0 = nn.Sequential(*list(self.dense)[:3])
        self.conv_layer1 = nn.Sequential(
        	self.dense.pool0,
        	self.dense.denseblock1,
        	*list(self.dense.transition1)[:3]
        )
        self.conv_layer2 = nn.Sequential(
        	self.dense.transition1[3],
        	self.dense.denseblock2,
        	*list(self.dense.transition2)[:3]
        )
        self.conv_layer3 = nn.Sequential(
        	self.dense.transition2[3],
        	self.dense.denseblock3,
        	*list(self.dense.transition3)[:3]
        )
        self.conv_layer4 = nn.Sequential(
        	self.dense.transition3[3],
        	self.dense.denseblock4
        )

        self.fusion_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 2208, out_channels = 512, kernel_size=3, padding=1, bias = False),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.fusion_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 512+1056, out_channels = 256, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 384+256, out_channels = 192, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 96, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.readout_layer = nn.Sequential(
            nn.Conv2d(in_channels = 1056, out_channels = 128, kernel_size = 3, padding = 1, bias = False),
            nn.ReLU(inplace=True),
            self.linear_upsampling,
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Tanh()
        )

        self.up_module_4=nn.UpsamplingBilinear2d(scale_factor=8)
        self.up_module_3=nn.UpsamplingBilinear2d(scale_factor=4)
        self.up_module_2=nn.UpsamplingBilinear2d(scale_factor=2)

        print(f'attention type:{self.type}')
        if self.type=='RI_spacial_channel':
            self.attention2= spacial_channel_enhanced_attention(dim=384)
            self.attention3= spacial_channel_enhanced_attention(dim=1056)
            self.attention4= spacial_channel_enhanced_attention(dim=2208)
        else: raise AssertionError('Please check the attention type!')
     

    def forward(self, images):
    
        out0 = self.conv_layer0(images)
        out1 = self.conv_layer1(out0)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        
        if self.type in ['RI_spacial_channel']:
            out2 = self.attention2(out2)
            out3 = self.attention3(out3)
            out4 = self.attention4(out4)
        else: raise AssertionError
            
        
        f4=self.fusion_layer4(out4)
        f4_fusion=self.up_module_4(f4)

        f3 = torch.cat((f4,out3), 1)
        f3 = self.fusion_layer3(f3)
        f3_fusion=self.up_module_3(f3)

        f2 = torch.cat((f3,out2), 1)
        f2 = self.fusion_layer2(f2)
        f2_fusion=self.up_module_2(f2)

        f1 = self.fusion_layer1(f2)
        f1_fusion=f1.clone()
        
        x = torch.cat((f1_fusion,f2_fusion,f3_fusion,f4_fusion), 1)# C=(512/8)+(256/4)+(192/2)+96=320 # C=512+256+192+96=1056
        x = self.readout_layer(x)
        x = x.squeeze(1)

        return x


