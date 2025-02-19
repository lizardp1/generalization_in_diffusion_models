import torch
import torch.nn as nn

class PointUNet(nn.Module):
    def __init__(self, args):
        super(PointUNet, self).__init__()
        
        self.num_blocks = args.num_blocks
        self.input_dim = args.input_dim
        self.hidden_dim = args.num_kernels
        
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b, args)
            
        ########## Mid-layers ##########
        mid_block = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            if l == 0:
                mid_block.append(nn.Linear(self.hidden_dim*(2**b), self.hidden_dim*(2**(b+1))))
            else:
                mid_block.append(nn.Linear(self.hidden_dim*(2**(b+1)), self.hidden_dim*(2**(b+1))))
            mid_block.append(nn.LayerNorm(self.hidden_dim*(2**(b+1))))
            mid_block.append(nn.ReLU(inplace=True))
            
        self.mid_block = nn.Sequential(*mid_block)
        
        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks-1, -1, -1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(b, args)
    
    def init_encoder_block(self, b, args):
        enc_layers = nn.ModuleList([])
        
        if b == 0:
            # First block processes raw input
            enc_layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            enc_layers.append(nn.ReLU(inplace=True))
            
            for _ in range(1, args.num_enc_conv):
                enc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=args.bias))
                enc_layers.append(nn.LayerNorm(self.hidden_dim))
                enc_layers.append(nn.ReLU(inplace=True))
        else:
            # Subsequent blocks process features
            for l in range(args.num_enc_conv):
                if l == 0:
                    enc_layers.append(nn.Linear(self.hidden_dim*(2**(b-1)), self.hidden_dim*(2**b), bias=args.bias))
                else:
                    enc_layers.append(nn.Linear(self.hidden_dim*(2**b), self.hidden_dim*(2**b), bias=args.bias))
                enc_layers.append(nn.LayerNorm(self.hidden_dim*(2**b)))
                enc_layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*enc_layers)
    
    def init_decoder_block(self, b, args):
        dec_layers = nn.ModuleList([])
        
        if b == 0:
            # Final block outputs same dimension as input
            for l in range(args.num_dec_conv-1):
                if l == 0:
                    upsample = nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=args.bias)
                    dec_layers.append(nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=args.bias))
                else:
                    dec_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=args.bias))
                dec_layers.append(nn.LayerNorm(self.hidden_dim))
                dec_layers.append(nn.ReLU(inplace=True))
            
            # Final layer to output score (same dimension as input)
            dec_layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        else:
            # Other decoder blocks
            for l in range(args.num_dec_conv):
                if l == 0:
                    upsample = nn.Linear(self.hidden_dim*(2**(b+1)), self.hidden_dim*(2**b), bias=args.bias)
                    dec_layers.append(nn.Linear(self.hidden_dim*(2**(b+1)), self.hidden_dim*(2**b), bias=args.bias))
                else:
                    dec_layers.append(nn.Linear(self.hidden_dim*(2**b), self.hidden_dim*(2**b), bias=args.bias))
                dec_layers.append(nn.LayerNorm(self.hidden_dim*(2**b)))
                dec_layers.append(nn.ReLU(inplace=True))
                
        return upsample, nn.Sequential(*dec_layers)
    
    def forward(self, x):
        # Ensure input is properly shaped
        if len(x.shape) == 3:  # If input is [B, 1, 2]
            x = x.squeeze(1)
            
        ########## Encoder ##########
        unpooled = []
        for b in range(self.num_blocks):
            x_unpooled = self.encoder[str(b)](x)
            # Instead of pooling, we use the full feature representation
            x = x_unpooled
            unpooled.append(x_unpooled)
        
        ########## Mid-layers ##########
        x = self.mid_block(x)
        
        ########## Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            # Concatenate skip connections along feature dimension
            x = torch.cat([x, unpooled[b]], dim=-1)
            x = self.decoder[str(b)](x)
        
        return x