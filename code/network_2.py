import numpy as np
import torch.nn as nn
import torch

################################################# network class #################################################
class PointUNet(nn.Module):
    def __init__(self, args):
        super(PointUNet, self).__init__()
        self.num_points = args.num_points
        self.point_dim = args.num_channels
        self.hidden_dim = args.num_kernels
        self.skip = args.skip
        
        # Point-wise feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Global context processing
        self.context_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(3)
        ])
        
        # Point refinement
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU()
            ) for _ in range(2)
        ])
        
        # Output layer
        self.output = nn.Linear(self.hidden_dim, self.point_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        original_input = x
        
        # Extract point features
        point_features = self.point_encoder(x.view(-1, self.point_dim))
        point_features = point_features.view(batch_size, self.num_points, -1)
        
        # Process global context
        for context_net in self.context_nets:
            # Compute global features
            global_features = torch.max(point_features, dim=1, keepdim=True)[0]
            global_features = global_features.expand(-1, self.num_points, -1)
            
            # Combine with point features
            combined = torch.cat([point_features, global_features], dim=-1)
            point_features = point_features + context_net(combined)
        
        # Refine point features
        for refine in self.refinement:
            global_features = torch.max(point_features, dim=1, keepdim=True)[0]
            global_features = global_features.expand(-1, self.num_points, -1)
            combined = torch.cat([point_features, global_features], dim=-1)
            point_features = point_features + refine(combined)
        
        # Generate output
        out = self.output(point_features)
        
        if self.skip:
            # Predict noise to subtract
            return out
        else:
            # Directly predict clean points
            return out

class PointUNet_Old(nn.Module):
    def __init__(self, args): 
        super(PointUNet_Old, self).__init__()

        #print(f"Init args - num_points: {args.num_points}, point_dim: {args.num_channels}")


        self.num_points = args.num_points  # Number of points per sample, 4 for 2x2, 9 for 3x3
        self.point_dim = args.num_channels  # Dimension of each point (2 for 2D GMM)
        self.hidden_dim = args.num_kernels
        self.num_blocks = args.num_blocks
        self.pool_factor = args.pool_window
        self.skip = args.skip

        #print("Initializing PointUNet with:")
        #print("num_points:", self.num_points)
        #print("point_dim:", self.point_dim)
        #print("expected first layer input dim:", self.num_points * self.point_dim)
        
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        self.pool = nn.ModuleDict([])

        # First layer input dimension
        current_dim = self.num_points * self.point_dim  # 4 * 2 = 8
        
        # Create encoder blocks
        for b in range(self.num_blocks):
            encoder_block = []
            out_dim = self.hidden_dim * (2**b)
            
            # Multiple linear layers per block
            for i in range(args.num_enc_conv):
                encoder_block.append(nn.Linear(current_dim, out_dim))
                encoder_block.append(BFBatchNorm1d(out_dim))
                encoder_block.append(nn.ReLU(inplace=True))
                current_dim = out_dim
                
            self.encoder[str(b)] = nn.Sequential(*encoder_block)

            pool_out_dim = current_dim // self.pool_factor
            self.pool[str(b)] = nn.Sequential(
                nn.Linear(current_dim, pool_out_dim),
                nn.ReLU(inplace=True)
            )
            current_dim = pool_out_dim

                                
        ########## Mid-layers ##########
        mid_block = []
        for i in range(args.num_mid_conv):
            mid_block.append(nn.Linear(current_dim, current_dim))
            mid_block.append(BFBatchNorm1d(current_dim))
            mid_block.append(nn.ReLU(inplace=True))
        
        self.mid_block = nn.Sequential(*mid_block)
            
                                    
        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        
        for b in range(self.num_blocks-1, -1, -1):
            # Upsampling layer
            upsample_out_dim = current_dim * self.pool_factor
            self.upsample[str(b)] = nn.Sequential(
                nn.Linear(current_dim, upsample_out_dim),
                nn.ReLU(inplace=True)
            )
            
            decoder_block = []
            in_dim = upsample_out_dim + self.hidden_dim * (2**b)  # Size after concat with skip connection
            if b == 0:
                out_dim = self.point_dim * self.num_points
            else:
                out_dim = self.hidden_dim * (2**b)
            
            # Multiple linear layers per block
            for i in range(args.num_dec_conv):
                if i == args.num_dec_conv - 1 and b == 0:
                    decoder_block.append(nn.Linear(in_dim, out_dim))
                else:
                    decoder_block.append(nn.Linear(in_dim, out_dim))
                    decoder_block.append(BFBatchNorm1d(out_dim))
                    decoder_block.append(nn.ReLU(inplace=True))
                in_dim = out_dim
            
            self.decoder[str(b)] = nn.Sequential(*decoder_block)
            current_dim = out_dim
        
        
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        #print("Model forward input shape:", x.shape)
        batch_size = x.size(0)
        original_shape = x.shape
        x = x.view(batch_size, -1)
        #print(f"After flatten: {x.shape}")
        #print("After flatten in forward:", x.shape)
 
        ########## Encoder ##########
        skip_connections = {}
        skip_connections['input'] = x
        for b in range(self.num_blocks):
            #print(f"\nEncoder block {b}")
            #print(f"Before encoder: {x.shape}")
            unpooled = self.encoder[str(b)](x)
            #print(f"After encoder: {unpooled.shape}")
            skip_connections[str(b)] = unpooled
            x = self.pool[str(b)](unpooled)
            #print(f"After pool: {x.shape}")
            
        ########## Mid-layers ##########
        #print(f"\nBefore mid: {x.shape}")
        x = self.mid_block(x)
        #print(f"After mid: {x.shape}")
        
        ########## Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            #print(f"\nDecoder block {b}")
            #print(f"Before upsample: {x.shape}")
            x = self.upsample[str(b)](x)
            #print(f"After upsample: {x.shape}")
            #print(f"Skip connection shape: {skip_connections[str(b)].shape}")
            x = torch.cat([x, skip_connections[str(b)]], dim=1)
            #print(f"After concat: {x.shape}")
            x = self.decoder[str(b)](x)
            #print(f"After decoder: {x.shape}")

        out = x.view(original_shape)
        #print(f"\nOutput shape: {out.shape}")

        if self.skip:
            # If skip is True, predict the noise to subtract
            #input_x = x.view(batch_size, -1)  # Original input flattened
            return skip_connections['input'].view(original_shape) - out
        else:
            # If skip is False, directly predict denoised points
            return out

class PointUNet_other(nn.Module):
    def __init__(self, args): 
        super(PointUNet, self).__init__()
        
        self.num_points = args.num_points
        self.point_dim = args.num_channels
        self.hidden_dim = args.num_kernels
        self.num_blocks = args.num_blocks
        self.pool_factor = args.pool_window 
        self.skip = args.skip
        
        # Keep original structure but modify dimensions
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        self.pool = nn.ModuleDict([])

        # First layer processes each point's features
        current_dim = self.point_dim  # Start with point dimension instead of flattening
        
        for b in range(self.num_blocks):
            encoder_block = []
            out_dim = self.hidden_dim * (2**b)
            
            # Multiple linear layers per block
            for i in range(args.num_enc_conv):
                encoder_block.append(nn.Linear(current_dim, out_dim))
                encoder_block.append(BFBatchNorm1d(out_dim))
                encoder_block.append(nn.ReLU(inplace=True))
                current_dim = out_dim
                
            self.encoder[str(b)] = nn.Sequential(*encoder_block)

            pool_out_dim = current_dim // self.pool_factor
            self.pool[str(b)] = nn.Sequential(
                nn.Linear(current_dim, pool_out_dim),
                nn.ReLU(inplace=True)
            )
            current_dim = pool_out_dim
                                
        ########## Mid-layers ##########
        mid_block = []
        for i in range(args.num_mid_conv):
            mid_block.append(nn.Linear(current_dim, current_dim))
            mid_block.append(BFBatchNorm1d(current_dim))
            mid_block.append(nn.ReLU(inplace=True))
        
        self.mid_block = nn.Sequential(*mid_block)
                                    
        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        
        for b in range(self.num_blocks-1, -1, -1):
            upsample_out_dim = current_dim * self.pool_factor
            self.upsample[str(b)] = nn.Sequential(
                nn.Linear(current_dim, upsample_out_dim),
                nn.ReLU(inplace=True)
            )
            
            decoder_block = []
            in_dim = upsample_out_dim + self.hidden_dim * (2**b)
            
            if b == 0:
                out_dim = self.point_dim  # Output point dimension
            else:
                out_dim = self.hidden_dim * (2**b)
            
            for i in range(args.num_dec_conv):
                if i == args.num_dec_conv - 1 and b == 0:
                    decoder_block.append(nn.Linear(in_dim, out_dim))
                else:
                    decoder_block.append(nn.Linear(in_dim, out_dim))
                    decoder_block.append(BFBatchNorm1d(out_dim))
                    decoder_block.append(nn.ReLU(inplace=True))
                in_dim = out_dim
            
            self.decoder[str(b)] = nn.Sequential(*decoder_block)
            current_dim = out_dim
        
    def forward(self, x):
        batch_size = x.size(0)
        noisy_input = x  # Store original input
        
        # Process each point separately
        # x is [batch_size, num_points, point_dim]
        # Reshape to [batch_size * num_points, point_dim]
        x = x.view(-1, self.point_dim)
        
        ########## Encoder ##########
        skip_connections = {}
        for b in range(self.num_blocks):
            unpooled = self.encoder[str(b)](x)
            skip_connections[str(b)] = unpooled
            x = self.pool[str(b)](unpooled)
            
        ########## Mid-layers ##########
        x = self.mid_block(x)
        
        ########## Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            x = torch.cat([x, skip_connections[str(b)]], dim=1)
            x = self.decoder[str(b)](x)
        
        # Reshape back to [batch_size, num_points, point_dim]
        out = x.view(batch_size, self.num_points, self.point_dim)
        
        if self.skip:
            # Return predicted noise
            return out
        else:
            # Return denoised points directly
            return noisy_input - out
        
class PointUNet_other_old(nn.Module):
    def __init__(self, args): 
        super(PointUNet, self).__init__()
        self.num_points = args.num_points
        self.point_dim = args.num_channels
        self.hidden_dim = args.num_kernels
        self.num_blocks = args.num_blocks
        self.skip = args.skip
        
        # Process each point individually first
        self.point_encoder = nn.Sequential(
            nn.Linear(self.point_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Process point relationships
        self.relationship_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(self.num_blocks)
        ])
        
        # Final point-wise prediction
        self.point_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.point_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each point independently first
        # x shape: [batch_size, num_points, point_dim]
        point_features = self.point_encoder(x)
        
        # Process relationships between points
        for layer in self.relationship_layers:
            # For each point, concatenate its features with mean of all other points
            other_points = point_features.mean(dim=1, keepdim=True).expand(-1, self.num_points, -1)
            combined = torch.cat([point_features, other_points], dim=-1)
            point_features = point_features + layer(combined)  # Skip connection
        
        # Predict noise for each point
        pred_noise = self.point_decoder(point_features)
        
        if self.skip:
            return pred_noise
        else:
            return pred_noise
        
class PointBFCNN(nn.Module):

    def __init__(self, args):
        super(PointBFCNN, self).__init__()

        self.num_points = args.num_points
        self.point_dim = args.num_channels
        self.hidden_dim = args.num_kernels
        self.num_layers = args.num_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(self.num_points * self.point_dim, self.hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(BFBatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.num_points * self.point_dim))
        
        self.network = nn.Sequential(*layers)


    def forward(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.network(x)
        return x.view(batch_size, self.point_dim, self.num_points)

class BFBatchNorm1d(nn.Module):
    def __init__(self, num_kernels):
        super(BFBatchNorm1d, self).__init__()
        self.register_buffer("running_sd", torch.ones(1,num_kernels,1))
        g = (torch.randn( (1,num_kernels,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        # Reshape to [batch_size, num_kernels, -1]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 1)
        
        training_mode = self.training       
        sd_x = torch.sqrt(x.var(dim=(0,2), keepdim=True, unbiased=False) + 1e-05)
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x)

            x = x * self.gammas.expand_as(x)

        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)
            
        # Reshape back
        x = x.view(batch_size, -1)
        return x

def initialize_network(arch_name, args):
    if arch_name == 'UNet':
        return PointUNet(args)
    elif arch_name == 'BF_CNN':
        return PointBFCNN(args)
    else:
        raise ValueError(f'Architecture {arch_name} not implemented for point data')
    