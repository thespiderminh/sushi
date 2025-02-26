# import torch
# import torch.nn as nn

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, fc_dims, dropout_p=0.2, use_batchnorm=True):
#         super(MLPClassifier, self).__init__()

#         assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
#             type(fc_dims))
        
#         layers = []
#         for dim in fc_dims:
#             layers.append(nn.Linear(input_dim, dim))
#             if use_batchnorm and dim != 1:
#                 layers.append(nn.BatchNorm1d(dim))
            
#             if dim != 1:
#                 layers.append(nn.LeakyReLU(inplace=True))
         
            
#             if dropout_p is not None and dim != 1:
#                 layers.append(nn.Dropout(p=dropout_p))
#             input_dim = dim
        
#         layers.append(nn.Linear(input_dim, 1))

#         self.fc_layers = nn.Sequential(*layers)

#         expand_node = []
#         expand_node.append(nn.Linear(32, 64))
#         expand_node.append(nn.LeakyReLU(inplace=True))
#         expand_node.append(nn.Linear(64, 128))
#         expand_node.append(nn.LeakyReLU(inplace=True))
#         expand_node.append(nn.Linear(128, 256))
#         expand_node.append(nn.LeakyReLU(inplace=True))
#         expand_node.append(nn.Linear(256, 512))
#         expand_node.append(nn.LeakyReLU(inplace=True))
#         expand_node.append(nn.Linear(512, 2048))
#         expand_node.append(nn.LeakyReLU(inplace=True))

#         self.expand_secondary = nn.Sequential(*expand_node)

#     def forward(self, input, x_node_secondary):
#                     #  x_node_secondary:    # n_det . 32
#         x_node = input.x_node               # n_det . 2048
#         x_node_clip = input.x_node_clip     # n_det . 512
#         x_text = input.x_text               # n_det . 512
        
#         # x_node_secondary = x_node_secondary.repeat(1, 2048 // 32)
#         x_node_secondary = self.expand_secondary(x_node_secondary)

#         combined_input = torch.cat([x_node, x_node_secondary, x_node_clip, x_text], dim=1)
#         return self.fc_layers(combined_input)

import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.2, use_batchnorm=True):
        super(MLPClassifier, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))
        
        #######
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))
            
            if dim != 1:
                layers.append(nn.ReLU(inplace=True))
            
            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        layers.append(nn.Linear(input_dim, 1))

        self.fc_layers = nn.Sequential(*layers)

        #######
        text_layers = []
        input_dim = 1024
        for dim in [512, 512, 256, 256, 128, 64, 64, 32]:
            text_layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                text_layers.append(nn.BatchNorm1d(dim))
            
            if dim != 1:
                text_layers.append(nn.ReLU(inplace=True))
            
            if dropout_p is not None and dim != 1:
                text_layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.text_layers = nn.Sequential(*text_layers)

        #######
        x_node_layers = []
        input_dim = 2048
        for dim in [1024, 512, 256, 256, 128, 128, 64, 32]:
            x_node_layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                x_node_layers.append(nn.BatchNorm1d(dim))
            
            if dim != 1:
                x_node_layers.append(nn.ReLU(inplace=True))
            
            if dropout_p is not None and dim != 1:
                x_node_layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.x_node_layers = nn.Sequential(*x_node_layers)


    def forward(self, input, x_node_secondary):
                    #  x_node_secondary:                            # n_det . 32
        x_node = input.x_node                                       # n_det . 2048
        x_node_clip = input.x_node_clip                             # n_det . 512
        x_text = input.x_text                                       # n_det . 512
        
        x_node = self.x_node_layers(x_node)                         # n_det . 32

        combined_text = torch.cat([x_node_clip, x_text], dim=1)
        combined_text = self.text_layers(combined_text)             # n_det . 32

        combined_input = torch.cat([x_node_secondary, x_node, combined_text], dim=1) # 96
        # combined_input = torch.cat([x_node, combined_text], dim=1) # 64
        return self.fc_layers(combined_input)
