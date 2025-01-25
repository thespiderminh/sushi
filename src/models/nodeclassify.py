import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.2, use_batchnorm=True):
        super(MLPClassifier, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))
        
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

    def forward(self, input, x_node_secondary):
                    #  x_node_secondary:    # n_det . 32
        x_node = input.x_node               # n_det . 2048
        x_node_clip = input.x_node_clip     # n_det . 512
        x_text = input.x_text               # n_det . 512
        
        x_node_secondary = x_node_secondary.repeat(1, 2048 // 32)

        combined_input = torch.cat([x_node, x_node_secondary, x_node_clip, x_text], dim=1)
        return self.fc_layers(combined_input)


# import torch
# import torch.nn as nn

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, fc_dims, dropout_p=0.2, use_batchnorm=True):
#         super(MLPClassifier, self).__init__()
#         assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(type(fc_dims))
#         layers = []
#         for dim in fc_dims:
#             layers.append(nn.Linear(input_dim, dim))
#             if use_batchnorm and dim != 1:
#                 layers.append(nn.BatchNorm1d(dim))
#             if dim != 1:
#                 layers.append(nn.ReLU(inplace=True)) 
#             if dropout_p is not None and dim != 1:
#                 layers.append(nn.Dropout(p=dropout_p))
#             input_dim = dim
        
#         # Thêm lớp Linear cuối cùng với output_dim=1 cho binary classification
#         layers.append(nn.Linear(input_dim, 1))

#         self.fc_layers = nn.Sequential(*layers)

#     def forward(self, input, x_node_secondary):
#         # x_node_secondary:                 # n_det . 32
#         # x_node = input.x_node               # n_det . 2048
#         x_node_clip = input.x_node_clip     # n_det . 512
#         x_text = input.x_text               # n_det . 512
        
#         x_node_secondary = x_node_secondary.repeat(1, 2048 // 32)

#         combined_input = torch.cat([x_node_secondary, x_node_clip, x_text], dim=1)
#         # combined_input = torch.cat([x_node, x_node_secondary, x_node_clip, x_text], dim=1)
#         return self.fc_layers(combined_input)