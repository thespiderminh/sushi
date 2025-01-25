import numpy as np
import pandas as pd
import wandb
import yaml
import torch.optim as optim
from torch_geometric.data import Batch
from src.models.mpntrack import MOTMPNet
from src.models.hiclnet import HICLNet
from src.utils.deterministic import seed_worker, seed_generator
from src.data.refer_kitti_datasets import NodeClassifierDataset
from torch_geometric.data import DataLoader
import torch
from src.utils.motion_utils import compute_giou_fwrd_bwrd_motion_sim
import os.path as osp
import time
import statistics
import os
from src.models.nodeclassify import MLPClassifier
from src.tracker.postprocessing import Postprocessor
from torch import nn


class NodeClassifier:
    def __init__(self, config, seqs, splits):
        self.config = config
        self.seqs = seqs
        self.train_split, self.val_split, self.test_split = splits

        # Load the model (currently MPNTrack)
        self.model, self.node_classify_model = self._get_model()

        # Load pre-train sushi
        sushi_model_folder = self.config.sushi_model_folder
        sushi_model_file = osp.join(sushi_model_folder, os.listdir(sushi_model_folder)[0])
        self.model.load_state_dict(torch.load(sushi_model_file, map_location=self.config.device))

        print("Finish creating model")

        # Training - Set up the dataset and optimizer
        self.train_dataset = self._get_dataset(mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=6, num_workers=self.config.num_workers,
                                shuffle=True,
                                worker_init_fn=seed_worker, generator=seed_generator(), )
        self.optimizer = self._get_optimizer(lr=self.config.mlp_lr, weight_decay=self.config.mlp_weight_decay)
            
        
        # Get validation dataset if exists
        if self.val_split:
            self.val_dataset = self._get_dataset(mode='val')
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, num_workers=self.config.num_workers,
                                shuffle=True,
                                worker_init_fn=seed_worker, generator=seed_generator(), )
            
        print("Finish setting up datasets")

        # Iteration and epoch
        self.train_iteration = 0
        self.train_epoch = 0
        self.verbose_iteration = self.config.mlp_verbose

    def _get_model(self):
        """
        Load the hierarchical model
        """

        # Read mpntrack config file
        with open(r'configs/mpntrack_cfg.yaml') as file:
            mpntrack_params = yaml.load(file, Loader=yaml.FullLoader)

        # Lắp mấy cái trong config vào mpntrack_cfg
        mpntrack_params['graph_model_params']['encoder_feats_dict']['node_in_dim'] = self.config.node_dim

        # Update the HICL feats config
        mpntrack_params['graph_model_params']['do_hicl_feats']=self.config.do_hicl_feats
        mpntrack_params['graph_model_params']['hicl_feats_encoder'].update(self.config.hicl_feats_args)

        return HICLNet(submodel_type=MOTMPNet, submodel_params=mpntrack_params['graph_model_params'],
                        hicl_depth=self.config.hicl_depth, use_motion=self.config.mpn_use_motion,
                        use_reid_edge=self.config.mpn_use_reid_edge, use_pos_edge=self.config.mpn_use_pos_edge,
                        share_weights=self.config.share_weights, edge_level_embed=self.config.edge_level_embed,
                        node_level_embed=self.config.node_level_embed
                        ).to(self.config.device), MLPClassifier(input_dim=self.config.mlp_input_dim, fc_dims=self.config.mlp_fc_dims,
                                                                dropout_p=self.config.mlp_drop_out, use_batchnorm=self.config.mlp_batch_norm).to(self.config.device)

    def _get_dataset(self, mode):
        """
        Create dataset objects
        """
        return NodeClassifierDataset(config=self.config, seqs=self.seqs[mode], mode=mode)

    def _get_optimizer(self, lr=0.001, weight_decay=0.0001):
        """
         Set up the optimizer
        """
        optimizer = optim.Adam(self.node_classify_model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer   

    def _save_model(self):
        """
        Save the model
        """
        # Create models folder
        model_path = osp.join(self.config.experiment_path, "models")
        os.makedirs(model_path, exist_ok=True)

        # Create the file
        file_name = osp.join(self.config.experiment_path, "models", "mlp_model_" + str(self.train_epoch) + ".pth")
        torch.save(self.node_classify_model.state_dict(), file_name)

    def _hicl_to_curr(self, hicl_graphs):
        """
        Method that creates a batch of current graphs from hierarchical graphs in three steps:
        1) Create the batch graphs with node features and all time valid edge connections
        2) **Optionally** compute motion features for all time valid edge connections
        3) Use those motion features, as well as reid to define KNN edges and define 
        edge features for each graph in the batch to obtain the final graphs
        """
        # Tạo graph batch từ cái phân cấp ban đầu
        batch = Batch.from_data_list([hicl_graph.construct_curr_graph_nodes(self.config)
                                                        for hicl_graph in hicl_graphs])
        
        curr_depth = hicl_graphs[0].curr_depth
        if self.config.do_motion and curr_depth >0:
            # Tính vị trí trước và sau theo dự đoán
            motion_pred = self.predict_motion(batch, curr_depth = curr_depth)
            batch.pruning_score = compute_giou_fwrd_bwrd_motion_sim(batch, motion_pred)
            
            if 'estimate_vel' in motion_pred[0]:
                batch.fwrd_vel, batch.bwrd_vel = motion_pred[0]['estimate_vel'], motion_pred[1]['estimate_vel']
                            
        else:
            motion_pred = None
        
        # Now unbatch graphs, add their remaining features, and batch them again
        # Vốn cái batch chỉ là chỗ chứa data, chưa có cấu trúc các cạnh, nên phải thêm cạnh vào chúng
        curr_graphs = Batch.to_data_list(batch)
        data_list = []
        unfit_batch = []
        fit_batch = []
        convert_batch = {}
        for _, a in enumerate(zip(curr_graphs, hicl_graphs)):
            curr_graph, hicl_graph = a
            if ((curr_graph.edge_index is not None) and (curr_graph.edge_index.numel())):
                data_list.append(hicl_graph.add_edges_to_curr_graph(self.config, curr_graph))
                fit_batch.append(_)
            else:
                unfit_batch.append(_)

        for i in range(len(fit_batch)):
            convert_batch[i] = fit_batch[i]
                
        curr_graph_batch = Batch.from_data_list(data_list)
        # curr_graph_batch = Batch.from_data_list([hicl_graph.add_edges_to_curr_graph(self.config, curr_graph)
        #                                          for curr_graph, hicl_graph in zip(curr_graphs, hicl_graphs) if ((curr_graph.edge_index is not None) and (curr_graph.edge_index.numel()))])

        return curr_graph_batch, unfit_batch, convert_batch, motion_pred

    def _log_tb_mot_metrics_on_wandb(self, mot_metrics):
        path = list(mot_metrics[self.metrics_func].keys())[0]
        _METRICS_GROUPS = ['HOTA', 'CLEAR', 'Identity']
        _METRICS_TO_LOG = ['HOTA','AssA', 'DetA', 'MOTA', 'IDF1']
        cls = mot_metrics[self.metrics_func][path]['COMBINED_SEQ'].keys()
        metrics_ = mot_metrics[self.metrics_func][path]['COMBINED_SEQ'][list(cls)[0]]
        for metrics_group_name in _METRICS_GROUPS:
            group_metrics = metrics_[metrics_group_name]
            for metric_name, metric_val in group_metrics.items():
                if metric_name in _METRICS_TO_LOG:
                    if isinstance(metric_val, np.ndarray):
                        metric_val = np.mean(metric_val)
                    wandb.log({"epoch": self.train_epoch, metric_name: metric_val})
                    if metric_name == 'HOTA':
                        print(f"HOTA: {metric_val}")

    def _train_epoch(self, dataloader, mode="Training"):
        """
        Train a single epoch
        """
        logs = {"Loss": [], "Time": [],
                "TP": [], "FP": [], "TN": [], "FN": [],
                "Accuracy": [], "Recall": [], "Precision": [], "F1": []}
        
        if mode == "Training":
            pos_weight = torch.tensor([self.train_dataset.N_neg / self.train_dataset.N_pos], device=self.config.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif mode == "Validating":
            self.criterion = nn.BCEWithLogitsLoss()

        self.criterion.to(self.config.device)

        for i, train_batch in enumerate(dataloader):
            t_start = time.time()

            # Iteration update
            self.train_iteration += 1

            train_batch.to(self.config.device)  # Send batch to the device
            data = train_batch.to_data_list()  # Initialize the hierarchical graphs
            curr_batch, unfit_batch, convert_batch, _ = self._hicl_to_curr(hicl_graphs=data)  # Create curr_graphs from hierarachical graphs
            outputs, x_node_secondary = self.model(curr_batch, 0)

            batch = Batch.from_data_list([datum for datum in data])

            # Tính output
            outputs = self.node_classify_model(batch, x_node_secondary)

            # Tính toán loss
            loss = self.criterion(outputs, batch.y_node)

            if mode == 'Training':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Keep track of the logs
            self._calculate_true_false_metrics(outputs, batch.y_node, logs)
            t_end = time.time()
            logs["Loss"].append(loss.detach().item())
            logs["Time"].append(t_end-t_start)

            # Verbose
            if i % self.verbose_iteration == 0 and i != 0:
                print(f"Iteration {i} / {len(dataloader)} - {mode} Loss:", statistics.mean(logs["Loss"][i-self.verbose_iteration:i]), '- Time:', sum(logs["Time"][i-self.verbose_iteration:i]))  # Verbose

            train_batch.cpu()
            for i in data:
                i.cpu()
            loss.cpu()

        if mode == 'Training':
            self.train_epoch += 1
        self._postprocess_logs(logs)
        
        return logs

    def train(self):
        """
        Perform a full training
        """

        #raise RuntimeError
        assert self.node_classify_model.training, "Training error: Model is not in training mode"

        train_logs = {"Loss": [], "Accuracy": [], "Recall": [], "Precision": [], "F1": [],}  # Training logs
        val_logs = {"Loss": [], "Accuracy": [], "Recall": [], "Precision": [], "F1": [],}  # Validating logs

        # Training loop
        for epoch in range(1, self.config.num_epoch_mlp + 1):
            t_start = time.time()
            print("###############")
            print("     Epoch ", epoch)
            print("###############")

            # Create epoch output dir
            epoch_path = osp.join(self.config.experiment_path, 'Epoch' + str(epoch) + '_mlp')
            os.makedirs(epoch_path, exist_ok=True)

            # Train for one epoch
            self.node_classify_model.train()
            epoch_train_logs = self._train_epoch(self.train_dataloader, mode="Training")

            # Train loss logs
            train_logs["Loss"].append(epoch_train_logs["Loss"])
            train_logs["Accuracy"].append(epoch_train_logs["Accuracy"])
            train_logs["Recall"].append(epoch_train_logs["Recall"])
            train_logs["Precision"].append(epoch_train_logs["Precision"])
            train_logs["F1"].append(epoch_train_logs["F1"])
            print("Average Training Loss: ", train_logs["Loss"][-1])

            # Validation steps
            if self.val_split:
                self.node_classify_model.eval()
                epoch_val_logs = self._train_epoch(self.val_dataloader, mode="Validating")

                val_logs["Loss"].append(epoch_val_logs["Loss"])
                val_logs["Accuracy"].append(epoch_val_logs["Accuracy"])
                val_logs["Recall"].append(epoch_val_logs["Recall"])
                val_logs["Precision"].append(epoch_val_logs["Precision"])
                val_logs["F1"].append(epoch_val_logs["F1"])

                print("Average Validating Loss: ", val_logs["Loss"][-1])

            # Plot losses
            self._plot_losses_on_wandb(train_logs, val_logs)

            # Save model checkpoint
            self._save_model()

            # Time information
            t_end = time.time()
            print(f"Epoch {epoch} completed in {round((t_end - t_start) / 60, 2)} minutes")

    def _calculate_true_false_metrics(self, edge_preds, edge_labels, logs):
        """
        Calculate TP, FP, TN, FN
        """
    
        # edge_preds needs to be already after a sigmoid
        sigmoid = nn.Sigmoid()
        edge_preds = sigmoid(edge_preds)
        preds = (edge_preds.view(-1) > 0.5).float()

        # Metrics
        TP, FP, TN, FN = 0, 0, 0, 0
        batch_size = 1024 # Too large, need batch
        for i in range(0, len(edge_labels), batch_size):
            TP += torch.sum((edge_labels[i:i+batch_size] == 1) & (preds[i:i+batch_size] == 1)).float()
            FP += torch.sum((edge_labels[i:i+batch_size] == 0) & (preds[i:i+batch_size] == 1)).float()
            TN += torch.sum((edge_labels[i:i+batch_size] == 0) & (preds[i:i+batch_size] == 0)).float()
            FN += torch.sum((edge_labels[i:i+batch_size] == 1) & (preds[i:i+batch_size] == 0)).float()

        # Update the logs
        logs["TP"].append(TP.item())
        logs["FP"].append(FP.item())
        logs["TN"].append(TN.item())
        logs["FN"].append(FN.item())

        return logs

    def _postprocess_logs(self, logs):
        """
        Calculate accuracy, precision, recall
        """
        logs["Loss"] = statistics.mean(logs["Loss"])
        logs["TP"] = sum(logs["TP"])
        logs["FP"] = sum(logs["FP"])
        logs["TN"] = sum(logs["TN"])
        logs["FN"] = sum(logs["FN"])

        logs["Accuracy"] = (logs["TP"] + logs["TN"]) / (logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])
        logs["Recall"] = logs["TP"] / (logs["TP"] + logs["FN"]) if logs["TP"] + logs["FN"] > 0 else 0
        logs["Precision"] = logs["TP"] / (logs["TP"] + logs["FP"]) if logs["TP"] + logs["FP"] > 0 else 0
        logs["F1"] = 2*logs["TP"] / (2*logs["TP"] + logs['FP']+ logs['FN']) if (logs["TP"] + logs["FP"] +logs['FN']) > 0 else 0

        # # Verbose
        # print("     Loss: ", logs["Loss"])
        # print("     Accuracy: ", logs["Accuracy"])
        # print("     Recall: ", logs["Recall"])
        # print("     Precision: ", logs["Precision"])
        # print("     TP+FP+TN+FN: ", int((logs["TP"] + logs["FP"] + logs["TN"] + logs["FN"])))

        return logs
    
    def _plot_losses_on_wandb(self, train_logs, val_logs):
        train_loss = train_logs['Loss']
        train_accuracy = train_logs['Accuracy']
        train_precision = train_logs['Precision']
        train_recall = train_logs['Recall']
        train_f1 = train_logs['F1']

        val_loss = val_logs['Loss']
        val_accuracy = val_logs['Accuracy']
        val_precision = val_logs['Precision']
        val_recall = val_logs['Recall']
        val_f1 = val_logs['F1']

        num_epoch = len(train_loss)

        wandb.log({"Epoch": len(train_loss),
                   "Train Loss": train_loss[-1],
                   "Train Accuracy": train_accuracy[-1],
                   "Validation loss": val_loss[-1],
                   "Validation Accuracy": val_accuracy[-1],
                   })

        # losses = []
        # for i in range(num_epoch):
        #     losses.append([i+1, train_loss[i], val_loss[i]])
        # table_losses = wandb.Table(data=losses, columns=["Epoch", "Train_loss", "Val_loss"])
        # line_plot_losses = wandb.plot.line(table_losses, x='Epoch', y='', title='Losses')

        # train_metrics = []
        # for i in range(num_epoch):
        #     losses.append([i+1, train_accuracy[i], train_precision[i], train_recall[i], train_f1[i]])
        # table_losses = wandb.Table(data=losses, columns=["Epoch", "Accuracy", "Val_loss"])
        # line_plot_losses = wandb.plot.line(table_losses, x='step', y='height', title='Line Plot')


        # wandb.log({"loss" : line_plot_losses})
