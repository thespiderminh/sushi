import pandas as pd
import torch
from src.data.seq_processor import ReferDanceTrackSeqProcessor
from src.data.augmentation import GraphAugmentor
import numpy as np
import os.path as osp
from src.data.graph import HierarchicalGraph
from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import CLIP.clip as clip

# làm 1 cái dict để lưu cache
sushi_node_dict = {}
clip_det_dict = {}
clip_text_dict = {}

def check_in_sushi_node_dict(seq_name, start_frame, end_frame):
    if seq_name in sushi_node_dict:
        if start_frame in sushi_node_dict[seq_name]:
            if end_frame in sushi_node_dict[seq_name][start_frame]:
                return True
    return False

def check_in_clip_det_dict(seq_name, start_frame, end_frame):
    if seq_name in clip_det_dict:
        if start_frame in clip_det_dict[seq_name]:
            if end_frame in clip_det_dict[seq_name][start_frame]:
                return True
    return False

def check_in_clip_text_dict(text):
    if text in clip_text_dict:
        return True
    return False

def add(dict, keys, value):
    d = dict
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

class ReferDanceSceneDataset:
    """
    Dataset class for MLP
    """
    def __init__(self, config, seqs, mode):
        assert mode in ('train', 'val', 'test'), "Dataset mode is not valid!"
        self.config = config
        self.seqs = seqs
        self.mode = mode

        # Load all dataframes
        # :seq_det_dfs: Df của các detections (top, left, height, width, in_what_frame)
        # :seq_info_dicts: Df của các thông tin về video (tên seq, seq_path, det_file_name, frame_height, frame_width, fpx,...)
        # :seq_names: Tên của seq (VD: 'MOT17-02-SDP')
        # Đồng thời tạo mục processed_data trong dataset, lưu những data đã xử lý
        self.seq_det_dfs, self.seq_info_dicts, self.seq_names, self.text_dicts = self._load_seq_dfs()

        # Index dataset
        self.seq_with_frame_and_texts = self._index_dataset()

        # Sparse index per sequence for val and test datasets
        if self.mode in ('val', 'test'):
            self.sparse_frames_per_seq = self._sparse_index_dataset()
            for seq in self.seq_det_dfs:
                df = self.seq_det_dfs[seq]
                self.seq_det_dfs[seq] = df[df['id'] != -1]

        # Calculate the ratio between selected nodes and unselected nodes
        self.N_pos, self.N_neg = self._calculate_ratio()

    def _load_seq_dfs(self):
        """
        Load the dataframes of the sequences to be used
        """

        # Initialize empty vars
        seq_names, seq_info_dicts, seq_det_dfs, text_dicts = [], {}, {}, {}

        # Loop over the seqs to retrieve
        for dataset_path, seq_list in self.seqs.items():
            for seq_name in seq_list:
                # Process or load the sequence df
                seq_processor = ReferDanceTrackSeqProcessor(dataset_path=dataset_path, seq_name=seq_name, config=self.config)
                # Nếu đã xử lý thì load, ko thì xử lý
                # trong processed_data chứa embedding của appearance sau khi đã qua fast-reid
                seq_det_df, text_dict = seq_processor.load_or_process_detections()

                # Accumulate
                seq_names.append(seq_name)
                seq_info_dicts[seq_name] = seq_det_df.seq_info_dict
                seq_det_dfs[seq_name] = seq_det_df
                text_dicts[seq_name] = text_dict

        assert len(seq_det_dfs) and len(seq_info_dicts) and len(seq_det_dfs), "No detections to process in the dataset"
        return seq_det_dfs, seq_info_dicts, seq_names, text_dicts

    def _index_dataset(self):
        seq_with_frame_and_texts = []
        # Loop over the scenes
        for scene in self.seq_names:
            # Get scene specific dataframe
            scene_df = self.seq_det_dfs[scene]
            text_dict = self.text_dicts[scene]
            frames_per_graph = self.config.frames_per_graph

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list())) # Các frame có sử dụng (VD: 1,2,3,...,600)
            start_frames = []
            end_frames = []

            # Loop over all frames
            for f in frames:
                if not start_frames or f >= start_frames[-1] + 200:
                    valid_frames = np.arange(f, f + frames_per_graph)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        for text in text_dict:
                            seq_with_frame_and_texts.append((scene, graph_df.frame.min(), graph_df.frame.max(), text))
                            start_frames.append(graph_df.frame.min())
                            end_frames.append(graph_df.frame.max())

        return tuple(seq_with_frame_and_texts)

    def _sparse_index_dataset(self):
        """
        Overlapping samples used for validation and test. This time we create a dictionary and bookkeep the sequence name
        """
        sparse_frames_per_seq = {}
        frames_per_graph = self.config.frames_per_graph# default: 512
        overlap_ratio = self.config.evaluation_graph_overlap_ratio # default: 0.5

        for scene in self.seq_names:
            scene_df = self.seq_det_dfs[scene]
            text_dict = self.text_dicts[scene]
            sparse_frames = []

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list()))
            start_frames = []
            end_frames = []

            min_frame = scene_df.frame.min()  # Initializer: Frame có chỉ số nhỏ nhất trong các frame (Thường là 1 nếu có detection ngay từ frame đầu)

            # Continue until all frames are processed
            while len(frames):
                # Valid regions of the df
                valid_frames = np.arange(min_frame, min_frame + frames_per_graph)
                graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy() # Lấy những hàng trong Df có frame thuộc valid_frames

                # Each frame can be a start and end frame only once. To prevent (1, 30), (2, 30) ... (29, 30)
                if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                        len(graph_df.frame.unique()) >= 2):
                    # Include the sample
                    for text in text_dict:
                        sparse_frames.append((scene, graph_df.frame.min(), graph_df.frame.max(), text))

                        # Update start and end frames
                        start_frames.append(graph_df.frame.min())
                        end_frames.append(graph_df.frame.max())

                    # Update the min frame
                    current_frames = sorted(list(graph_df.frame.unique()))
                    num_current_frame = len(current_frames)
                    num_overlaps = round(overlap_ratio * num_current_frame)
                    assert num_overlaps < num_current_frame and num_overlaps > 0, "Evaluation overlap ratio leads to either all frames or no frames"
                    min_frame = current_frames[-num_overlaps]

                    # Remove current frames from the remaining frames list
                    frames = [f for f in frames if f not in current_frames]

                else:
                    current_frames = sorted(list(graph_df.frame.unique()))
                    frames = [f for f in frames if f not in current_frames]
                    min_frame = min(frames)

            # To prevent empty lists
            if sparse_frames:

                # Accumulate sparse_frames_per_seq
                sparse_frames_per_seq[scene] = tuple(sparse_frames)

        return sparse_frames_per_seq

    def _calculate_ratio(self):
        N_pos, N_neg, N_tot = 0, 0, 0
        for scene, seq_info_dict in self.seq_info_dicts.items():
            N_pos += seq_info_dict['total_valid_det']
            N_tot += seq_info_dict['total_det'] * seq_info_dict['num_text']
        N_neg = N_tot - N_pos
        return N_pos, N_neg

    def _load_precomputed_embeddings(self, det_df, seq_info_dict, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        # Retrieve the embeddings we need from their corresponding locations
        training_path = osp.dirname(osp.dirname(seq_info_dict['seq_path']))
        embeddings_path = osp.join(training_path, 'processed_refer_data', seq_info_dict['seq'][6:], 'embeddings',
                                   seq_info_dict['det_file_name'],
                                   embeddings_dir)
        # print("EMBEDDINGS PATH IS ", embeddings_path)
        frames_to_retrieve = sorted(det_df.frame.unique())
        embeddings_list = [torch.load(osp.join(embeddings_path, f"{frame_num}.pt")) for frame_num in frames_to_retrieve]
        embeddings = torch.cat(embeddings_list, dim=0)

        # First column in embeddings is the index. Drop the rows of those that are not present in det_df
        ixs_to_drop = list(set(embeddings[:, 0].int().numpy()) - set(det_df['detection_id']))
        embeddings = embeddings[~np.isin(embeddings[:, 0], ixs_to_drop)]  # Not so clean, but faster than a join
        assert_str = "Problems loading embeddings. Indices between query and stored embeddings do not match. BOTH SHOULD BE SORTED!"
        assert (embeddings[:, 0].numpy() == det_df['detection_id'].values).all(), assert_str

        #embeddings = embeddings[:, 1:]  # Get rid of the detection index (MOVED TO OUT OF THIS FUNCTION)

        return embeddings
    
    def _load_precomputed_text_embeddings(self, text, seq_info_dict, embeddings_dir):
        training_path = osp.dirname(osp.dirname(seq_info_dict['seq_path']))
        embeddings_path = osp.join(training_path, 'processed_refer_data', seq_info_dict['seq'][6:], 'embeddings',
                                   self.config.text_file, embeddings_dir)
        # print("EMBEDDINGS PATH IS ", embeddings_path)
        return torch.load(osp.join(embeddings_path, f"{text}.pt"))

    def get_df_from_seq_and_frames(self, seq_name, start_frame, end_frame, hicl_graphs=None):
        """
        Returns a dataframe and a seq_info_dict belonging to the specified sequence range
        """
        # Load the corresponding part of the dataframe
        seq_det_df = self.seq_det_dfs[seq_name]  # Sequence specific dets: Df chứa thông tin về các detection trong video
        seq_info_dict = self.seq_info_dicts[seq_name]  # Sequence info dict: Df chứa thông tin về video
        valid_frames = np.arange(start_frame, end_frame + 1)  # Frames to be processed together
        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()  # Take only valid frames
        if hicl_graphs is not None:
            det_ids = hicl_graphs[0].det_id.t()[0].tolist()
            filtered_df = graph_df[graph_df.apply(lambda row: (row['detection_id'] in det_ids), axis=1)]
            graph_df = filtered_df

        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)  # Sort

        return graph_df, seq_info_dict


    def get_graph_from_seq_and_frames_and_text(self, seq_name, start_frame, end_frame, text):
        """
        Main dataloading function. Returns a hierarchical graph belonging to the specified sequence range
        Hàm chính để xử lý các video thành dạng graph
        """

        # :graph_df: Df chứa thông tin các detection từ start_frame đến end_frame
        graph_df, seq_info_dict = self.get_df_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame)


        # Ensure that there are at least 2 frames in the sampled graph
        assert len(graph_df['frame'].unique()) > 1, "There aren't enough frames in the sampled graph. Either 0 or 1"

        # Data augmentation
        if self.mode=='train' and self.config.augmentation:
            augmentor = GraphAugmentor(graph_df=graph_df, config=self.config)
            graph_df = augmentor.augment()

        # Load appearance data
        if check_in_sushi_node_dict(seq_name, start_frame, end_frame):
            x_reid = sushi_node_dict[seq_name][start_frame][end_frame]['x_reid'].detach().clone()
            x_node = sushi_node_dict[seq_name][start_frame][end_frame]['x_node'].detach().clone()
        else:
            x_reid = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                    embeddings_dir=self.config.reid_embeddings_dir)
            add(sushi_node_dict, [seq_name, start_frame, end_frame, 'x_reid'], x_reid)
            
            x_node = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                        embeddings_dir=self.config.node_embeddings_dir)
            add(sushi_node_dict, [seq_name, start_frame, end_frame, 'x_node'], x_node)
        #################
        if check_in_clip_det_dict(seq_name, start_frame, end_frame):
            x_node_clip = clip_det_dict[seq_name][start_frame][end_frame]['x_node_clip'].detach().clone()
        else:
            x_node_clip = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                            embeddings_dir=self.config.node_embeddings_clip_dir)
            add(clip_det_dict, [seq_name, start_frame, end_frame, 'x_node_clip'], x_node_clip)
        #################
        if check_in_clip_text_dict(text):
            x_text = clip_text_dict['text'].detach().clone()
        else:
            x_text = self._load_precomputed_text_embeddings(text=text, seq_info_dict=seq_info_dict,
                                                        embeddings_dir=self.config.text_embeddings_dir)
            add(clip_det_dict, [text], x_text)
        #################


        # Copy node frames and ground truth ids from the dataframe
        x_frame = torch.tensor(graph_df[['detection_id', 'frame']].values)
        x_bbox = torch.tensor(graph_df[['detection_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']].values)
        x_feet = torch.tensor(graph_df[['detection_id', 'feet_x', 'feet_y']].values)
        y_id = torch.tensor(graph_df[['detection_id', 'id']].values)

        # Assert that order of all the loaded values are the same
        assert (x_reid[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_node[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_frame[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_bbox[:, 0].numpy() == y_id[:, 0].numpy()).all() and \
               (x_feet[:, 0].numpy() == y_id[:, 0].numpy()).all(), "Feature and id mismatch while loading"

        # Get rid of the detection id index
        det_id = x_frame[:, :1]
        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_node_clip = x_node_clip[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5* x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]

        if self.config.l2_norm_reid:
            # x_reid = F.normalize(x_reid, dim = -1, p=2)
            x_node = F.normalize(x_node, dim = -1, p=2)

        # Further important parameters to pass
        fps = torch.tensor(seq_info_dict['fps'])
        frames_total = torch.tensor(self.config.frames_per_graph)
        frames_per_level = torch.tensor(self.config.frames_per_level)
        start_frame = torch.tensor(start_frame)
        end_frame = torch.tensor(end_frame)

        # Create the object with float32 and int64 precision and send to the device
        # :x_reid: Embedding reid
        # :x_node: Embedding node
        # :x_frame: Các frame tương ứng với các detection
        # :x_bbox: Giá trị của bounding box (x, y, w, h)
        # :x_feet: Toạ độ chân của bounding box (x, y)
        # :x_center: Tâm của bbox
        # :y_id: ID của bounding box sau khi đã so với groundtruth

        # Make y_node
        batch_size = x_node.size(0)
        valid_ids = self.text_dicts[seq_name][text]['label']
        y_node = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            if str(x_frame[i].item()) in valid_ids:
                if y_id[i].item() in valid_ids[str(x_frame[i].item())]:
                    y_node[i] = 1

        x_text = x_text.expand(batch_size, -1)
        text = clip.tokenize(text)
        text = text.expand(batch_size, -1)  
        
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(), 
                                               x_text=x_text.float(), x_node_clip=x_node_clip.float(), det_id=det_id,
                                               text=text,
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               y_node=y_node.float(),
                                               frames_per_level=frames_per_level.long(), 
                                               start_frame=start_frame.long(), end_frame=end_frame.long())

        return hierarchical_graph

    def __len__(self):
        return len(self.seq_with_frame_and_texts)

    def __getitem__(self, ix):
        seq_name, start_frame, end_frame, text = self.seq_with_frame_and_texts[ix]
        return self.get_graph_from_seq_and_frames_and_text(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame, text=text)