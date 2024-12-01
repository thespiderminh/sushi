import pandas as pd
import torch
from src.data.seq_processor import ReferKITTISeqProcessor
from src.data.augmentation import GraphAugmentor
import numpy as np
import os.path as osp
from src.data.graph import HierarchicalGraph
from collections import OrderedDict
from torch.nn import functional as F


class ReferKITTISceneDataset:
    """
    Main dataset class
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
        # Do 512 frame sẽ được sử dụng trong 1 graph, nên phải lấy những tập chứa đúng 512 frames có thể đưa vào
        # Mỗi lần sẽ cách ra 1 khoảng config.train_dataset_frame_overlap (default: 20)
        # (VD: (1, 512), (21, 532), (41, 552), ...)
        self.seq_with_frames_and_text = self._index_dataset()

        # Sparse index per sequence for val and test datasets
        # Tương tự như _index_dataset, tuy nhiên mỗi tập sẽ chồng lấn lên nhau một tỉ lệ bằng config.evaluation_graph_overlap_ratio (default: 0.5)
        if self.mode in ('val', 'test'):
            self.sparse_frames_per_seq = self._sparse_index_dataset()
            

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
                seq_processor = ReferKITTISeqProcessor(dataset_path=dataset_path, seq_name=seq_name, config=self.config)
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
        """
        Index the dataset in a form that we can sample
        """
        seq_with_frames_and_text = []
        # Loop over the scenes
        for scene in self.seq_names:
            # Get scene specific dataframe
            scene_df = self.seq_det_dfs[scene]
            text_dict = self.text_dicts[scene]
            frames_per_graph = self.config.frames_per_graph # default: 512: Số frame được đưa vào 1 graph

            # Scene specific values
            frames = list(OrderedDict.fromkeys(scene_df['frame'].to_list())) # Các frame có sử dụng (VD: 1,2,3,...,600)
            start_frames = []
            end_frames = []

            # Loop over all frames
            for f in frames:
                # Nếu chưa có j trong start_frames, hoặc f > frame cuối cộng với train_dataset_frame_overlap
                if not start_frames or f >= start_frames[-1] + self.config.train_dataset_frame_overlap:
                    valid_frames = np.arange(f, f + frames_per_graph)
                    graph_df = scene_df[scene_df.frame.isin(valid_frames)].copy()
                    # Each frame can be a start and end frame only once. To prevent (1, 30), (2, 30) ... (29, 30)
                    if (graph_df.frame.min() not in start_frames) and (graph_df.frame.max() not in end_frames) and (
                            len(graph_df.frame.unique()) >= 2):
                        for text in text_dict.keys():
                            seq_with_frames_and_text.append((scene, graph_df.frame.min(), graph_df.frame.max(), text))
                            start_frames.append(graph_df.frame.min())
                            end_frames.append(graph_df.frame.max())

        return tuple(seq_with_frames_and_text)

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

    def _load_precomputed_embeddings(self, det_df, seq_info_dict, embeddings_dir):
        """
        Load the embeddings corresponding to the detections specified in the det_df
        """
        # Retrieve the embeddings we need from their corresponding locations
        training_path = osp.dirname(osp.dirname(seq_info_dict['seq_path']))
        embeddings_path = osp.join(training_path, 'processed_refer_data', seq_info_dict['seq'][-4:], 'embeddings',
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

    def get_df_from_seq_and_frames(self, seq_name, start_frame, end_frame):
        """
        Returns a dataframe and a seq_info_dict belonging to the specified sequence range
        """
        # Load the corresponding part of the dataframe
        seq_det_df = self.seq_det_dfs[seq_name]  # Sequence specific dets: Df chứa thông tin về các detection trong video
        seq_info_dict = self.seq_info_dicts[seq_name]  # Sequence info dict: Df chứa thông tin về video
        valid_frames = np.arange(start_frame, end_frame + 1)  # Frames to be processed together
        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()  # Take only valid frames
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)  # Sort

        return graph_df, seq_info_dict

    def get_graph_from_seq_and_frames(self, seq_name, start_frame, end_frame, text):
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
        x_reid = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                   embeddings_dir=self.config.reid_embeddings_dir)
        
        x_node = self._load_precomputed_embeddings(det_df=graph_df, seq_info_dict=seq_info_dict,
                                                    embeddings_dir=self.config.node_embeddings_dir)


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
        x_reid = x_reid[:, 1:]
        x_node = x_node[:, 1:]
        x_frame = x_frame[:, 1:]
        x_bbox = x_bbox[:, 1:]
        x_center = x_bbox[:, :2] + 0.5* x_bbox[:, 2:]
        x_feet = x_feet[:, 1:]
        y_id = y_id[:, 1:]

        if self.config.l2_norm_reid:
            x_reid = F.normalize(x_reid, dim = -1, p=2)
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
        hierarchical_graph = HierarchicalGraph(x_reid=x_reid.float(), x_node=x_node.float(), x_frame=x_frame.long(),
                                               x_bbox=x_bbox.float(), x_feet=x_feet.float(), x_center=x_center.float(), 
                                               y_id=y_id.long(), fps=fps.long(), frames_total=frames_total.long(),
                                               frames_per_level=frames_per_level.long(), 
                                               start_frame=start_frame.long(), end_frame=end_frame.long())

        return hierarchical_graph

    def __len__(self):
        return len(self.seq_with_frames_and_text)

    def __getitem__(self, ix):
        seq_name, start_frame, end_frame, text = self.seq_with_frames_and_text[ix]
        return self.get_graph_from_seq_and_frames(seq_name=seq_name, start_frame=start_frame, end_frame=end_frame, text=text)