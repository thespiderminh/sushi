import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import mot17_config, kitti_config, refer_kitti_config
from src.utils.deterministic import make_deterministic
from src.tracker.hicl_tracker import HICLTracker
from src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
from TrackEval.scripts.run_kitti import evaluate_kitti
from TrackEval.scripts.run_refer_kitti import evaluate_refer_kitti

_EVAL_FUNC = {'mot17': evaluate_mot17, 'kitti': evaluate_kitti, 'refer': evaluate_refer_kitti}
_CONFIG_FILE = {'mot17': mot17_config, 'kitti': kitti_config, 'refer': refer_kitti_config}

if __name__ == "__main__":
    run_id = os.getenv('RUN')
    config = _CONFIG_FILE[run_id[:5]].get_arguments()
    make_deterministic(config.seed)  # Make the experiment deterministic
    eval_func = _EVAL_FUNC[config.run_id[:5]]

    # # Print the config and experiment id
    # print("Experiment ID:", config.experiment_path)
    # print("Experiment Mode:", config.experiment_mode)
    # print("----- CONFIG -----")
    # for key, value in vars(config).items():
    #     print(key, ':', value)
    # print("------------------")

    # TRAINING
    if config.experiment_mode == 'train':
        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=config.train_splits[0], val_split=config.val_splits[0])
        # With mot17-train-all
        # seqs =  {'train': {'datasets/MOT17/train':
        #                    ['MOT17-02-SDP',
        #                     'MOT17-04-SDP',
        #                     'MOT17-05-SDP',
        #                     'MOT17-09-SDP',
        #                     'MOT17-10-SDP',
        #                     'MOT17-11-SDP',
        #                     'MOT17-13-SDP']},
        #           'val': {'datasets/MOT17/train':
        #                    ['MOT17-02-SDP',
        #                     'MOT17-04-SDP',
        #                     'MOT17-05-SDP',
        #                     'MOT17-09-SDP',
        #                     'MOT17-10-SDP',
        #                     'MOT17-11-SDP',
        #                     'MOT17-13-SDP']}}
        # splits =  ('mot17-train-all', 'mot17-train-all', None)

        # Initialize the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

        if config.load_train_ckpt:
            print("Loading checkpoint from ", config.hicl_model_path)
            hicl_tracker.model = hicl_tracker.load_pretrained_model()

        # Train the tracker
        hicl_tracker.train()

    # CROSS-VALIDATION
    elif config.experiment_mode == 'train-cval':
        # Each training/validation split
        for train_split, val_split in zip(config.train_splits, config.val_splits):
            # Get the splits for the experiment
            seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=train_split, val_split=val_split)
            # Initialize the tracker
            hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)
            # Train the tracker
            hicl_tracker.train()
            print("####################")

        # Evaluate the performance of oracles and each epoch
        eval_func(tracker_path=osp.join(config.experiment_path, 'oracle'), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)
        for e in range(1, config.num_epoch+1):
            eval_func(tracker_path=osp.join(config.experiment_path, 'Epoch' + str(e)), split=config.cval_seqs, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder, output_sub_folder=config.mot_sub_folder)

    # TESTING
    elif config.experiment_mode == 'test':
        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, test_split=config.test_splits[0])

        # Initialize the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

        # Load the pretrained model
        hicl_tracker.model = hicl_tracker.load_pretrained_model()

        # Track
        epoch_val_logs, epoc_val_logs_per_depth = hicl_tracker.track(dataset=hicl_tracker.test_dataset, output_path=osp.join(hicl_tracker.config.experiment_path, 'test'),
                                                                     mode='test',
                                                                     oracle=False)
               
        # Only works if you are testing on train or val data. Will fail in case of a test set
        eval_func(tracker_path=osp.join(hicl_tracker.config.experiment_path, 'test'), split=hicl_tracker.test_split,
                   data_path=hicl_tracker.config.data_path,
                   tracker_sub_folder=hicl_tracker.config.mot_sub_folder,
                   output_sub_folder=hicl_tracker.config.mot_sub_folder)
