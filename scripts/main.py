import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import mot17_config, kitti_config, refer_kitti_config, refer_dance_config
from src.utils.deterministic import make_deterministic
from src.tracker.hicl_tracker import HICLTracker
from src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
from TrackEval.scripts.run_kitti import evaluate_kitti
from TrackEval.scripts.run_refer_kitti import evaluate_refer_kitti
from TrackEval.scripts.run_refer_dance import evaluate_refer_dance
import wandb

_RUN_IDS = {'mot17': (evaluate_mot17, mot17_config),
            'kitti': (evaluate_kitti, kitti_config),
            'refer_kitti': (evaluate_refer_kitti, refer_kitti_config),
            'refer_dance': (evaluate_refer_dance, refer_dance_config),
            }

if __name__ == "__main__":
    run_id = os.getenv('RUN')
    for i in _RUN_IDS:
        if run_id.startswith(i):
            eval_func, config = _RUN_IDS[i]
            config = config.get_arguments()
    make_deterministic(config.seed)  # Make the experiment deterministic

    # # Print the config and experiment id
    # print("Experiment ID:", config.experiment_path)
    # print("Experiment Mode:", config.experiment_mode)
    # print("----- CONFIG -----")
    # for key, value in vars(config).items():
    #     print(key, ':', value)
    # print("------------------")

    # TRAINING
    if config.experiment_mode == 'train':
        # WANDB
        wandb.login(key='510212463700ec0ca7fbca1d123da47705710f0c')
        wandb.init(
            # set the wandb project where this run will be logged
            project="sushi",
            name=osp.basename(config.experiment_path),

            # track hyperparameters and run metadata
            config=config
        )

        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=config.train_splits[0], val_split=config.val_splits[0])
        # Initialize the tracker

        # Train the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits, run_id=run_id)
        if config.load_train_ckpt:
            print("Loading checkpoint from ", config.hicl_model_path)
            hicl_tracker.model = hicl_tracker.load_pretrained_model()
        hicl_tracker.train()

    # TESTING
    elif config.experiment_mode == 'test':
        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, test_split=config.test_splits[0])

        # Initialize the tracker
        hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)

        # Load the pretrained model
        hicl_tracker.model = hicl_tracker.load_pretrained_model()

        # Track
        output_path = osp.join(config.experiment_path, "test", "kitti_files")
        hicl_tracker.track_with_mlp(dataset=hicl_tracker.test_dataset, output_path=output_path, mode="Test", mlp_path=config.mlp_model_file)               

        # Only works if you are testing on train or val data. Will fail in case of a test set
        eval_func(tracker_path=osp.dirname(output_path), split=hicl_tracker.test_split,
                   data_path=hicl_tracker.config.data_path,
                   tracker_sub_folder=hicl_tracker.config.mot_sub_folder,
                   output_sub_folder=hicl_tracker.config.mot_sub_folder,
                   text=hicl_tracker.test_dataset.text_dicts)
