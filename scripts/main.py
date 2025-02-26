import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import mot17_config, kitti_config, refer_kitti_config
from src.utils.deterministic import make_deterministic
from src.tracker.hicl_tracker import HICLTracker
from src.tracker.node_classfier import NodeClassifier
from src.data.splits import get_seqs_from_splits
import os.path as osp
from TrackEval.scripts.run_mot_challenge import evaluate_mot17
from TrackEval.scripts.run_kitti import evaluate_kitti
from TrackEval.scripts.run_refer_kitti import evaluate_refer_kitti
import wandb

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
        # WANDB
        wandb.login(key='510212463700ec0ca7fbca1d123da47705710f0c')
        wconfig={
            "input_dim": config.mlp_input_dim,
            "hidden_mlp_size": config.mlp_fc_dims,
            "drop_out": config.mlp_drop_out,
            "lr": config.mlp_lr,
            'weight_decay': config.mlp_weight_decay,
            "batch_norm": config.mlp_batch_norm,
            "gpu": config.device,
            "batch_size": config.mlp_num_batch,
        }
        wandb.init(
            # set the wandb project where this run will be logged
            project="sushi",
            name=osp.basename(config.experiment_path),

            # track hyperparameters and run metadata
            # config=wconfig
            config=config
        )

        # Get the splits for the experiment
        seqs, splits = get_seqs_from_splits(data_path=config.data_path, train_split=config.train_splits[0], val_split=config.val_splits[0])
        # Initialize the tracker

        # Train the tracker
        if config.sushi_mode == 'train':
            hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)
            # if config.load_train_ckpt:
            #     print("Loading checkpoint from ", config.hicl_model_path)
            #     hicl_tracker.model = hicl_tracker.load_pretrained_model()
            hicl_tracker.train()

            print("##############")
            print("Finish training SUSHI, start training MLP...")
            print("##############")

        # TODO Train Node classifier
        if config.node_clasifier_mode == 'train':
            node_classifier = NodeClassifier(config=config, seqs=seqs, splits=splits)
            node_classifier.train()

        # # Track again
        # if config.sushi_mode == 'pre-train':
        #     hicl_tracker = HICLTracker(config=config, seqs=seqs, splits=splits)
        # output_path = osp.join(config.experiment_path, "Tracking_files")
        # hicl_tracker.track_with_mlp(dataset=hicl_tracker.val_dataset, output_path=output_path)
        # eval_func(tracker_path=output_path,
        #             split=hicl_tracker.val_split, data_path=config.data_path, tracker_sub_folder=config.mot_sub_folder,
        #             output_sub_folder=config.mot_sub_folder,text=hicl_tracker.val_dataset.text_dicts)

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
