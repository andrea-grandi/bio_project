import pandas as pd
import argparse
import os
import tensorflow as tf
import numpy as np
import gc
import random

from models.apkmil import APKMIL # A-Priori Knowledge Multiple Instance Learning


def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS']='1'
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train APKMIL')
    parser.add_argument('--save_dir', dest='save_dir',help='directory where the weights of the model are stored',default="saved_models", type=str)
    parser.add_argument('--lr', dest='init_lr',help='initial learning rate',default=0.0002, type=float)
    parser.add_argument('--decay', dest='weight_decay',help='weight decay',default=1e-5, type=float)
    parser.add_argument('--thresh',  help='thresh', default=0.5, type=float)
    parser.add_argument('--epochs', dest='epochs',help='number of epochs to train CAMIL',default=30, type=int)
    parser.add_argument('--seed_value', dest='seed_value',help='use same seed value for reproducability',default=12321, type=int)
    parser.add_argument('--feature_path', dest='feature_path',help='directory where the images are stored',default="h5_files", type=str)
    parser.add_argument('--dataset_path', dest='data_path',help='directory where the images are stored',default="slides",type=str)
    parser.add_argument('--experiment_name', dest='experiment_name',help='the name of the experiment needed for the logs', default="test", type=str)
    parser.add_argument('--input_shape', dest="input_shape",help='shape of the image',default=(512,), type=int, nargs=3)
    parser.add_argument('--label_file', dest="label_file",help='csv file with information about the labels',default="label_files/camelyon_data.csv",type=str)
    parser.add_argument('--csv_file', dest="csv_file", help='csv file with information about the labels',default="camelyon_csv_splits/splits_0.csv",type=str)
    parser.add_argument('--raw_save_dir', dest="raw_save_dir", help='directory where the attention weights are saved', default="heatmaps", type=str)
    parser.add_argument('--retrain', dest="retrain", action='store_true', default=False)
    parser.add_argument('--save_exp_code', type=str, default=None,help='experiment code')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    parser.add_argument('--subtyping', dest="subtyping",action='store_true', default=False)
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--test', action='store_true', default=False, help='test only')

    args = parser.parse_args()

    print('Called with args:')
    print(args)

    adj_dim = None
    set_seed(12321)

    csv_file=args.csv_file
    acc=[]
    recall=[]
    f_score=[]
    auc=[]
    precision=[]

    fold_id = os.path.splitext(csv_file)[0].split("_")[3]
    references = pd.read_csv(csv_file)
    train_bags = references["train"].apply(lambda x: os.path.join(args.feature_path, x + ".h5")).values.tolist()

    def func_val(x):
        value = None
        if isinstance(x, str):
            value = os.path.join(args.feature_path, x + ".h5")
        return value

    val_bags = references.apply(lambda row: func_val(row.val), axis=1).dropna().values.tolist()
    test_bags = references.apply(lambda row: func_val(row.test), axis=1).dropna().values.tolist()

    if not args.test:
        train_net = APKMIL(args)
        train_net.train(train_bags, fold_id, val_bags, args)
    else:
        test_net = APKMIL(args)
        test_acc, test_auc = test_net.predict(test_bags, 
                                              fold_id, 
                                              args,
                                              test_model=test_net.model
                                              )