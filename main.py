
import yaml
import warnings
from datapipeline import load_dataset
from simulator import FederatedSimulator
from trainer import normal_local_train
import argparse


def run(config_dir):
    warnings.filterwarnings("ignore")
    cfg = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    trainset, valset, testset = load_dataset(cfg)
    simulator = FederatedSimulator(
        cfg=cfg,
        trainset=trainset,
        valset=valset,
        testset=testset,
        local_train_fn=normal_local_train)
    simulator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Federated Simulator with a configuration file.')
    parser.add_argument('config_dir', type=str, help='Path to the configuration file')

    args = parser.parse_args()
    run(args.config_dir)

    
