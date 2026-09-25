import argparse
import os

import yaml

from genrec.utils import get_pipeline, parse_command_line_args

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'configs', 'experiments.yaml')


def load_experiment(name, path):
    """Load one experiment block of configs/experiments.yaml.

    A legacy top-level `defaults` section, if present, is merged underneath.
    """
    with open(path, 'r') as f:
        spec = yaml.safe_load(f)
    experiments = spec.get('experiments', {})
    if name not in experiments:
        raise ValueError(f'Unknown experiment "{name}". '
                         f'Available: {", ".join(sorted(experiments))}')
    config = dict(spec.get('defaults', {}))
    config.update(experiments[name] or {})
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        type=str,
                        default=None,
                        help='Experiment name in configs/experiments.yaml '
                        '(beauty, sports, toys, scientific, instrument, games, '
                        'arts, yelp)')
    parser.add_argument('--experiment_file',
                        type=str,
                        default=EXPERIMENTS,
                        help='Path to the experiment configuration file')
    parser.add_argument('--model', type=str, default='SPRINT', help='Model name')
    parser.add_argument('--dataset',
                        type=str,
                        default='AmazonReviews2014',
                        help='Dataset name')
    return parser.parse_known_args()


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args, unparsed_args = parse_args()
    config = {}
    if args.experiment:
        config = load_experiment(args.experiment, args.experiment_file)
    # the command line always wins over the experiment file
    config.update(parse_command_line_args(unparsed_args))

    model_name = config.pop('model', args.model)
    dataset_name = config.pop('dataset', args.dataset)

    pipeline = get_pipeline(model_name)(model_name=model_name,
                                        dataset_name=dataset_name,
                                        config_dict=config)
    pipeline.run()
