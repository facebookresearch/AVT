#!~/.conda/envs/avt/bin/python
"""Launch script to run arguments stored in txt files."""
import argparse
import subprocess
import os
import socket
import glob
from omegaconf import OmegaConf
import inquirer
import pathlib

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper


CODE_DIR = str(pathlib.Path(__file__).parent.resolve())
BASE_RUN_DIR = f'{CODE_DIR}/OUTPUTS'


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--cfg',
                        type=str,
                        required=True,
                        help='Overrides config file')
    parser.add_argument('-l',
                        '--local',
                        action='store_true',
                        help='Run locally instead of launching to cluster')
    parser.add_argument('-g',
                        '--debug',
                        action='store_true',
                        help='Run in debug mode: 1 GPU, when locally')
    parser.add_argument('-t',
                        '--test',
                        action='store_true',
                        help='Run testing mode (will pick the last ckpt)')
    parser.add_argument('-p',
                        '--partition',
                        type=str,
                        default=None,
                        help='Specify SLURM partition to run on')
    parser.add_argument('--tb',
                        action='store_true',
                        help='Run tensorboard on this directory')
    parser.add_argument('-f',
                        '--fl',
                        action='store_true',
                        help='View the folder (run a python server)')
    parser.add_argument('-d',
                        '--delete',
                        action='store_true',
                        help='Delete the folder')
    parser.add_argument('-k',
                        '--kill',
                        action='store_true',
                        help='Kill jobs running this config.')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Run with kernprof. Decorate fn with @profile')
    parser.add_argument('--cls',
                        action='store_true',
                        help='Gen classification file and run that')
    parser.add_argument('--run_id',
                        type=int,
                        default=None,
                        help='Run for this specific run_id, if known')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.debug:
        args.local = True
    return args


def get_sweep_param_from_combinations(clis):
    """
    Returns:
        [(run_id, overrides_dict)]. The run_id can be None if unsure what hydra
            would use.
    """

    sweeper = BasicSweeper(max_batch_size=None)
    parser = OverridesParser.create()
    overrides = parser.parse_overrides(clis)
    run_args = sweeper.split_arguments(overrides, max_batch_size=None)[0]
    res = []
    for i, run_arg in enumerate(run_args):
        res.append((i, dict([el.split('=') for el in run_arg])))
    return res


def get_sweep_param_from_runs(conf_path):
    exp_path = os.path.join(BASE_RUN_DIR, conf_path)
    run_dirs = glob.glob(os.path.join(exp_path, r'[0-9]*'))
    if len(run_dirs) == 0:
        return []
    res = []
    for run_dir in run_dirs:
        run_id = int(os.path.basename(run_dir))
        override_fpath = os.path.join(run_dir, '.hydra/overrides.yaml')
        if not os.path.exists(override_fpath):
            # Likely deleted, so run_dirs may not be useful..
            # Happens when we delete the output folder, but the run folders
            # don't get deleted, because I opened in toplog and the nfs
            # files aren't deleted until I kill the toplog on that folder
            return []
        conf = OmegaConf.load(override_fpath)
        res.append((run_id, dict([el.split('=') for el in conf])))
    return res


def subselect_dict_keys_diff(run_id_param_dicts):
    """Select keys from the param_dicts that actually change between configs."""
    key_vals = {}
    for _, param_dict in run_id_param_dicts:
        for key, val in param_dict.items():
            if key not in key_vals:
                key_vals[key] = []
            key_vals[key].append(val)
    keys_to_keep = [
        key for key, vals in key_vals.items() if len(set(vals)) > 1
    ]
    return [(el[0], {key: el[1][key]
                     for key in keys_to_keep}) for el in run_id_param_dicts]


def escape_str(input_str):
    return f"'{input_str}'"


def choose_single_run(clis, fpath, run_id):
    """
    clis are a list of flags provided in the config overrides file.
    Args:
        clis: List of clis from the txt file
        run_id: If known which model to run locally, the run_id of that sweep
    """
    run_id_param_dicts = get_sweep_param_from_combinations(clis)

    if len(run_id_param_dicts) == 1:
        final_run_id, param_dict = run_id_param_dicts[0]
        assert run_id is None or run_id == final_run_id
    elif run_id is not None:
        final_run_id = run_id
        param_dicts = [el[1] for el in run_id_param_dicts if el[0] == run_id]
        assert len(param_dicts) == 1, 'run_id not found, or multiple found'
        param_dict = param_dicts[0]
    else:
        # Show options to the user and let her pick
        run_id_param_dicts_diff = subselect_dict_keys_diff(run_id_param_dicts)
        print('Choose from: \n' +
              '\n'.join([str(el) for el in run_id_param_dicts_diff]))
        qst = [
            inquirer.List(
                'r',
                message='Which sweep config to use?',
                choices=range(len(run_id_param_dicts)),
                carousel=True,
            ),
        ]
        final_run_id, param_dict = run_id_param_dicts[inquirer.prompt(qst)
                                                      ['r']]
    return final_run_id, [f'{key}={val}' for key, val in param_dict.items()]


def read_file_into_cli(fpath, running_local=False, run_id=None):
    """Read cli from file into a string."""
    res = []
    with open(fpath, 'r') as fin:
        for line in fin:
            args = line.split('#')[0].strip()
            if len(args) == 0:
                continue
            res.append(args)
    if running_local:
        final_run_id, res = choose_single_run(res, fpath, run_id)
    else:
        final_run_id = None  # not local, launch all, so run_id is irrelevant
    return final_run_id, res


def get_models_dir(dpath):
    """Go inside the dpath to get the model dir."""
    runs = sorted([el for el in next(os.walk(dpath))[1] if el.isdigit()])
    if len(runs) > 1:
        # Ask which run to use
        question = [
            inquirer.List(
                'run',
                message='Which run to use?',
                choices=runs,
            ),
        ]
        answers = inquirer.prompt(question)
    else:
        answers = dict(run=runs[0])
    return dpath + '/' + answers['run']


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_free_port():
    # Make sure to forward these ports in et
    potential_ports = range(30303, 30399)
    for port in potential_ports:
        if not is_port_in_use(port):
            return port
    raise ResourceWarning('No empty port found')


def num_gpus():
    output = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader'
                            '| wc -l', shell=True, capture_output=True)
    return int(output.stdout.decode().strip())


def construct_cmd(args):
    """Construct the cmd as provided in args."""
    if args.cfg:
        assert args.cfg.startswith('expts'), 'Must be wrt this directory'
    agent_folder = '{}/{}'.format(BASE_RUN_DIR,
                                  args.cfg if args.cfg else 'default')
    if args.kill:
        slurm_ids = os.listdir(os.path.join(agent_folder, '.submitit/'))
        shall = input("Kill %s (y/N) " % slurm_ids).lower() == 'y'
        if shall:
            return 'scancel {}'.format(' '.join(slurm_ids))
    if args.tb:  # Run tensorboard only
        # Clear the cli and just run tensorboard
        cli = ('cd {agent_folder}; tensorboard --logdir . --port {port} '
               '--max_reload_threads 10 --window_title {name} ').format(
                   agent_folder=agent_folder,
                   port=get_free_port(),
                   name=args.cfg)
        return cli
    if args.fl:  # Visualize the folder only
        # Clear the cli and just run tensorboard
        cli = 'cd {}; python -m http.server {}'.format(agent_folder,
                                                       get_free_port())
        return cli
    if args.delete:
        cli = 'rm -r {f}/* {f}/.*'.format(f=agent_folder)
        shall = input("Run %s (y/N) " % cli).lower() == 'y'
        if shall:
            return cli
        return ''
    # Else, it is the general train command
    run_id, cli_stuff = read_file_into_cli(args.cfg,
                                           running_local=args.local,
                                           run_id=args.run_id)
    cli_stuff = [escape_str(el) for el in cli_stuff]
    cli_stuff = ' '.join(cli_stuff)
    if args.debug:
        if args.test:
            # If args.test, then might be testing a model from other dir
            agent_folder = os.path.join(agent_folder, str(run_id))
        else:
            agent_folder = os.path.join(agent_folder, 'local')
    # Delete the sync file if it exists
    clear_cmd = f'find {agent_folder} -iname sync_file_init -delete'
    print(f'Clearing out the sync files using: {clear_cmd}')
    subprocess.call(clear_cmd, shell=True)
    cli = (
        'export NCCL_SOCKET_IFNAME=; export GLOO_SOCKET_IFNAME=; '
        ' HYDRA_FULL_ERROR=1 '
        ' {} train_net.py hydra.run.dir={} ').format(
            'kernprof -l ' if args.profile else 'python ', agent_folder)
    cli += cli_stuff
    if args.test:
        cli += ' test_only=True '
    if args.local:
        cli += (' hydra.launcher.nodes=1 '
                f' hydra.launcher.gpus_per_node={num_gpus()} '
                ' hydra/launcher=submitit_local ')
    else:
        cli += (' hydra.launcher.max_num_timeout=3 ')
    if args.partition is not None and not args.local:
        cli += f' +hydra.launcher.partition="{args.partition}" '
    if args.debug:
        cli += (' data_train.workers=0 data_eval.workers=0 ')
    cli += ' ' + ' '.join(args.rest)
    # This must go at the end, the other args must go before
    if not args.debug:
        cli += ' -m '
    return cli


def main():
    """Main func."""
    args = parse_args()
    # if args.cls:
    #     args = gen_cls_override_file(args)
    cmd = construct_cmd(args)
    print('>> Running "{}"'.format(cmd))
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
