#!/private/home/rgirdhar/.conda/envs/vidcls2/bin/python
"""Launch script to run arguments stored in txt files."""
import argparse
import getpass
import subprocess
import os
import socket
import glob
import re
import time
import operator
from multiprocessing import Process
from omegaconf import OmegaConf
import inquirer

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper

BASE_RUN_DIR = ('/checkpoint/{}/Work/FB/2020/001_VideoSSL/VidCls/'
                'Outputs/'.format(getpass.getuser()))
PRIORITY_QNAME = 'prioritylab'
# No spaces in the comment; somehow doesn't work with the spaces
PRIORITY_COMMENT = 'ICCV rebuttal'


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
    parser.add_argument('-v',
                        '--vis',
                        action='store_true',
                        help='Generate visualizations when testing')
    parser.add_argument('-t',
                        '--test',
                        action='store_true',
                        help='Run testing mode (will pick the last ckpt)')
    parser.add_argument('-b',
                        '--big_gpu',
                        action='store_true',
                        help='Run on 32gb volta')
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
    parser.add_argument('-p',
                        '--profile',
                        action='store_true',
                        help='Run with kernprof. Decorate fn with @profile')
    parser.add_argument('-r',
                        '--priority',
                        action='store_true',
                        help='Run in priority queue')
    parser.add_argument('--dev', action='store_true', help='Run in dev queue')
    parser.add_argument('-s',
                        '--scavenge',
                        action='store_true',
                        help='Run in scavenge queue')
    parser.add_argument(
        '--kill_trained',
        action='store_true',
        help='Kill runs that already have a fully trained model')
    parser.add_argument('--kill_duplicates',
                        action='store_true',
                        help='Kill duplicate runs')
    parser.add_argument('--kill_all_but_params',
                        type=str,
                        default=None,
                        help=('A comma sep list of config params that must be '
                              'satisfied for all jobs that are not killed. Eg '
                              '"train.n_fwd_times=10,fold_id=0" '))
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
    """Escape a string for running on bash.
    Based on https://stackoverflow.com/a/18935765 Just needed it for the $ used
        in variable interpolation.
    TODO(rgirdhar): Do this in a robust way.. right now it assumes escaped or
        not, and then escapes everything
    """
    # if '\$' in input_str:
    #     return input_str  # Likely is already escaped
    # escaped = input_str.translate(str.maketrans({
    #     "$": r"\$",
    # }))
    # Just putting the string in single quotes seems to be able to handle $
    # stuff too.
    return f"'{input_str}'"


def choose_single_run(clis, fpath, run_id):
    """
    clis are a list of flags provided in the config overrides file.
    Args:
        clis: List of clis from the txt file
        run_id: If known which model to run locally, the run_id of that sweep
    """
    # Check if this has been run before, then we can pick the overrides from
    # the .hydra folder. Else, will have to manually construct potential
    # combinations that will be run by hydra
    # TODO(rgirdhar): This needs to be improved.. we shouldn't just take the
    # final overrides file.. since it could be corrupted by the later runs
    # We should just use the override file to figure out the params that are
    # different across the multirun, and use the params as passed in from the
    # txt config file for everything else

    # run_id_param_dicts = get_sweep_param_from_runs(fpath)
    # if len(run_id_param_dicts) == 0:
    # 7/29/2020: Now no longer reading the config from the saved file, and only
    # using the following function to get the params from the file. This is
    # because now I use the hydra function to get the sweep params, so it pretty
    # much should give the exact config that would be run for each run number.
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
        ' PYTHONPATH=$PYTHONPATH:external/VMZ/pt/ '  # for VMZ models
        ' {} train_net.py hydra.run.dir={} ').format(
            'kernprof -l ' if args.profile else 'python ', agent_folder)
    cli += cli_stuff
    if args.test:
        cli += ' test_only=True '
    if args.dev:
        cli += ' +hydra.launcher.partition="devlab" '
    elif args.priority:
        cli += (' +hydra.launcher.partition="{}"'
                ' +hydra.launcher.comment="{}" ').format(
                    PRIORITY_QNAME, PRIORITY_COMMENT)
    elif args.scavenge:
        cli += ' +hydra.launcher.partition="scavenge" '
    elif not args.local and not args.debug:
        cli += ' +hydra.launcher.partition="learnlab" '
    if args.big_gpu:
        cli += (' +hydra.launcher.constraint="volta32gb" ')
    if args.local:
        cli += (' hydra.launcher.nodes=1 '
                ' hydra.launcher.gpus_per_node=2 '
                ' hydra/launcher=submitit_local ')
    else:
        cli += (' hydra.launcher.max_num_timeout=3 ')
    if args.debug:
        cli += (' data_train.workers=0 data_eval.workers=0 ')
    cli += ' ' + ' '.join(args.rest)
    # This must go at the end, the other args must go before
    if not args.debug:
        cli += ' -m '
    return cli


def _smart_kill_trained(args, agent_folder, slurm_master_ids):
    # TODO(rgirdhar): This was copied from phyre code.. needs to be modified
    # to work here
    to_kill = set()
    run_sweep_params = get_sweep_param_from_runs(args.cfg)
    sweep_runs = [run_id for run_id, _ in run_sweep_params]
    done_sweep_runs = []
    for folder in sweep_runs:
        if folder is None:
            # This happens when this is the first run, so can't figure out
            # run_ids. In this case, of course need to launch this job
            continue
        # The following should not happen if the run_id exists, but just
        # doing to be safe
        out_dir = f'{agent_folder}/{folder}'
        if not os.path.exists(out_dir):
            continue
        # If the folder exists, and results is stored in, that means it is
        # finished and does not need to be launched
        res_file_name = 'results-vis.json' if args.vis else 'results.json'
        if res_file_name in os.listdir(out_dir):
            done_sweep_runs.append(folder)
    to_kill.update([
        f'{slurm_id}_{run_id}' for slurm_id in slurm_master_ids
        for run_id in done_sweep_runs
    ])
    return to_kill


def check_conf_satisfied(conf, conf_const):
    """
    Check if the config constraint is satisfied by this conf.
    Args:
        conf: An OmegaConf
        conf_const: A string, with "A=B,C=D,..." format
    Returns:
        boolean: satisfied or not
    """
    if len(conf_const) == 0:
        return True
    for param_val in conf_const.split(','):
        param, val = param_val.split('=')
        # Get the value in the config
        if isinstance(conf, dict):
            # This is most likely a dict with override strings and corr values
            conf_val = conf[param]
        else:
            # Could be OmegaConf, or sth else.. try to get the element
            conf_val = operator.attrgetter(param)(conf)
        if str(conf_val) != val:
            return False
    return True


def _smart_kill_params(args, slurm_master_ids):
    """
    Kill all jobs except the ones that satisfy the params
    """
    # Only consider the job that was just submitted
    last_slurm_master_id = sorted(slurm_master_ids, key=int)[-1]
    run_sweep_params = get_sweep_param_from_runs(args.cfg)
    runs_to_kill = [
        run_id for run_id, conf in run_sweep_params
        if (not check_conf_satisfied(conf, args.kill_all_but_params)
            and run_id is not None)
    ]
    slurm_to_kill = set(
        [f'{last_slurm_master_id}_{run_id}' for run_id in runs_to_kill])
    return slurm_to_kill


def _smart_kill_duplicates(slurm_master_ids):
    to_kill = set()
    queue = subprocess.check_output('squeue -u $USER', shell=True)
    queue = str(queue).split('\\n')[1:]  #strip header
    sweep_id_jobs = {}
    single_job = re.compile(r'(\d+)_(\d+)')
    job_array = re.compile(r'(\d+)_\[.*%\d+\]')
    for job_info in queue:
        single_job_info = single_job.findall(job_info)
        job_array_info = job_array.findall(job_info)
        if len(single_job_info) > 0:
            master_id, sweep_id = single_job_info[0]
            if master_id not in slurm_master_ids:
                continue
            sweep_id_jobs[sweep_id] = sweep_id_jobs.get(sweep_id,
                                                        []) + [master_id]
        elif len(job_array_info) > 0:
            master_id = job_array_info[0]
            if master_id not in slurm_master_ids:
                continue
            job_array_items = re.compile(r'(\[.*%\d+\])')
            single = re.compile(r'[\[,](\d+)[,%]')
            ranges = re.compile(r'(\d+)-(\d+)[,%]')

            array_items = job_array_items.findall(job_info)[0]
            for each in single.findall(array_items):
                sweep_id_jobs[each] = sweep_id_jobs.get(each, []) + [master_id]
            for min_id, max_id in ranges.findall(array_items):
                for each in range(int(min_id), int(max_id) + 1):
                    sweep_id_jobs[str(each)] = sweep_id_jobs.get(
                        str(each), []) + [master_id]
    # Keep oldest job per sweep, kill rest
    sweep_id_jobs = {
        k: [f'{each}_{k}' for each in sorted(v, key=int)[1:]]
        for k, v in sweep_id_jobs.items()
    }
    for key in sweep_id_jobs:
        to_kill.update(sweep_id_jobs[key])
    return to_kill


def _smart_kill(args):
    time.sleep(20)  # Make sure jobs have launched
    slurm_ids = set()
    agent_folder = '{}/{}'.format(BASE_RUN_DIR,
                                  args.cfg if args.cfg else 'default')
    submitted_jobs = glob.glob(
        os.path.join(agent_folder, '.slurm/*_submitted.pkl'))
    slurm_master_ids = set(
        os.path.basename(el).split('_')[0] for el in submitted_jobs)

    # Kill runs for trained models
    if args.kill_trained:
        slurm_ids.update(
            _smart_kill_trained(args, agent_folder, slurm_master_ids))

    # Kill runs for duplicates
    if args.kill_duplicates:
        slurm_ids.update(_smart_kill_duplicates(slurm_master_ids))

    # Kill runs that don't specify specific params
    if args.kill_all_but_params:
        slurm_ids.update(_smart_kill_params(args, slurm_master_ids))

    # Kill jobs
    print(f'Killing the following {len(slurm_ids)} jobs: {slurm_ids}')
    subprocess.call('scancel {}'.format(' '.join(slurm_ids)), shell=True)


def main():
    """Main func."""
    args = parse_args()
    # if args.cls:
    #     args = gen_cls_override_file(args)
    cmd = construct_cmd(args)
    if args.kill_trained or args.kill_duplicates or args.kill_all_but_params:
        proc = Process(target=_smart_kill, args=(args, ))
        proc.start()
    print('>> Running "{}"'.format(cmd))
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
