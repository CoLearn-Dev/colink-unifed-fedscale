import os
import json
import sys
import subprocess
import tempfile
from typing import List
import time
import datetime
import yaml
import socket
import pickle
import flbenchmark.datasets
import flbenchmark.logging

import colink as CL

from unifed.frameworks.fedscale.util import store_error, store_return, GetTempFileName, get_local_ip

pop = CL.ProtocolOperator(__name__)
UNIFED_TASK_DIR = "unifed:task"

def load_FedScale_data(config_dataset_name):
    out_dir_base='~/flbenchmark.working/data'
    flbd = flbenchmark.datasets.FLBDatasets(f'{out_dir_base}')

    dataset_name = ('student_horizontal','breast_horizontal','default_credit_horizontal','give_credit_horizontal','vehicle_scale_horizontal')

    for x in dataset_name:
        if config_dataset_name == x:
            train_dataset, test_dataset = flbd.fateDatasets(x)
            flbenchmark.datasets.convert_to_csv(train_dataset, out_dir=f'{out_dir_base}/csv_data/{x}/train')
            if x != 'vehicle_scale_horizontal':
                flbenchmark.datasets.convert_to_csv(test_dataset, out_dir=f'{out_dir_base}/csv_data/{x}/test')

    vertical = ('breast_vertical','give_credit_vertical','default_credit_vertical')
    for x in vertical:
        if config_dataset_name == x:
            my_dataset = flbd.fateDatasets(x)
            flbenchmark.datasets.convert_to_csv(my_dataset[0], out_dir=f'{out_dir_base}/csv_data/{x}')
            if my_dataset[1] != None:
                flbenchmark.datasets.convert_to_csv(my_dataset[1], out_dir=f'{out_dir_base}/csv_data/{x}')

    leaf = ('femnist','reddit','celeba')
    for x in leaf:
        if config_dataset_name == x:
            my_dataset = flbd.leafDatasets(x)
    
def load_config_from_param_and_check(param: bytes):
    unifed_config = json.loads(param.decode())
    framework = unifed_config["framework"]
    assert framework == "fedscale"
    deployment = unifed_config["deployment"]
    if deployment["mode"] != "colink":
        raise ValueError("Deployment mode must be colink")
    return unifed_config

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def config_to_FedScale_format(origin_json_conf):
    json_conf = dict()
    json_conf["framework"] = "FedScale"
    json_conf["dataset"] = origin_json_conf["dataset"]
    if origin_json_conf["algorithm"] == "fedavg":
        json_conf["algorithm"] = "fed_avg"
    json_conf["model"] = origin_json_conf["model"]
    json_conf["bench_param"] = {"mode": "local","device": "gpu"}
    json_conf["training_param"] = origin_json_conf["training"]
    json_conf["data_dir"] = "~/flbenchmark.working/data"
    
    load_FedScale_data(json_conf["dataset"])
    return json_conf

def load_json_conf(json_file):
    with open(json_file) as fin:
        data = json.load(fin)
    return data

def process_cmd_server(json_conf, server_ip, local=False):
    yaml_conf = {'ps_ip': '', 'ps_port': 29664, 'worker_ips': ['localhost:[2]'], 'exp_path': '$FEDSCALE_HOME/fedscale/cloud', 'executor_entry': 'execution/executor.py', 'aggregator_entry': 'aggregation/aggregator.py', 'auth': {'ssh_user': '', 'ssh_private_key': '~/.ssh/id_rsa'}, 'job_conf': [{'job_name': ''}, {'seed': 1}, {'log_path': './benchmark'}, {'task': 'simple'}, {'num_participants': 2}, {'data_set': 'breast_horizontal'}, {'data_dir': ''}, {'model': 'logistic_regression'}, {'gradient_policy': 'fed-avg'}, {'eval_interval': 5}, {'rounds': 6}, {'filter_less': 1}, {'num_loaders': 2}, {'local_steps': 5}, {'inner_step': 1}, {'learning_rate': 0.01}, {'batch_size': 32}, {'test_bsz': 32}, {'use_cuda': False}]}

    print("process_cmd_server start")

    ps_ip = server_ip
    ps_port = yaml_conf['ps_port']
    worker_ips, total_gpus = [], []
    max_process = min(4, json_conf["training_param"]["client_per_round"])

    executor_configs = "=".join(yaml_conf['worker_ips']).split(':')[0] + f':[{max_process}]'
    if 'worker_ips' in yaml_conf:
        for ip_gpu in yaml_conf['worker_ips']:
            ip, gpu_list = ip_gpu.strip().split(':')
            worker_ips.append(ip)
            # total_gpus.append(eval(gpu_list))
            total_gpus.append([max_process])

    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                'ps_port': ps_port,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''

    for conf_name in job_conf:
        if conf_name == "job_name":
            job_conf[conf_name] = json_conf["dataset"] + '+' + json_conf["model"]
        elif conf_name == "task":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = 'cv'
            else:
                job_conf[conf_name] = "simple" # TO-DO ?
        elif conf_name == "num_participants":
            job_conf[conf_name] = json_conf["training_param"]["client_per_round"]
        elif conf_name == "data_set":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = 'femnist2'
            else:
                job_conf[conf_name] = json_conf["dataset"]
        elif conf_name == "data_dir":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = json_conf["data_dir"] + "/" + json_conf["dataset"]
            else:
                job_conf[conf_name] = json_conf["data_dir"] + "/csv_data/" + json_conf["dataset"]
        elif conf_name == "model":
            job_conf[conf_name] = json_conf["model"]
        elif conf_name == "gradient_policy":
            job_conf[conf_name] = json_conf["algorithm"]
        elif conf_name == "eval_interval":
            job_conf[conf_name] = 1 # json_conf["training_param"]["epochs"] 
        elif conf_name == "rounds":
            job_conf[conf_name] = json_conf["training_param"]["epochs"] + 1
        elif conf_name == "inner_step":
            job_conf[conf_name] = json_conf["training_param"]["inner_step"]
        elif conf_name == "learning_rate":
            job_conf[conf_name] = json_conf["training_param"]["learning_rate"]
        elif conf_name == "batch_size":
            job_conf[conf_name] = json_conf["training_param"]["batch_size"]
        elif conf_name == "use_cuda":
            job_conf[conf_name] = (json_conf["bench_param"]["device"] == "gpu")

        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    if json_conf['dataset'] == 'femnist':
        conf_script = conf_script + ' --temp_tag=simple_femnist'

    print(conf_script)

    total_gpu_processes = sum([sum(x) for x in total_gpus])

    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    print(f"Starting aggregator on {ps_ip}...")
    ps_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['aggregator_entry']} {conf_script} --this_rank=0 --num_executors={total_gpu_processes} --executor_configs={executor_configs} "

    return time_stamp, ps_cmd

def process_cmd_client(participant_id, json_conf, time_stamp, server_ip):

    yaml_conf = {'ps_ip': '', 'ps_port': 29664, 'worker_ips': ['localhost:[2]'], 'exp_path': '$FEDSCALE_HOME/fedscale/cloud', 'executor_entry': 'execution/executor.py', 'aggregator_entry': 'aggregation/aggregator.py', 'auth': {'ssh_user': '', 'ssh_private_key': '~/.ssh/id_rsa'},  'job_conf': [{'job_name': ''}, {'seed': 1}, {'log_path': './benchmark'}, {'task': 'simple'}, {'num_participants': 2}, {'data_set': 'breast_horizontal'}, {'data_dir': ''}, {'model': 'logistic_regression'}, {'gradient_policy': 'fed-avg'}, {'eval_interval': 5}, {'rounds': 6}, {'filter_less': 1}, {'num_loaders': 2}, {'local_steps': 5}, {'inner_step': 1}, {'learning_rate': 0.01}, {'batch_size': 32}, {'test_bsz': 32}, {'use_cuda': False}]}


    ps_ip = server_ip
    ps_port = yaml_conf['ps_port']
    worker_ips, total_gpus = [], []
    max_process = min(4, json_conf["training_param"]["client_per_round"])

    if 'worker_ips' in yaml_conf:
        for ip_gpu in yaml_conf['worker_ips']:
            ip, gpu_list = ip_gpu.strip().split(':')
            worker_ips.append(ip)
            # total_gpus.append(eval(gpu_list))
            total_gpus.append([max_process])

    running_vms = set()
    job_name = 'fedscale_job'

    job_conf = {'time_stamp': time_stamp,
                'ps_ip': ps_ip,
                'ps_port': ps_port,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''

    for conf_name in job_conf:
        if conf_name == "job_name":
            job_conf[conf_name] = json_conf["dataset"] + '+' + json_conf["model"]
        elif conf_name == "task":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = 'cv'
            else:
                job_conf[conf_name] = "simple" # TO-DO ?
        elif conf_name == "num_participants":
            job_conf[conf_name] = json_conf["training_param"]["client_per_round"]
        elif conf_name == "data_set":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = 'femnist2'
            else:
                job_conf[conf_name] = json_conf["dataset"]
        elif conf_name == "data_dir":
            if json_conf['dataset'] == 'femnist':
                job_conf[conf_name] = json_conf["data_dir"] + "/" + json_conf["dataset"]
            else:
                job_conf[conf_name] = json_conf["data_dir"] + "/csv_data/" + json_conf["dataset"]
        elif conf_name == "model":
            job_conf[conf_name] = json_conf["model"]
        elif conf_name == "gradient_policy":
            job_conf[conf_name] = json_conf["algorithm"]
        elif conf_name == "eval_interval":
            job_conf[conf_name] = 1 # json_conf["training_param"]["epochs"] 
        elif conf_name == "rounds":
            job_conf[conf_name] = json_conf["training_param"]["epochs"] + 1
        elif conf_name == "inner_step":
            job_conf[conf_name] = json_conf["training_param"]["inner_step"]
        elif conf_name == "learning_rate":
            job_conf[conf_name] = json_conf["training_param"]["learning_rate"]
        elif conf_name == "batch_size":
            job_conf[conf_name] = json_conf["training_param"]["batch_size"]
        elif conf_name == "use_cuda":
            job_conf[conf_name] = (json_conf["bench_param"]["device"] == "gpu")

        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'

    if json_conf['dataset'] == 'femnist':
        conf_script = conf_script + ' --temp_tag=simple_femnist'

    print(conf_script)


    total_gpu_processes = sum([sum(x) for x in total_gpus])

    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)

        print(f"Starting workers on {worker} ...")

        for cuda_id in range(len(gpu)):
            for _ in range(gpu[cuda_id]):
                worker_cmd = f" python {yaml_conf['exp_path']}/{yaml_conf['executor_entry']} {conf_script} --this_rank={rank_id} --num_executors={total_gpu_processes} "
                if job_conf['use_cuda'] == True:
                    worker_cmd += f" --cuda_device=cuda:{cuda_id}"

                if rank_id == participant_id:
                    print(f"submitted: rank_id:{rank_id} worker_cmd:{worker_cmd}")
                    return worker_cmd
                rank_id += 1

    print(f"Submitted job!")

    return ''


@pop.handle("unifed.fedscale:server")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_server(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    print("start run server")
    unifed_config = load_config_from_param_and_check(param)
    Config = config_to_FedScale_format(unifed_config)
    print("Config",Config)
    # for certain frameworks, clients need to learn the ip of the server
    # in that case, we get the ip of the current machine and send it to the clients
    server_ip = get_local_ip()
    cl.send_variable("server_ip", server_ip, [p for p in participants if p.role == "client"])
    # run external program
    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    
    time_stamp, ps_cmd = process_cmd_server(Config, server_ip)

    cl.send_variable("time_stamp", json.dumps(time_stamp), [p for p in participants if p.role == "client"])

    # process = subprocess.Popen(f'FEDSCALE_HOME=$(pwd)/FedScale && echo $FEDSCALE_HOME && kill -9 1136896 1136899 1150863 1150866 1156634 1156639 1157618  && ps aux | grep feds',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    process = subprocess.Popen(f'FEDSCALE_HOME=$(pwd)/FedScale && echo $FEDSCALE_HOME && kill -9 1136896 && ps aux | grep feds',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    returncode = process.returncode

    output = stdout
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    log = stderr
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)

    with open("./log/0.log", "rb") as f:
        result = f.read()
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:result", result)

    print('res:',result)
    
    return json.dumps({
        "server_ip": server_ip,
        "stdout": output.decode(),
        "stderr": log.decode(),
        "returncode": returncode,
    })


@pop.handle("unifed.fedscale:client")
@store_error(UNIFED_TASK_DIR)
@store_return(UNIFED_TASK_DIR)
def run_client(cl: CL.CoLink, param: bytes, participants: List[CL.Participant]):
    print("start run client")
    unifed_config = load_config_from_param_and_check(param)
    Config = config_to_FedScale_format(unifed_config)
    
    server_in_list = [p for p in participants if p.role == "server"]
    assert len(server_in_list) == 1

    participant_id = [i for i, p in enumerate(participants) if p.user_id == cl.get_user_id()][0]
    p_server = server_in_list[0]


    server_ip = cl.recv_variable("server_ip", p_server).decode()    
    time_stamp = cl.recv_variable("time_stamp", p_server).decode()
    print(f"time_stamp:{time_stamp}")
    print(f"participant_id:{participant_id}")

    ps_cmd = process_cmd_client(participant_id, Config, time_stamp, server_ip)

    # if participant_id == 1:
    #     process = subprocess.Popen(f'FEDSCALE_HOME=$(pwd)/FedScale && echo $FEDSCALE_HOME && kill -9 437211 437216 437387  && ps aux | grep feds',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # else :
        # process = subprocess.Popen(f'FEDSCALE_HOME=$(pwd)/FedScale && echo $FEDSCALE_HOME && kill -9 2740895 2740898 2768498 2768504 && ps aux | grep feds',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    process = subprocess.Popen(f'FEDSCALE_HOME=$(pwd)/FedScale && echo $FEDSCALE_HOME && ps aux | grep feds',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    returncode = process.returncode

    output = stdout
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:output", output)
    log = stderr
    cl.create_entry(f"{UNIFED_TASK_DIR}:{cl.get_task_id()}:log", log)
    return json.dumps({
        "server_ip": server_ip,
        "stdout": output.decode(),
        "stderr": log.decode(),
        "returncode": returncode,
    })