#supervisor code
import subprocess
import time
import sys
import os
import signal
import cv2
import numpy as np
import io
from flask import Flask, request, send_file
from flask_cors import CORS
from mpi4py import MPI
import boto3
import json
import uuid

NUM_WORKERS = 4
WORKER_SCRIPT = 'worker.py'
WORKER_NODES = ['worker1', 'worker2', 'worker3', 'worker4']
global workers

def signal_handler(sig, frame):
    print("Termination signal received in supervisor. Shutting down...")
    for worker_process in workers.values():
        stop_worker(worker_process)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_worker(worker_id, worker_node):
    command = f"ssh {worker_node} 'python3 Phase4/{WORKER_SCRIPT} {worker_id}'"
    try:
        print(f"Worker {worker_node} attempting to start")
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        print(f"Worker {worker_node} started successfully")
        return process
    except Exception as e:
        print(f"Failed to start worker {worker_id} on node {worker_node}: {e}")
        return None

def stop_worker(worker_process):
    try:
        os.killpg(os.getpgid(worker_process.pid), signal.SIGTERM)
    except:
        pass

def kill_worker_processes(worker_node):
    command = f"ssh {worker_node} 'pkill -f {WORKER_SCRIPT}'"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"Successfully killed worker processes on node {worker_node}")
        else:
            print(f"Failed to kill worker processes on node {worker_node}")
    except subprocess.CalledProcessError as e:
        print(f"Error killing worker processes on node {worker_node}: {e}")

def restart_worker(worker_process, worker_id, worker_node):
    kill_worker_processes(worker_process)
    return start_worker(worker_id, worker_node)

def start_supervisor():
    print("Supervisor started, checking workers...")
    global workers
    workers = {}

    for i in range(1, NUM_WORKERS + 1):
        worker_node = WORKER_NODES[i - 1]
        workers[i] = start_worker(i, worker_node)
    try:
        while True:
            for worker_id, worker_process in workers.items():
                if worker_process.poll() is not None:
                    print(f"Worker {worker_id} on {WORKER_NODES[worker_id - 1]} failed. Restarting...")
                    workers[worker_id] = restart_worker(worker_process, worker_id, WORKER_NODES[worker_id - 1])
            #time.sleep(2)
            
    except KeyboardInterrupt:
        print("Shutting down supervisor...")
        for worker_process in workers.values():
            stop_worker(worker_process)
    except Exception as e:
        print("xd")
        for worker_process in workers.values():
            stop_worker(worker_process)
        sys.exit(0)

if __name__ == '__main__':
    start_supervisor()