import subprocess
import signal
import os
import time
import sys

# Function to handle termination signals
def signal_handler(sig, frame):
    print('Termination signal received. Shutting down gracefully...')
    for process in processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Terminate the process group
        except OSError:
            pass
    
    time.sleep(10)
    print("Terminating failed, forcefully killing process(es)...")
    # Force kill the processes if they are still running
    for process in processes:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)  # Force kill the process group
        except OSError:
            pass  # Ignore if the process has already finished

    sys.exit(0)

# List to hold subprocesses
processes = []

# Register the signal handler for SIGINT (Ctrl + C) and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start the supervisor.py process
supervisor_process = subprocess.Popen(['python3', 'supervisor.py'], preexec_fn=os.setsid)
processes.append(supervisor_process)

# Start the master.py process
master_process = subprocess.Popen(['python3', 'master.py'], preexec_fn=os.setsid)
processes.append(master_process)

# Wait for the master process to complete
master_process.wait()

# Once master.py completes, signal to shut down the supervisor
print('Master process completed. Shutting down supervisor...')
os.killpg(os.getpgid(supervisor_process.pid), signal.SIGTERM)

# Wait for 5 seconds
time.sleep(5)

# Force kill the supervisor process if it's still running
try:
    os.killpg(os.getpgid(supervisor_process.pid), signal.SIGKILL)
except OSError:
    pass  # Ignore if the process has already finished

# Ensure the signal handler handles any remaining subprocesses
for process in processes:
    if process.poll() is None:  # Check if the process is still running
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# Wait for both subprocesses to complete
for process in processes:
    process.wait()

# Exit the parent script
sys.exit(0)