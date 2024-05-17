import boto3
import time
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
ec2 = boto3.client('ec2', region_name='eu-north-1')
cloudwatch = boto3.client('cloudwatch', region_name='eu-north-1')

# Define the instance IDs of your master and worker nodes
instance_ids = {
    'i-01a28545a12ed89df': 'masterNodeStatus',
    'i-0339bfe4046feb681': 'workerNode1Status',
    'i-081cc29642db6b146': 'workerNode2Status',
    'i-0141d647c9071826e': 'workerNode3Status',
    'i-027ebee0933814be0': 'workerNode4Status'
}

def get_instance_status(instance_id):
    response = ec2.describe_instance_status(InstanceIds=[instance_id])
    if len(response['InstanceStatuses']) == 0:
        return 'stopped'  # No status means the instance is stopped
    state = response['InstanceStatuses'][0]['InstanceState']['Name']
    return state

def get_cpu_utilization(instance_id):
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=time.time() - 120,  # Last 2 minutes
        EndTime=time.time(),
        Period=60,  # 1-minute granularity
        Statistics=['Average']
    )
    datapoints = response['Datapoints']
    if datapoints:
        return datapoints[-1]['Average']
    return 0.0

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to status server'})

def monitor_status():
    while True:
        statuses = {}
        for instance_id, node_name in instance_ids.items():
            status = get_instance_status(instance_id)
            if status == 'running':
                cpu_utilization = get_cpu_utilization(instance_id)
                statuses[node_name] = f'{status} (CPU: {cpu_utilization:.2f}%)'
            else:
                statuses[node_name] = status
        print("Sending statuses:", statuses)
        socketio.emit('node_status', statuses)
        time.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    print("Starting WebSocket server on port 5001...")
    socketio.start_background_task(monitor_status)
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)