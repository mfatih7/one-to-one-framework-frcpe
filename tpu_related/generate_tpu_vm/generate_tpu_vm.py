import subprocess


first_machine_no = 0
last_machine_no = 8

# first_machine_no = 0
# last_machine_no = 2

machine_type = 'on_demand'
# machine_type = 'preemptible'

operation = 'generate'
# operation = 'start'

# accelerator_type = 'v_2'
accelerator_type = 'v_3'
# accelerator_type = 'v_4' # BETTER TO USE queued-resources https://cloud.google.com/tpu/docs/queued-resources

iteration_count = 0

for machine_no in range(first_machine_no, last_machine_no, 1):
    
    if(machine_type=='on_demand' and operation=='generate' and accelerator_type == 'v_2'):
        command = 'gcloud compute tpus tpu-vm create tpu-v2-8-us-central1-f-' + f"{machine_no:03}" + ' --zone=us-central1-f --accelerator-type=v2-8 --version=tpu-vm-pt-2.0'
    elif(machine_type=='on_demand' and operation=='generate' and accelerator_type == 'v_3'):
        command = 'gcloud compute tpus tpu-vm create tpu-v3-8-europe-west4-a-' + f"{machine_no:03}" + ' --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-2.0'
    elif(machine_type=='on_demand' and operation=='start' and accelerator_type == 'v_2'):
        command = 'gcloud compute tpus tpu-vm start tpu-v2-8-us-central1-f-' + f"{machine_no:03}" + ' --zone=us-central1-f'
    elif(machine_type=='on_demand' and operation=='start' and accelerator_type == 'v_3'):
        command = 'gcloud compute tpus tpu-vm start tpu-v3-8-europe-west4-a-' + f"{machine_no:03}" + ' --zone=europe-west4-a'
    elif(machine_type=='preemptible' and operation=='generate' and accelerator_type == 'v_2'):
        command = 'gcloud compute tpus tpu-vm create tpu_preemp-v2-8-us-central1-f-' + f"{machine_no:03}" + ' --zone=us-central1-f --accelerator-type=v2-8 --version=tpu-vm-pt-2.0 --preemptible'
    elif(machine_type=='preemptible' and operation=='generate' and accelerator_type == 'v_3'):
        command = 'gcloud compute tpus tpu-vm create tpu_preemp-v3-8-europe-west4-a-' + f"{machine_no:03}" + ' --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-vm-pt-2.0 --preemptible'
    elif(machine_type=='preemptible' and operation=='start' and accelerator_type == 'v_2'):
        command = 'gcloud compute tpus tpu-vm start tpu_preemp-v2-8-us-central1-f-' + f"{machine_no:03}" + ' --zone=us-central1-f'
    elif(machine_type=='preemptible' and operation=='start' and accelerator_type == 'v_3'):
        command = 'gcloud compute tpus tpu-vm start tpu_preemp-v3-8-europe-west4-a-' + f"{machine_no:03}" + ' --zone=europe-west4-a'
        
    elif(machine_type=='on_demand' and operation=='generate' and accelerator_type == 'v_4'):
        command = 'gcloud compute tpus tpu-vm create tpu-v4-8-us-central2-b-' + f"{machine_no:03}" + ' --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-vm-v4-pt-2.0'
    elif(machine_type=='on_demand' and operation=='start' and accelerator_type == 'v_4'):
        command = 'gcloud compute tpus tpu-vm start tpu-v4-8-us-central2-b-' + f"{machine_no:03}" + ' --zone=us-central2-b'
    
    while True:

        p1 = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(p1.returncode)
        print(p1.stderr)
        iteration_count += 1
        print(iteration_count)
        print(machine_no)
    
        if(p1.returncode==0):
            break                
