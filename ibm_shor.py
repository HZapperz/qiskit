from qiskit_ibm_runtime import QiskitRuntimeService
import time
from script import run_shor_on_77  # Import your existing function
from ibm_config import IBM_TOKEN  # Import token from config file

def manual_job_monitor(job):
    """
    Monitor job status with manual polling
    """
    while not job.status().is_final():
        print(f"Job status: {job.status()}")
        time.sleep(5)
    print(f"Job finished with status: {job.status()}")

def setup_ibmq(token):
    """
    Setup IBM Quantum account using QiskitRuntimeService
    """
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    
    # List available backends
    print("\nAvailable backends:")
    for backend in service.backends():
        print(f"- {backend.name}: {backend.num_qubits} qubits")
    
    return service

def run_on_ibmq(service, backend_name='ibm_brisbane'):
    """
    Run Shor's algorithm on specified IBM Q backend
    """
    try:
        backend = service.backend(backend_name)
        print(f"\nUsing backend: {backend.name}")
        
        # Try different 'a' values
        for a in [2, 3, 5]:
            print(f"\n{'='*60}")
            print(f"Trying with a = {a} on {backend.name}")
            print('='*60)
            
            # Just pass the 'a' value since run_shor_on_77 doesn't accept backend
            job = run_shor_on_77(a=a)
            manual_job_monitor(job)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nAvailable backends are:")
        for backend in service.backends():
            print(f"- {backend.name}")

if __name__ == "__main__":
    # Use token from config
    service = setup_ibmq(IBM_TOKEN)
    
    # Run on available backend
    run_on_ibmq(service)  # Will use ibm_brisbane by default 