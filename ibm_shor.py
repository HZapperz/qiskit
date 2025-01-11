from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, RuntimeJob
import time
from script import run_shor_on_77, process_results  # Import process_results too
from ibm_config import IBM_TOKEN  # Import token from config file

def manual_job_monitor(job):
    """
    Monitor job status with manual polling
    """
    try:
        # Check if job is a RuntimeJob
        if isinstance(job, RuntimeJob):
            while not job.status().is_final():
                print(f"Job status: {job.status()}")
                time.sleep(5)
            print(f"Job finished with status: {job.status()}")
            
            # Get and process results
            result = job.result()
            
            # Process the sampler results
            if hasattr(result, 'quasi_dists'):
                print("\nQuasi-probability distribution:")
                quasi_dist = result.quasi_dists[0]
                print(quasi_dist)
                
                # Convert to approximate counts
                shots = 2048  # or get this from the job configuration
                counts = {k: int(v * shots) for k, v in quasi_dist.items()}
                print("\nApproximate counts:")
                print(counts)
                
                # Plot the results
                try:
                    from qiskit.visualization import plot_histogram
                    import matplotlib.pyplot as plt
                    plot_histogram(counts)
                    plt.show()
                except Exception as e:
                    print(f"Couldn't plot histogram: {str(e)}")
            else:
                print("\nRaw result:")
                print(result)
                
        else:
            print(f"Job ID: {job}")
            print("Job submitted successfully")
    except Exception as e:
        print(f"Error monitoring job: {str(e)}")

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
            
            # Pass both backend and service to run_shor_on_77
            job = run_shor_on_77(a=a, backend=backend, service=service)
            if job is not None:
                manual_job_monitor(job)
            else:
                print("Failed to create job")
            
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