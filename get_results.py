from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from ibm_config import IBM_TOKEN
import numpy as np

def process_primitive_result(job):
    """
    Process results from IBM Quantum Runtime primitive jobs
    """
    result = job.result()
    
    try:
        # Get the BitArray data
        bit_array = result[0].data.c
        array_data = bit_array.array
        
        # Convert measurements to binary and create counts
        counts = {}
        binary_counts = {}
        
        for value in array_data:
            # Get decimal value
            decimal = int(value[0])
            counts[decimal] = counts.get(decimal, 0) + 1
            
            # Convert to 8-bit binary string
            binary = format(decimal, '08b')
            binary_counts[binary] = binary_counts.get(binary, 0) + 1
        
        # Sort by frequency
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        sorted_binary = sorted(binary_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 most frequent measurements:")
        print("\nDecimal  Binary     Count   Percentage")
        print("-" * 40)
        for i, (decimal, count) in enumerate(sorted_counts[:10]):
            binary = format(decimal, '08b')
            percentage = (count / 2048) * 100
            print(f"{decimal:7d}  {binary}  {count:5d}   {percentage:6.2f}%")
            
        return binary_counts
            
    except Exception as e:
        print(f"\nError accessing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_job_results(job_id):
    """
    Retrieve and display results for a specific job
    """
    # Connect to IBM Quantum
    service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
    
    # Get the job
    job = service.job(job_id)
    status = job.status()
    print(f"Job status: {status.name if hasattr(status, 'name') else status}")
    
    # Get results
    counts = process_primitive_result(job)
    if counts:
        # Plot histogram of binary results
        plot_histogram(counts)
        plt.title(f"Results for Job {job_id}")
        plt.show()

if __name__ == "__main__":
    # Use the job IDs from your recent jobs that are DONE
    job_ids = [
        'cy18hhpcw2k0008jegkg',  # DONE
        'cy18hgpnrmz0008549v0',  # DONE
        'cy18fyz01rbg008hryhg'   # DONE
    ]
    
    # First, let's list recent jobs
    service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
    print("\nRecent jobs:")
    for job in service.jobs(limit=5):
        status = job.status()
        print(f"- {job.job_id()}: {status.name if hasattr(status, 'name') else status}")
    
    print("\nTrying to get results for specified jobs:")
    for job_id in job_ids:
        print(f"\nProcessing job: {job_id}")
        get_job_results(job_id) 