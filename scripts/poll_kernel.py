#!/usr/bin/env python3
"""Poll a Kaggle kernel until it completes."""
import subprocess
import sys
import time
import argparse

def poll_kernel(kernel_id, max_wait_minutes=30, poll_interval=30):
    """
    Poll a Kaggle kernel status until completion or timeout.
    
    Args:
        kernel_id: Kaggle kernel ID (e.g., "leventecsibi/vlm-chart-benchmark-baseline-smolvlm2-2.2b")
        max_wait_minutes: Maximum time to wait in minutes
        poll_interval: Seconds between status checks
    """
    print(f"Polling kernel: {kernel_id}")
    print(f"Max wait time: {max_wait_minutes} minutes")
    print(f"Poll interval: {poll_interval} seconds\n")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    iteration = 0
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print(f"[{iteration}] Checking status... ({elapsed_min:.1f} min elapsed)")
        
        try:
            result = subprocess.run(
                ["poetry", "run", "kaggle", "kernels", "status", kernel_id],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            output = result.stdout.strip()
            print(f"    Status: {output}")
            
            if "complete" in output.lower():
                print(f"\n✓ Kernel completed successfully!")
                return "complete"
            elif "error" in output.lower():
                print(f"\n✗ Kernel errored!")
                return "error"
            elif "running" in output.lower():
                print(f"    Still running...")
            
        except subprocess.TimeoutExpired:
            print(f"    Status check timed out")
        except Exception as e:
            print(f"    Error checking status: {e}")
        
        if elapsed > max_wait_seconds:
            print(f"\n⏰ Timeout reached ({max_wait_minutes} minutes)")
            return "timeout"
        
        print(f"    Waiting {poll_interval} seconds...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll Kaggle kernel status")
    parser.add_argument("kernel_id", help="Kaggle kernel ID")
    parser.add_argument("--max-wait", type=int, default=30, help="Maximum wait time in minutes")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    
    args = parser.parse_args()
    
    status = poll_kernel(args.kernel_id, args.max_wait, args.interval)
    sys.exit(0 if status == "complete" else 1)
