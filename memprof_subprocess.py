import subprocess
import resource
import sys

COMMAND = sys.argv[1]

p = subprocess.Popen(
    COMMAND, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
p.wait()
print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
