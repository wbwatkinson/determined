import subprocess
import time
from typing import List

def humanize_float(num): return "{0:,.2f}".format(num)
h = humanize_float

COMMAND = "det --help"

def print_install_info():
    subprocess.check_call("pip freeze | grep determined", shell=True)

def run_cmd() -> float:
    start = time.monotonic()
    subprocess.check_call(COMMAND, shell=True, stdout=subprocess.DEVNULL)
    end = time.monotonic()
    dur = end - start
    return dur

def average_durs(durs: List[float]) -> float:
    return sum(durs)/len(durs)


if __name__ == '__main__':
    ITERATIONS = 100
    durs = []
    for i in range(ITERATIONS):
        print(f"{i+1} of {ITERATIONS}")
        cmd_dur = run_cmd()
        durs.append(cmd_dur)

    avg_dur = average_durs(durs)
    print(f"Over {ITERATIONS} iterations, '{COMMAND}' averaged {h(avg_dur)} seconds")
    print_install_info()




###### det --help (3.8)
# Wheel install          = 0.76
# Setup.py install       = 1.02
# fastentrypoint install = 1.02

###### det --help (3.7)
# Wheel install          = 0.78
# Setup.py install       = 1.11
# fastentrypoint install = 1.10


###### det version (3.7)
# Wheel install          = 0.88s
# Setup.py install       = 1.18s
# fastentrypoint install = 1.10

