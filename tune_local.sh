#!/bin/bash

# This script runs tune.py using configurations suitable for testing
# the tune script locally. For our deployed models, use "tune_deployment.sh".

time python3 tune.py --device "cpu"
