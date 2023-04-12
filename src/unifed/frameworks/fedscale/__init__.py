import sys

from unifed.frameworks.fedscale import protocol
from unifed.frameworks.fedscale.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

