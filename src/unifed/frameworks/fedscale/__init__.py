import sys

from unifed.frameworks.fedscale import protocol


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

