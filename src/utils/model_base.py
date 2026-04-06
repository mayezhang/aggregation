import os
import argparse

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')

class ModelBase(object):
    def __init__(self, env, args: argparse.Namespace):
        self.args = args
        self.env = env
        self.env_evaluate = env
        self.agent = None
        self.model_name = None
    
    def train(self):
        raise NotImplementedError
