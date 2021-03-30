import argparse

class BaseScoringUDF:
    def __init__(self):
        self.arg_parser = argparse.ArgumentParser()

    def get_arg_parser(self):
        return self.arg_parser
    
    def initialize(self, opt, gpu=None):
        self.opt = opt
    
    def set_args(self, args):
        for key, value in args.items():
            self.opt.__dict__[key] = value

    def get_scores(self, imgs, visualize=False):
        return [0] * len(imgs)
    
    def get_img_size(self):
        return (416, 416)