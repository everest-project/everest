import os
import importlib
import argparse

def get_udf_class(udf_name):
    class_name = "".join([s.capitalize() for s in udf_name.split("_")])
    udf_module = importlib.import_module(f"oracle.udf.{udf_name}.score_func")
    udf_class = getattr(udf_module, class_name)
    return udf_class

def dict_to_opt(params, parser):
    cmd_arg = []
    for key, value in params.items():
        cmd_arg.append(f"--{key}")
        cmd_arg.append(f"{value}")
    opt, _ = parser.parse_known_args(cmd_arg)
    return opt