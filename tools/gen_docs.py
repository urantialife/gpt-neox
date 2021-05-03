import os
import sys
import inspect
import re

from argparse import ArgumentParser
from dataclasses import is_dataclass
from pytablewriter import MarkdownTableWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.neox_arguments import neox_args
from megatron.neox_arguments import deepspeed_args


def get_all_dataclasses(module):
    dclasses = []
    for attribute_name in dir(module):
        obj = getattr(module, attribute_name)
        if is_dataclass(obj):
            dclasses.append(obj)
    return dclasses


def get_docs(module):
    # N.B: arg / typehint / default combinations that extend over more than a line will break this, because i'm bad
    # at regex
    regexp_magic = r'([^:"]+)[:]?([^:\n]+)=[\s]?(.+)[\s+]?\n[\s+]?\s+"""([\s\S]*?)"""'
    comment = "#.*\n"
    dclasses = get_all_dataclasses(module)  # get all instances of 'dataclass' in the module
    docs = ""
    for dclass in dclasses:
        source = inspect.getsource(dclass)  # get source code
        matches = re.findall(regexp_magic, source)  # find parts that match `arg: typehint = default \n """docstring"""`
        values = []
        for match in matches:
            # format
            arg_name, type_hint, default_value, docstring = match
            arg_name = arg_name.strip()
            type_hint = type_hint.strip()
            default_value = default_value.strip()
            docstring = docstring.strip()
            match_comment = re.match(comment, arg_name)
            if match_comment:
                arg_name = arg_name.replace(match_comment.group(0), "").strip()
            values.append([arg_name, type_hint, default_value, docstring])
        if values:
            # construct md table
            class_name = dclass.__name__
            desc = class_name.replace("NeoXArgs", "")
            writer = MarkdownTableWriter(
                table_name=f"{desc} arguments",
                headers=["Argument Name", "Type Hint", "Default", "Docstring"],
                value_matrix=values
            )
            docs += writer.dumps() + "\n"
    return docs

def parse_args():
    parser = ArgumentParser("Autogenerate docs from NeoXArgs dataclasses")
    parser.add_argument("save_path", help="path relative to save (markdown) docs to", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    docs = get_docs(neox_args) + "\n" + get_docs(deepspeed_args)
    print(docs)
    with open(f'../{args.save_path}', 'w') as f:
        f.write(docs)
