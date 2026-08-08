"""Entry point: dispatch to a task pipeline.

    python main.py loss_law --input ... --output_dir ...   # recover a loss scaling law
    python main.py hp_law   --input ... --output_dir ...   # recover an optimal-HP scaling law

Each task's flags come from its pipeline (see ``core/pipelines/``); run
``python main.py <task> --help`` for the task-specific options.
"""
import argparse

from core.pipelines import PIPELINES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="task", required=True, metavar="{" + ",".join(PIPELINES) + "}")
    for name, pipeline in PIPELINES.items():
        p = sub.add_parser(name, help=pipeline.__doc__.splitlines()[0] if pipeline.__doc__ else name)
        pipeline.add_args(p)
    args = parser.parse_args()
    PIPELINES[args.task].run(args)


if __name__ == "__main__":
    main()
