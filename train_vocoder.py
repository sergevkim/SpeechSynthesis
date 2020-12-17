from argparse import ArgumentParser
from pathlib import Path

from spynt.datamodules import LJSpeechDataModule2
from spynt.loggers import NeptuneLogger
from spynt.models import WaveNetVocoder
from spynt.trainer import Trainer

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    vocoder = WaveNetVocoder().to(args.device)
    datamodule = LJSpeechDataModule2(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    #logger = NeptuneLogger(
    #    api_key=None,
    #    args.project_name=None,
    #)
    logger = None
    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=vocoder,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()

    main(args)

