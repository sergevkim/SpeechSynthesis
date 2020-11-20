from argparse import ArgumentParser
from pathlib import Path

from spynt.datamodules import LJSpeechDataModule
from spynt.loggers import NeptuneLogger
from spynt.models import Tacotron2Synthesizer
from spynt.trainer import Trainer

from config import Arguments


def main(args):
    synthesizer = Tacotron2Synthesizer(
    ).to(args.device)
    datamodule = LJSpeechDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    #logger = NeptuneLogger(
    #    api_key=None,
    #    args.project_name=None,
    #)
    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=synthesizer,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    #parser = ArgumentParser()
    #args = parser.parse_args()
    args = Arguments()

    main(args)

