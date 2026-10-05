import csv
from pathlib import Path

from Bio import SeqIO
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    EsmFold2Model,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)


def get_skip_ids_set(path):
    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)  # skip header
        return {row[0] for row in reader}


def get_model():
    model = EsmFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
    return model


def predict_structure(model, protein, **kwargs):
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=protein),
        ]
    )
    result = ESMFold2InputBuilder().fold(model, spi, **kwargs)
    return result


def make_predictions(
    fasta_path, out_dir, max_length=float("inf"), skip_completed=False
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    data_path = Path(out_dir / "data.tsv")
    if not data_path.is_file():
        skip_completed = False
    if skip_completed:
        skip_ids_set = get_skip_ids_set(data_path)
        mode = "a"
    else:
        skip_ids_set = {}
        mode = "w"

    model = get_model()
    with open(fasta_path) as fasta_file, open(
        data_path, mode, buffering=1, newline=""
    ) as data_file:
        writer = csv.writer(data_file, delimiter="\t")
        if not skip_completed:
            writer.writerow(("seq_id", "plddt_mean", "ptm"))
        proteins = SeqIO.parse(fasta_file, "fasta")
        for i, rec in enumerate(proteins, start=1):
            length = len(rec.seq)
            if length > max_length:
                print(f"skipping {rec.id}: too long ({length})", flush=True)
                continue
            if rec.id in skip_ids_set:
                print(f"skipping {rec.id}: already completed", flush=True)
                continue
            print(f"predicting {i} {rec.id}", flush=True)
            result = predict_structure(
                model,
                str(rec.seq),
                num_loops=20,
                num_sampling_steps=100,
                num_diffusion_samples=1,
                seed=0,
            )
            with open(out_dir / f"{rec.id}.cif", "w") as file:
                file.write(result.complex.to_mmcif())
            writer.writerow(
                (rec.id, float(result.plddt.mean()), float(result.ptm))
            )


if __name__ == "__main__":
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Process a FASTA file.")
        parser.add_argument(
            "--input",
            "-i",
            type=Path,
            required=True,
            help="Path to the input FASTA file",
        )
        parser.add_argument(
            "--max-length",
            "-l",
            type=int,
            help="Skip sequences longer than max length",
            default=float("inf"),
        )
        parser.add_argument(
            "--skip-completed",
            "-s",
            action="store_true",
            help="Skip completed sequences in data.tsv",
        )
        parser.add_argument(
            "--out-dir",
            "-o",
            type=Path,
            required=True,
            help="Directory for output files",
        )

        args = parser.parse_args()

        if not args.input.is_file():
            parser.error(f"Input file does not exist: {args.input}")
        make_predictions(
            args.input,
            args.out_dir,
            max_length=args.max_length,
            skip_completed=args.skip_completed,
        )

    main()
