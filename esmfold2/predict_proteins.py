from pathlib import Path

import polars as pl
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


def make_predictions(fasta_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    model = get_model()
    data = {"seq_id": [], "plddt_mean": [], "pTM": []}
    with open(fasta_path) as fasta_file:
        proteins = SeqIO.parse(fasta_file, "fasta")
        for i, rec in enumerate(proteins, start=1):
            print(f"predicting {i} {rec.id}")
            result = predict_structure(
                model,
                str(rec.seq),
                num_loops=20,
                num_sampling_steps=100,
                num_diffusion_samples=1,
                seed=0,
            )
            data["seq_id"].append(rec.id)
            data["plddt_mean"].append(float(result.plddt.mean()))
            data["pTM"].append(float(result.ptm))
            with open(out_dir / f"{rec.id}.cif", "w") as file:
                file.write(result.complex.to_mmcif())
    pl.DataFrame(data).write_csv(out_dir / "data.tsv", separator="\t")


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
            "--out-dir",
            "-o",
            type=Path,
            required=True,
            help="Directory for output files",
        )

        args = parser.parse_args()

        if not args.input.is_file():
            parser.error(f"Input file does not exist: {args.input}")
        make_predictions(args.input, args.out_dir)

    main()
