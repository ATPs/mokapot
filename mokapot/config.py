"""
Contains all of the configuration details for running mokapot
from the command line.
"""

import argparse
import textwrap
from pathlib import Path

from mokapot import __version__


class MokapotHelpFormatter(argparse.HelpFormatter):
    """Format help text to keep newlines and expose defaults."""

    def _fill_text(self, text, width, indent):
        text_list = text.splitlines(keepends=True)
        return "\n".join(
            _process_line(line, width, indent) for line in text_list
        )

    def _get_help_string(self, action):
        help_string = action.help or ""
        if "%(default)" in help_string:
            return help_string
        if action.default is argparse.SUPPRESS:
            return help_string
        if isinstance(
            action,
            (argparse._StoreTrueAction, argparse._StoreFalseAction),
        ):
            return help_string
        if action.option_strings and action.default is not None:
            return f"{help_string} (default: %(default)s)"
        return help_string


class Config:
    """
    The mokapot configuration options.

    Options can be specified as command-line arguments.
    """

    def __init__(self, parser=None, main_args=None) -> None:
        """Initialize configuration values."""
        self._namespace = None
        if parser is None:
            self.parser = _parser()
        else:
            self.parser = parser
        self.main_args = main_args

    @property
    def args(self):
        """Collect args lazily."""
        if self._namespace is None:
            self._namespace = vars(self.parser.parse_args(self.main_args))

        return self._namespace

    def __getattr__(self, option):
        return self.args[option]


def _parse_model_n_jobs(value: str) -> int | str:
    """Parse --model_n_jobs as a positive integer or 'auto'."""
    if value == "auto":
        return value

    try:
        n_jobs = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer or 'auto'."
        ) from err

    if n_jobs < 1:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer or 'auto'."
        )

    return n_jobs


def _parse_workers_or_auto(value: str) -> int | str:
    """Parse worker counts as a positive integer or 'auto'."""
    if value == "auto":
        return value

    try:
        workers = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer or 'auto'."
        ) from err

    if workers < 1:
        raise argparse.ArgumentTypeError(
            "Expected a positive integer or 'auto'."
        )

    return workers


def _parser():
    """The parser"""
    desc = (
        f"mokapot version {__version__}.\n"
        "Written by William E. Fondrie (wfondrie@talus.bio) while in the \n"
        "Department of Genome Sciences at the University of Washington.\n\n"
        "Official code website: https://github.com/wfondrie/mokapot\n\n"
        "More documentation and examples: https://mokapot.readthedocs.io"
    )

    parser = argparse.ArgumentParser(
        description=desc, formatter_class=MokapotHelpFormatter
    )

    parser.add_argument(
        "psm_files",
        type=Path,
        nargs="+",
        help=(
            "A collection of PSMs in the Percolator tab-delimited "
            "or PepXML format. Gzip-compressed Percolator tab-delimited "
            "files (e.g., .pin.gz, .tsv.gz) are supported."
        ),
    )

    parser.add_argument(
        "-d",
        "--dest_dir",
        type=Path,
        help=(
            "The directory in which to write the result files. If omitted, "
            "the current working directory is used."
        ),
    )

    parser.add_argument(
        "--temp",
        type=Path,
        help=(
            "Temporary workspace directory for intermediate files. "
            "If omitted, uses --dest_dir when provided; otherwise the "
            "current working directory."
        ),
    )

    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help=(
            "Force regeneration of temporary intermediates in --temp "
            "(for example, auto-parquet converted files)."
        ),
    )

    parser.add_argument(
        "-w",
        "--max_workers",
        default=1,
        type=int,
        help=(
            "The number of processes to use for model training. Note that "
            "using more than one worker will result in garbled logging "
            "messages."
        ),
    )

    parser.add_argument(
        "--worker_policy",
        default="auto",
        choices=["auto", "manual"],
        help=(
            "Worker budget policy. 'auto' caps --max_workers to available "
            "CPUs. 'manual' keeps the requested value as-is."
        ),
    )

    parser.add_argument(
        "--read_workers",
        default="auto",
        type=_parse_workers_or_auto,
        help=(
            "Workers used to read multiple input files in parallel. "
            "Use 'auto' to infer from I/O probing."
        ),
    )

    parser.add_argument(
        "--confidence_workers",
        default="auto",
        type=_parse_workers_or_auto,
        help=(
            "Workers used to process multiple datasets during confidence "
            "assignment. Use 'auto' to infer from --max_workers."
        ),
    )

    parser.add_argument(
        "--auto-parquet",
        "--auto_parquet",
        dest="auto_parquet",
        default=True,
        action="store_true",
        help=(
            "Automatically convert large text PIN/TSV inputs to parquet "
            "in the temporary workspace before parsing. Enabled by default."
        ),
    )

    parser.add_argument(
        "--no-auto-parquet",
        "--no_auto_parquet",
        dest="auto_parquet",
        action="store_false",
        help="Disable automatic text-to-parquet conversion.",
    )

    parser.add_argument(
        "--auto-parquet-min-bytes",
        "--auto_parquet_min_bytes",
        default=8 * 1024**3,
        type=int,
        help=(
            "Trigger auto-parquet conversion when the total size of text "
            "inputs reaches this many bytes (8 GiB by default)."
        ),
    )

    parser.add_argument(
        "--auto-parquet-min-files",
        "--auto_parquet_min_files",
        default=256,
        type=int,
        help=(
            "Trigger auto-parquet conversion when at least this many text "
            "input files are provided."
        ),
    )

    parser.add_argument(
        "--auto-parquet-workers",
        "--auto_parquet_workers",
        default=None,
        type=int,
        help=(
            "Number of workers used for parallel text-to-parquet conversion. "
            "Defaults to --max_workers."
        ),
    )

    parser.add_argument(
        "--memmap-threshold-psms",
        "--memmap_threshold_psms",
        default=100_000_000,
        type=int,
        help=(
            "Use numpy.memmap for selected large intermediate arrays when a "
            "dataset has at least this many PSM rows."
        ),
    )

    parser.add_argument(
        "--model_n_jobs",
        default="auto",
        type=_parse_model_n_jobs,
        help=(
            "The number of jobs used by the model hyperparameter search. "
            "Use 'auto' to derive this from --max_workers and --folds."
        ),
    )

    parser.add_argument(
        "-r",
        "--file_root",
        type=str,
        help="The prefix added to all file names.",
    )

    parser.add_argument(
        "--proteins",
        type=Path,
        help=(
            "The FASTA file used for the database search. Using this "
            "option enable protein-level confidence estimates using "
            "the 'picked-protein' approach. Note that the FASTA file "
            "must contain both target and decoy sequences. "
            "Additionally, verify that the '--enzyme', "
            "'--missed_cleavages, '--min_length', '--max_length', "
            "'--semi', '--clip_nterm_methionine', and '--decoy_prefix' "
            "parameters match your search engine conditions."
        ),
    )

    parser.add_argument(
        "--decoy_prefix",
        type=str,
        default="decoy_",
        help=(
            "The prefix used to indicate a decoy protein in the "
            "FASTA file. For mokapot to provide accurate confidence "
            "estimates, decoy proteins should have same description "
            "as the target proteins they were generated from, but "
            "this string prepended."
        ),
    )

    parser.add_argument(
        "--enzyme",
        type=str,
        default="[KR]",
        help=(
            "A regular expression defining the enzyme specificity. "
            "The cleavage site is interpreted as the end of the match. "
            "The default is trypsin, without proline suppression: [KR]"
        ),
    )

    parser.add_argument(
        "--missed_cleavages",
        type=int,
        default=2,
        help="The allowed number of missed cleavages",
    )

    parser.add_argument(
        "--clip_nterm_methionine",
        default=False,
        action="store_true",
        help=(
            "Remove methionine residues that occur at the protein N-terminus."
        ),
    )

    parser.add_argument(
        "--min_length",
        type=int,
        default=6,
        help="The minimum peptide length to consider.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="The maximum peptide length to consider.",
    )

    parser.add_argument(
        "--semi",
        default=False,
        action="store_true",
        help=(
            "Was a semi-enzymatic digest used to assign PSMs? If"
            " so, the protein database will likely contain "
            "shared peptides and yield unhelpful protein-level confidence "
            "estimates. We do not recommend using this option."
        ),
    )

    parser.add_argument(
        "--train_fdr",
        default=0.01,
        type=float,
        help=(
            "The maximum false discovery rate at which to "
            "consider a target PSM as a positive example "
            "during model training."
        ),
    )

    parser.add_argument(
        "--test_fdr",
        default=0.01,
        type=float,
        help=(
            "The false-discovery rate threshold at which to "
            "evaluate the learned models."
        ),
    )

    parser.add_argument(
        "--max_iter",
        default=10,
        type=int,
        help="The number of iterations to use for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="An integer to use as the random seed.",
    )

    parser.add_argument(
        "--direction",
        type=str,
        help=(
            "The name of the feature to use as the initial "
            "direction for ranking PSMs. The default "
            "automatically selects the feature that finds "
            "the most PSMs below the `train_fdr`."
        ),
    )

    parser.add_argument(
        "--aggregate",
        default=False,
        action="store_true",
        help=(
            "If used, PSMs from multiple PIN files will be "
            "aggregated and analyzed together. Otherwise, "
            "a joint model will be trained, but confidence "
            "estimates will be calculated separately for "
            "each PIN file. This flag only has an effect "
            "when multiple PIN files are provided."
        ),
    )

    parser.add_argument(
        "--subset_max_train",
        type=int,
        default=None,
        help=(
            "Maximum number of PSMs to use during the training "
            "of each of the cross validation folds in the model. "
            "This is useful for very large datasets and will be "
            "ignored if less PSMS are available."
        ),
    )

    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help=(
            "Use the learned model even if it performs worse"
            " than the best feature."
        ),
    )

    parser.add_argument(
        "--save_models",
        default=False,
        action="store_true",
        help=(
            "Save the models learned by mokapot as pickled Python "
            "objects (.pkl)."
        ),
    )

    parser.add_argument(
        "--save-percolator-models",
        "--save_percolator_models",
        default=False,
        action="store_true",
        help=(
            "Save the trained models as a single Percolator-style "
            "weights file ('mokapot.model.weights.txt'). This file can "
            "be reused with --load_models."
        ),
    )

    parser.add_argument(
        "--load_models",
        type=Path,
        nargs="+",
        help=(
            "Load previously saved models and skip model training. "
            "Supports mokapot pickled models (.pkl) and Percolator "
            "--weights files (which may contain multiple fold models). "
            "After expansion, the number of models must match --folds."
        ),
    )

    parser.add_argument(
        "--keep_decoys",
        default=False,
        action="store_true",
        help="Keep the decoys in the output .txt files",
    )

    parser.add_argument(
        "--skip_deduplication",
        default=False,
        action="store_true",
        help="Keep deduplication of psms wrt scan number and expMass.",
    )

    parser.add_argument(
        "--skip_rollup",
        default=False,
        action="store_true",
        help="Don't do the rollup to peptide (or other) levels.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help=(
            "The number of cross-validation folds to use. "
            "PSMs originating from the same mass spectrum "
            "are always in the same fold."
        ),
    )

    parser.add_argument(
        "--ensemble",
        default=False,
        action="store_true",
        help="Activate ensemble prediction.",
    )

    parser.add_argument(
        "--peps_error",
        default=False,
        action="store_true",
        help="Raise error when all PEPs values are equal to 1.",
    )

    parser.add_argument(
        "--peps_algorithm",
        default="qvality",
        choices=["qvality", "qvality_bin", "kde_nnls", "hist_nnls"],
        help=(
            "Specify the algorithm for pep computation. 'qvality_bin' works "
            "only if the qvality binary is on the search path"
        ),
    )

    parser.add_argument(
        "--qvalue_algorithm",
        default="tdc",
        choices=["tdc", "from_peps", "from_counts"],
        help=(
            "Specify the algorithm for qvalue computation. `tdc` is "
            "the default mokapot algorithm."
        ),
    )

    parser.add_argument(
        "--open_modification_bin_size",
        type=float,
        help=(
            "This parameter only affect reading PSMs from PepXML files. "
            "If specified, modification masses are binned according to the "
            "value. The binned mass difference is appended to the end of the "
            "peptide and will be used when grouping peptides for peptide-level"
            " confidence estimation. Using this option for open modification "
            "search results. We recommend 0.01 as a good starting point."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        default=2,
        type=int,
        choices=[0, 1, 2, 3],
        help=(
            "Specify the verbosity of the current "
            "process. Each level prints the following "
            "messages, including all those at a lower "
            "verbosity: 0-errors, 1-warnings, 2-messages"
            ", 3-debug info."
        ),
    )

    parser.add_argument(
        "--suppress_warnings",
        default=False,
        action="store_true",
        help=(
            "Suppress warning messages when running mokapot. "
            "Should only be used when running mokapot in production."
        ),
    )

    parser.add_argument(
        "--sqlite_db_path",
        default=None,
        type=Path,
        help="Optionally, sets a path to an MSAID sqlite result database "
        "for writing outputs to. If not set (None), results are "
        "written in the standard TSV format.",
    )

    parser.add_argument(
        "--stream_confidence",
        default=False,
        action="store_true",
        help="Specify whether confidence assignment shall be streamed.",
    )

    return parser


def _process_line(line: str, width: int, indent: str) -> str:
    line = textwrap.fill(
        line,
        width,
        initial_indent=indent,
        subsequent_indent=indent,
        replace_whitespace=False,
    )
    return line.strip()
