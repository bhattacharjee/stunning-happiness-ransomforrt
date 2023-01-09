#!/usr/bin/env python3


import argparse
import copy
import gc
import glob
import json
import os
import random
import sys
import time
import uuid
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from loguru import logger
from sklearn import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

# import dotenv


# Way to dynamically change the number of jobs at run time
def get_num_jobs(default_jobs: int) -> int:
    """This function provides a way to override the number of jobs specified
    in the command line arguments dynamically.
    A file called num_jobs.txt can be created and the first line
    should contain the number of jobs.

    Args:
        default_jobs (int): default value if it is not overridden

    Returns:
        int: number of jobs to run
    """
    if not os.path.exists("num_jobs.txt"):
        return default_jobs
    with open("num_jobs.txt") as f:
        try:
            line = f.readlines()[0].strip()
            temp_jobs = int(line)
            if temp_jobs > 0 and temp_jobs < 20:
                logger.info(f"NUM_JOBS override: {temp_jobs}")
                return temp_jobs
        except:
            return default_jobs
    return default_jobs


def random_seed() -> None:
    np.random.seed(0)
    random.seed(0)


def get_save_filename() -> str:
    return f"{str(uuid.uuid4())}.csv.gz"


def gc_collect() -> None:
    for i in range(3):
        for j in range(3):
            gc.collect(j)


def get_columns_and_types(thisdf: pd.DataFrame) -> Dict[str, List[str]]:
    """For each feature set type, get the relevant columns.

    Args:
        thisdf (pd.DataFrame): Input dataframe.

    Returns:
        Dict[str, List[str]]: Dictionary that maps the feature type to the
            list of columns to the feature type.
    """
    columns = [c for c in thisdf.columns if not c.startswith("an_")]

    def get_columns(columns: List[str], start_string: str) -> List[str]:
        columns = [c for c in columns if c.startswith(start_string)]
        columns = [c for c in columns if "head" not in c and "tail" not in c]
        columns = [c for c in columns if "begin" not in c and "end" not in c]
        columns = [c for c in columns if "filesize" not in c]
        return columns

    baseline_columns = get_columns(columns, "baseline")
    advanced_columns = get_columns(columns, "advanced")
    fourier_columns = get_columns(columns, "fourier")
    fourier_min_columns = [
        "fourier.stat.1byte.autocorr",
        "fourier.stat.1byte.mean",
        "fourier.stat.1byte.std",
        "fourier.stat.1byte.chisq",
        "fourier.stat.1byte.moment.2",
        "fourier.stat.1byte.moment.3",
        "fourier.stat.1byte.moment.4",
        "fourier.stat.1byte.moment.5",
    ]

    baseline_and_advanced = list(set(baseline_columns + advanced_columns))
    baseline_and_fourier = list(set(baseline_columns + fourier_columns))
    advanced_and_fourier = list(set(advanced_columns + fourier_columns))
    baseline_and_fourier_min = list(
        set(baseline_columns + fourier_min_columns)
    )
    advanced_and_fourier_min = list(
        set(advanced_columns + fourier_min_columns)
    )

    baseline_advanced_fourier = list(
        set(baseline_columns + advanced_columns + fourier_columns)
    )
    baseline_advanced_and_fourier_min = list(
        set(baseline_columns + advanced_columns + fourier_min_columns)
    )

    rv = {
        "baseline-only": baseline_columns,
        "advanced-only": advanced_columns,
        "fourier-only": fourier_columns,
        "fourier-min-only": fourier_min_columns,
        "baseline-and-fourier": baseline_and_fourier,
        "baseline-and-fourier-min": baseline_and_fourier_min,
        "baseline-and-advanced": baseline_and_advanced,
        "advanced-and-fourier": advanced_and_fourier,
        "advanced-and-fourier-min": advanced_and_fourier_min,
        "baseline-advanced-and-fourier": baseline_advanced_fourier,
        "baseline-advanced-and-fourier-min": baseline_advanced_and_fourier_min,
    }

    logger.info(f"Features = {rv}")

    return rv


def get_annotation_columns(thisdf: pd.DataFrame) -> List[str]:
    """List of columns used for annotation.

    Args:
        thisdf (pd.DataFrame): Input dataframe.

    Returns:
        _type_: List of columns
    """
    return [c for c in thisdf.columns if c.startswith("an_")]


def annotate_df_with_additional_fields(
    name: str, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Add some metadata to each dataframe

    Args:
        name (str): Name of the csv/parquet file
        dataframe (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with additional information
    """
    if "base32" in name or "b32" in name:
        dataframe["an_is_base32"] = 1
    else:
        dataframe["an_is_base32"] = 0
    dataframe["an_is_base32"] = dataframe["an_is_base32"].astype(np.bool_)

    if "encrypt" in name:
        dataframe["is_encrypted"] = 1
    else:
        dataframe["is_encrypted"] = 0
    dataframe["is_encrypted"] = dataframe["is_encrypted"].astype(np.bool_)

    if "v1" in name:
        dataframe["an_v1_encrypted"] = 1
    else:
        dataframe["an_v1_encrypted"] = 0
    dataframe["an_v1_encrypted"] = dataframe["an_v1_encrypted"].astype(
        np.bool_
    )

    if "v2" in name:
        dataframe["an_v2_encrypted"] = 1
    else:
        dataframe["an_v2_encrypted"] = 0
    dataframe["an_v2_encrypted"] = dataframe["an_v2_encrypted"].astype(
        np.bool_
    )

    if "v3" in name:
        dataframe["an_v3_encrypted"] = 1
    else:
        dataframe["an_v3_encrypted"] = 0
    dataframe["an_v3_encrypted"] = dataframe["an_v3_encrypted"].astype(
        np.bool_
    )

    def is_webp(filename: str) -> int:
        return 1 if ".webp" in filename else 0

    dataframe["an_is_webp"] = (
        dataframe["extended.base_filename"].map(is_webp).astype(np.bool_)
    )

    return dataframe


def load_data(input_directory: str) -> pd.DataFrame:
    """Load all pandas data files from a directory and annotate them with
    additional fields

    Args:
        input_directory (str): input directory

    Returns:
        pd.DataFrame: A combined dataframe of all files
    """
    p = 0.1
    logger.info("Loading dataframes")
    dataframes = {
        # f: pd.read_csv(f, skiprows=lambda i: i > 0 and random.random() > p)
        f: pd.read_csv(f)
        for f in glob.glob(f"{input_directory}{os.path.sep}*.csv.gz")
    }
    logger.info("Annotating dataframes with additional fields")
    dataframes = {
        f: annotate_df_with_additional_fields(f, df)
        for f, df in dataframes.items()
    }

    logger.info("Combining dataframes into a single dataframe")
    df = (
        pd.concat([df for _, df in dataframes.items()])
        .sample(frac=1)
        .reset_index(drop=True)
    )

    gc_collect()

    logger.info("done...")
    return df


def get_pipeline(
    X: pd.DataFrame, n_jobs: int = 4
) -> Tuple[pipeline.Pipeline, Callable[[np.array], np.array]]:
    random_seed()
    num_jobs = get_num_jobs(n_jobs)
    pipe = pipeline.Pipeline(
        [
            ("std", MinMaxScaler()),
            ("classif", RandomForestClassifier(n_jobs=num_jobs)),
        ]
    )
    # In case the predicted value needs to be converted to an integer
    # this lambda will do the work
    return pipe, lambda x: x


def get_metrics(y_true: np.array, y_pred: np.array) -> List[float]:
    def error_checked_metric(fn, y_true, y_pred):
        try:
            return fn(y_true, y_pred)
        except Exception as e:
            return 1.0  # Treat as fully accurate

    return [
        error_checked_metric(fn, y_true, y_pred)
        for fn in [
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        ]
    ]


def evaluate_features_folded(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    metrics = []
    random_seed()
    colnames = [c for c in feature_column_names if "is_encrypted" not in c]
    colnames = [c for c in colnames if c not in annotation_columns]
    colnames = [c for c in colnames if not c.startswith("an_")]
    X = data[colnames].to_numpy()
    # print("COLUMNS: ", data[colnames].columns)
    y = data["is_encrypted"].to_numpy().flatten()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for nid, (train_idx, test_idx) in enumerate(
        tqdm.tqdm(
            skf.split(X, y),
            desc=f"Running {folds} folds verification: ",
            colour="red",
            total=folds,
        )
    ):
        logger.info(
            f"---> Running iteration #{nid:02d} for {folds} fold verification."
        )
        pline, y_pred_fn = get_pipeline(X, n_jobs=n_jobs)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pline.fit(X_train, y_train)
        y_pred = pline.predict(X_test)
        y_pred_final = y_pred_fn(y_pred)
        y_predict_proba = pline.predict_proba(X_test)
        metrics.append(get_metrics(y_test, y_pred_final))
        logger.opt(colors=True).info(
            f"<magenta>another fold done. {metrics[-1]}</>"
        )

        save_filename = output_directory + os.path.sep + get_save_filename()
        df2 = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_predict_proba[:, 1],
            }
        )
        df2.to_csv(save_filename)
        logger.info(f"Saved result to {save_filename}.")

    return True, combine_metrics(metrics)


def evaluate_features_regular(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
) -> Tuple[bool, List[float]]:
    colnames = [c for c in feature_column_names if "is_encrypted" not in c]
    colnames = [c for c in colnames if not c.startswith("an_")]
    X = data[colnames].to_numpy()
    y = data["is_encrypted"].to_numpy().flatten()
    pline = get_pipeline(X, n_jobs=n_jobs)
    return True, []


def evaluate_features(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    random_seed()
    if folds != -1:
        return evaluate_features_folded(
            name=name,
            data=data,
            output_directory=output_directory,
            feature_column_names=feature_column_names,
            annotation_columns=annotation_columns,
            n_jobs=n_jobs,
            folds=folds,
        )
    else:
        return evaluate_features_regular(
            name=name,
            data=data,
            output_directory=output_directory,
            feature_column_names=feature_column_names,
            annotation_columns=annotation_columns,
            n_jobs=n_jobs,
        )


def trim_dataset(
    df: pd.DataFrame,
    exclude_plaintext_nonbase32: bool = False,
    exclude_plaintext_base32: bool = False,
    exclude_encrypted_v1: bool = False,
    exclude_encrypted_v2: bool = False,
    exclude_encrypted_base32: bool = False,
    exclude_encrypted_nonbase32: bool = False,
    exclude_webp: bool = False,
    exclude_nonwebp: bool = False,
):
    df = df.copy()
    logger.debug(f"0 ===> {len(df)}")

    # This is not a realistic combination and the caller should never
    # call this. Putting this assert in place for debugging reasons.
    assert exclude_plaintext_nonbase32 == False

    if exclude_plaintext_nonbase32:
        selector = ~(df["is_encrypted"].astype(np.bool_)) & ~(
            df["an_is_base32"].astype(np.bool_)
        )
        df = df[~selector]
    logger.debug(f"1 ===> {len(df)}")

    if exclude_plaintext_base32:
        selector = ~(df["is_encrypted"].astype(np.bool_)) & df[
            "an_is_base32"
        ].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"2 ===> {len(df)}")

    if exclude_encrypted_v1:
        selector = df["an_v1_encrypted"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"3 ===> {len(df)}")

    if exclude_encrypted_v2:
        selector = df["an_v2_encrypted"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"4 ===> {len(df)}")

    if exclude_encrypted_base32:
        selector = df["is_encrypted"].astype(np.bool_) & df[
            "an_is_base32"
        ].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"5 ===> {len(df)}")

    if exclude_encrypted_nonbase32:
        selector = df["is_encrypted"].astype(np.bool_) & ~(
            df["an_is_base32"].astype(np.bool_)
        )
        df = df[~selector]
    logger.debug(f"6 ===> {len(df)}")

    if exclude_webp:
        selector = df["an_is_webp"].astype(np.bool_)
        df = df[~selector]
    logger.debug(f"7 ===> {len(df)}")

    if exclude_nonwebp:
        selector = ~(df["an_is_webp"].astype(np.bool_))
        df = df[~selector]
    logger.debug(f"8 ===> {len(df)}")

    try:
        non_encrypted_count = (~df["is_encrypted"]).astype(np.int8).abs().sum()
    except:
        non_encrypted_count = 0
    try:
        encrypted_count = df["is_encrypted"].astype(np.int8).abs().sum()
    except:
        encrypted_count = 0

    logger.info(
        f"Encrypted: {encrypted_count} Non-Encrypted: {non_encrypted_count}"
    )

    gc_collect()

    if encrypted_count == 0 or non_encrypted_count == 0:
        return None

    return df


def combine_metrics(list_of_lists: List[List[float]]) -> List[float]:
    if len(list_of_lists) == 0:
        return []
    outlist = [0.0] * len(list_of_lists[0])
    count = 0
    for thelist in list_of_lists:
        count += 1
        for i in range(len(thelist)):
            outlist[i] += thelist[i]
    for i in range(len(outlist)):
        outlist[i] /= count
    return outlist


def evaluate(
    name: str,
    data: pd.DataFrame,
    output_directory: str,
    feature_column_names: List[str],
    annotation_columns: List[str],
    n_jobs: int,
    folds: int = -1,
) -> Tuple[bool, List[float]]:
    # This layer loops over the 54 different combinations
    random_seed()

    list_of_combinations = [
        "exclude_plaintext_nonbase32",
        "exclude_plaintext_base32",
        "exclude_encrypted_v1",
        "exclude_encrypted_v2",
        "exclude_encrypted_base32",
        "exclude_encrypted_nonbase32",
        "exclude_webp",
        "exclude_nonwebp",
    ]

    combinations = [(True,), (False,)]
    for i in range(7):
        temp = []
        for e in combinations:
            et = e + (True,)
            temp.append(et)
            et = e + (False,)
            temp.append(et)
        combinations = temp

    all_metrics = []
    for n, combination in enumerate(
        tqdm.tqdm(
            combinations,
            desc=f"{name}: Combinations of file types:",
            colour="green",
        )
    ):
        random_seed()
        message = " ".join(
            [f"{e1}:{e2}" for e1, e2 in zip(list_of_combinations, combination)]
        )
        logger.opt(colors=True).info(
            "<yellow>- - - - - - - - - - - - - - - - - - - - - </>"
        )
        logger.opt(colors=True).info(f">> Combination {n:02d}: {message}")

        # Skip this condition as this is not a realistic combination
        if combination[0]:  # exclude_plaintext_nonbase32
            continue

        temp_data = trim_dataset(data, *combination)
        if temp_data is not None:
            temp_dir = output_directory + os.path.sep + f"run-{n}"

            # TODO: uncomment this if restarting. Delete the last worked upon
            # folder
            if os.path.exists(temp_dir):
                continue

            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            logid2 = logger.add(
                temp_dir + os.path.sep + "log.log",
                backtrace=True,
                diagnose=True,
                level="INFO",
            )
            logger.info(
                f"*** Processing Combination {n:02d} combination = {message}"
            )
            comb_json_str = json.dumps(
                {e1: e2 for e1, e2 in zip(list_of_combinations, combination)}
            )
            logger.info(f"*** combination_json = {comb_json_str}")
            success, metric = evaluate_features(
                name=name,
                data=temp_data,
                output_directory=temp_dir,
                feature_column_names=feature_column_names,
                annotation_columns=annotation_columns,
                n_jobs=n_jobs,
                folds=folds,
            )
            logger.remove(logid2)
            all_metrics.append(metric)
            if not success:
                logger.error(
                    f"Fatal: Failed to process iteration {n} ... "
                    f"combination = {message}"
                )
                raise Exception(
                    f"Fatal: Failed to process iteration {n} ... "
                    f"combination = {message}"
                )
                return False, []
            else:
                logger.opt(colors=True).info(
                    f"<magenta>Combination {n:02d} done {metric=}</>"
                )
        else:
            logger.info(f"Combination {n:02d} had no elements")

    return True, combine_metrics(all_metrics)


def main() -> None:
    # dotenv.load_dotenv()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser("Run experiments")
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        required=True,
        help="Input directory for data files.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "-nj", "--n-jobs", type=int, default=4, help="Number of jobs to run."
    )
    parser.add_argument(
        "-nf", "--n-folds", type=int, default=-1, help="Folds to run for"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_directory) or not os.path.isdir(
        args.input_directory
    ):
        raise Exception(f"Path {args.input_directory} does not exist")
    if not os.path.exists(args.output_directory) or not os.path.isdir(
        args.output_directory
    ):
        os.mkdir(args.output_directory)

    log_file = f"{args.output_directory}{os.path.sep}log.log"
    if os.path.exists(log_file):
        os.unlink(log_file)
    if os.path.exists(f"{log_file}.debug.log"):
        os.unlink(f"{log_file}.debug.log")
    logger.remove()
    logger.add(log_file, backtrace=True, diagnose=True, level="INFO")
    logger.add(
        f"{log_file}.debug.log", backtrace=True, diagnose=True, level="DEBUG"
    )
    logger.add(sys.stderr, backtrace=True, diagnose=True, level="ERROR")
    logger.opt(colors=True).info(f"<blue>Running with {args}</>")

    random_seed()

    data = load_data(args.input_directory)

    annot_columns = get_annotation_columns(data)

    for n, (fsname, fscolumns) in enumerate(
        tqdm.tqdm(
            get_columns_and_types(data).items(),
            desc="Iterating through feature sets",
            colour="blue",
        )
    ):
        temp_output_dir = (
            f"{args.output_directory}" + os.path.sep + f"{fsname}"
        )

        # TODO: uncomment this if restarting. May need to modify the list
        if fsname in {"baseline-only", "advanced-only", "fourier-only"}:
            continue
        # TODO: uncomment if required
        if os.path.exists(temp_output_dir) and os.path.isdir(temp_output_dir):
            logger.info(f"{temp_output_dir} exists, skipping feature set")
            continue

        print_text = (
            f"******** Processing {fsname} and writing into {temp_output_dir}"
        )
        logger.opt(colors=True).info(f"<green>{print_text}</>")
        logger.opt(colors=True).info(f"<green>{'-' * len(print_text)}</>")

        columns = copy.copy(fscolumns)
        columns += annot_columns
        columns += ["is_encrypted"]

        if not os.path.exists(temp_output_dir):
            os.mkdir(temp_output_dir)
        t1 = time.perf_counter()

        logger.info(f"**** {n:02d}. Started evaluating feature set: {fsname}")
        logid = logger.add(
            temp_output_dir + os.path.sep + "log.log",
            backtrace=True,
            diagnose=True,
            level="INFO",
        )
        # print(fscolumns)
        retval, metrics = evaluate(
            name=fsname,
            data=data[columns].copy(),
            output_directory=temp_output_dir,
            feature_column_names=fscolumns,
            annotation_columns=annot_columns,
            n_jobs=args.n_jobs,
            folds=args.n_folds,
        )
        logger.remove(logid)

        t2 = time.perf_counter()
        logger.info(
            f"{n:02d}. Completed running feature {fsname} in {t2 - t1} seconds"
        )
        logger.opt(colors=True).info(f"<magenta>{fsname=} {metrics=}</>")
        print("*" * 80)
        print(f"{fsname=} {metrics=}")
        print()
        logger.opt(colors=True).info(
            "<green>*******************************************************</>"
        )
        gc_collect()
        if not retval:
            logger.error(
                f"Error evaluating feature set '{fsname}', metrics = {metrics}"
            )
            break
        logger.opt(colors=True).info(f"<magenta>{fsname} : {metrics}</>")
    print("Finished... OK")
    logger.opt(colors=True).info(f"<green>Finished... OK</>")


if "__main__" == __name__:
    main()
