from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import re

import evalys
from evalys.jobset import JobSet


def logs_to_df(folder: Path, file_types: list[str]):
    df = { file_type: pd.DataFrame() for file_type in file_types }

    for file_type in file_types:
        for experience in folder.glob(f"{file_type}-*.out"):
            exp_name = re.search(r"-([^-.]+).out", str(experience)).group(1)

            tmp_df = pd.read_csv(str(experience))
            tmp_df["schedule"] = exp_name

            df[file_type] = df[file_type].append(tmp_df)
        df[file_type].set_index("schedule", inplace=True)

    return df

def logs_to_jobset(folder: Path):
    jobsets = {}
    for experience in folder.glob(f"JobMonitor-*.out"):
        exp_name = re.search(r"-([^-.]+).out", str(experience)).group(1)
        jobsets[exp_name] = JobSet.from_csv(str(experience))

    return jobsets

