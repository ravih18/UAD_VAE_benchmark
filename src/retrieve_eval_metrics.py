import os
import glob
import pandas as pd
import argparse

from typing import Optional


# Custom rounding function to round to a specific number of decimal places
def custom_round(value: float, significant_figures: int = 5) -> float:
    sci_notation = "{:.{dec}e}".format(value, dec=significant_figures - 1)
    return float(sci_notation)


def extract_data(
    file_path: str,
    selected_models: Optional[list[str]],
    selected_splits: Optional[list[int]],
    selected_groups: Optional[list[str]],
) -> Optional[pd.DataFrame]:
    # Extract model, split_nb, and group from the file path
    parts = file_path.split(os.path.sep)
    model = parts[-5].strip("MAPS_")
    split_nb = parts[-4].split("-")[1]
    group = parts[-2]

    # Check if the extracted model, split, and group are in the selected lists
    if (
        (not selected_models or model in selected_models)
        and (not selected_splits or split_nb in selected_splits)
        and (not selected_groups or group in selected_groups)
    ):
        # Read the TSV file into a DataFrame
        data = pd.read_csv(file_path, sep="\t")
        data.rename(columns={"Unnamed: 0": "statistic"}, inplace=True)

        # Apply custom rounding only to numeric columns
        data = data.applymap(
            lambda x: custom_round(x, significant_figures=5)
            if isinstance(x, (int, float))
            else x
        )

        # Add model, split_nb, and group columns to the DataFrame
        data["model"] = model
        data["split_nb"] = split_nb
        data["group"] = group

        # Rearrange columns with model, split_nb, and group as the first columns
        columns = ["model", "split_nb", "group"] + [
            col for col in data if col not in ["model", "split_nb", "group"]
        ]
        data = data[columns]

        return data

    return None


def combine_tsv_files(
    maps_dir: str,
    combined_file_path: str,
    selected_models: list[str] = None,
    selected_splits: list[int] = None,
    selected_groups: list[str] = None,
    validation: bool = False,
    file_extension: str = "metrics.tsv",
):
    # Create a list of file paths matching the pattern
    tsv_files = glob.glob(
        f"{maps_dir}/*/split-*/best-loss/*/*_image_level_{file_extension}"
    )

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate through the found TSV files and combine only the selected ones
    for file_path in tsv_files:
        data = extract_data(
            file_path, selected_models, selected_splits, selected_groups
        )

        if data is not None:
            # Append the data to the combined_data DataFrame
            combined_data = pd.concat([combined_data, data])

    # Save the combined data to a new TSV file
    combined_data.to_csv(combined_file_path, sep="\t", index=False)

    print(f"Combined data for {file_extension} saved to {combined_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine TSV files with optional selection of models, splits, and groups."
    )
    parser.add_argument(
        "maps_dir", type=str, help="Directory where TSV files are located"
    )
    parser.add_argument(
        "combined_file_path", type=str, help="Path to save the combined data"
    )
    parser.add_argument(
        "--selected_models", nargs="*", default=None, help="Selected models"
    )
    parser.add_argument(
        "--selected_splits", nargs="*", default=None, help="Selected splits"
    )
    parser.add_argument(
        "--selected_groups", nargs="*", default=None, help="Selected groups"
    )

    args = parser.parse_args()

    # Run the script for "metrics.tsv" files
    combine_tsv_files(
        args.maps_dir,
        args.combined_file_path,
        args.selected_models,
        args.selected_splits,
        args.selected_groups,
        file_extension="metrics.tsv",
    )

    # Run the script for "prediction.tsv" files
    combine_tsv_files(
        args.maps_dir,
        args.combined_file_path.replace("metrics.tsv", "prediction.tsv"),
        args.selected_models,
        args.selected_splits,
        args.selected_groups,
        file_extension="prediction.tsv",
    )
