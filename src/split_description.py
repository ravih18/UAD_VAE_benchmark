import pandas as pd

tsv_paths = {}

for i in range(6):
    tsv_paths[f"split-{i}_train"] = f"/gpfswork/rech/krk/commun/anomdetect/tsv_files/6_fold/split-{i}/train.tsv"
    tsv_paths[f"split-{i}_validation"] = f"/gpfswork/rech/krk/commun/anomdetect/tsv_files/6_fold/split-{i}/validation_baseline.tsv"

tsv_paths["-_CN test"] = "/gpfswork/rech/krk/commun/anomdetect/tsv_files/CN-test_baseline.tsv"
tsv_paths["-_AD test"] = "/gpfswork/rech/krk/usy14zi/vae_benchmark/adni_tsv/deep_learning_exp/test/AD_baseline.tsv"

for name, tsv_path in tsv_paths.items():
    tsv_df = pd.read_csv(tsv_path, delimiter="\t")
    data_split, data_set = name.split("_")
    age_mean = round(tsv_df["age"].mean(), 1)
    age_std = round(tsv_df["age"].std(), 1)
    age = f"${age_mean} \pm {age_std}$"
    age_min = tsv_df["age"].min()
    age_max = tsv_df["age"].max()
    nb_datapoints = len(tsv_df)
    nb_subjects = len(tsv_df["participant_id"].unique())
    nb_M = len(tsv_df.drop_duplicates(subset=["participant_id"], keep="first")[tsv_df["sex"] == "M"])
    nb_F = len(tsv_df.drop_duplicates(subset=["participant_id"], keep="first")[tsv_df["sex"] == "F"])
    row = {
        "Fold": data_split.split("-")[1],
        "Set": data_set,
        "# subjects (M, F)": f"{nb_subjects} ({nb_M}, {nb_F})",
        "# images": nb_datapoints,
        f"Avg age ($\pm$ SD)": f"{age_mean} $\pm$ {age_std}",
        "Age range": f"{age_min}, {age_max}"
    }
    df = df.append(row, ignore_index=True)

print("\\begin{table}[!htb]\n\\caption{}\n\\centering\n")
print(df.to_latex(index=False))
print("\\label{tab:fold-description}\n\end{table}")
