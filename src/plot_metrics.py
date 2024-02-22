# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 7)
plt.rcParams["figure.autolayout"] = True

# %%

results_dir = "/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/results"
df = pd.read_csv(
    f"{results_dir}/eval_prediction.tsv",
    delimiter="\t",
)


# %%
def plot_metrics(
    data: pd.DataFrame,
    group: str,
    split: int,
    metric: str,
    results_dir: str,
    increasing: bool = False,
):
    if increasing:
        my_order = (
            df[df["group"] == "validation"]
            .groupby(by=["model"])[metric]
            .median()
            .sort_values()
            .iloc[::-1]
            .index
        )
    else:
        my_order = sorted(list(df["model"].unique()))
    if group and split:
        filtered_data = data[(data["group"] == group) & (data["split_nb"] == split)]
        figname = f"{group}-split_{split}-{metric}"
    elif group is not None:
        filtered_data = data[(data["group"] == group)]
        figname = f"{group}-{metric}"
    elif split is not None:
        filtered_data = data[(data["split_nb"] == split)]
        figname = f"split_{split}-{metric}"
    else:
        raise ("ValueError: group and split can't both be None")
    plot = sns.boxplot(
        data=filtered_data,
        x="model",
        y=metric,
        hue="group" if group is None else "split_nb",
        order=my_order,
    )
    plt.xticks(rotation=45)
    # plt.savefig(f'{results_dir}/{figname}.pdf')
    plt.savefig(f"{results_dir}/{figname}.png")
    plt.show()


# %%
for metric in ["MSE", "MAE", "PSNR", "SSIM"]:
    for split in [0, 1]:
        # plot_metrics(df, group=None, split=split, metric=metric, results_dir=results_dir, increasing=False)
        plot_metrics(
            df,
            group=None,
            split=split,
            metric=metric,
            results_dir=results_dir,
            increasing=True,
        )
    for group in ["validation"]:
        # plot_metrics(df, group=group, split=None, metric=metric, results_dir=results_dir, increasing=False)
        plot_metrics(
            df,
            group=group,
            split=None,
            metric=metric,
            results_dir=results_dir,
            increasing=True,
        )

# %%
