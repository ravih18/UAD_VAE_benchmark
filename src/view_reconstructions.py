# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.transform import resize
from scipy import ndimage
import os
from os import path


# %%
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1
            / 2
            * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


# %%
def get_images(X, Y):
    sag, cor, ax = map(lambda x: x // 2, input_cn_img.shape)

    # Sagital setting
    X_sag = resize(np.rot90(X[sag, :, :]), (80, 80))
    Y_sag = resize(np.rot90(Y[sag, :, :]), (80, 80))

    # Coronal setting
    X_cor = resize(np.rot90(X[:, cor, :]), (80, 80))
    Y_cor = resize(np.rot90(Y[:, cor, :]), (80, 80))

    # Axial setting
    X_ax = resize(np.rot90(X[:, :, ax]), (80, 80))
    Y_ax = resize(np.rot90(Y[:, :, ax]), (80, 80))

    return [Y_ax, Y_ax - X_ax, Y_sag, Y_sag - X_sag, Y_cor, Y_cor - X_cor]


# %%
sub = "sub-ADNI002S1280"
ses = "ses-M48"

input_file_name = f"{sub}_{ses}_image-0_input.pt"
output_file_name = f"{sub}_{ses}_image-0_output.pt"

models_dir = "/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/maps/"
model_list = {"Input": "Input", "empty": " "}
with open(
    "/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/models.txt", "r"
) as f:
    for line in f.readlines():
        line = line.strip("\n")
        model_list[line] = line
with open(
    "/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/models_0.txt", "r"
) as f:
    for line in f.readlines():
        line = line.strip("\n")
        model_list[line] = line
print(model_list)

# %%
print("creating blank img")
blank = np.zeros((80, 80))

# input_cn_img_path = path.join(
# models_dir, "MAPS_VAE", "split-0/best-loss/test_CN/tensors", input_file_name
# )
# input_hy_img_path = path.join(
#     models_dir, "MAPS_VAE", "split-0/best-loss/test_hypo_ad_30/tensors", input_file_name
# )
input_cn_img_path = path.join(
    models_dir,
    "MAPS_VAE_32_5_512",
    "split-0/best-loss/validation/tensors",
    input_file_name,
)

print("Reading input tensors")
input_cn_img = torch.load(input_cn_img_path).detach().numpy()[0]
# input_hy_img = torch.load(input_hy_img_path).detach().numpy()[0]

sag, cor, ax = map(lambda x: x // 2, input_cn_img.shape)

ax_cn = resize(np.rot90(input_cn_img[:, :, ax]), (80, 80))
sa_cn = resize(np.rot90(input_cn_img[sag, :, :]), (80, 80))
co_cn = resize(np.rot90(input_cn_img[:, cor, :]), (80, 80))

# ax_hy = resize(np.rot90(input_hy_img[:, :, 28]), (80, 80))
# sa_hy = resize(np.rot90(input_hy_img[44, :, :]), (80, 80))
# co_hy = resize(np.rot90(input_hy_img[:, 29, :]), (80, 80))

print("Making input images")
# input_images = [
#     ax_cn,
#     blank,
#     sa_cn,
#     blank,
#     co_cn,
#     blank,
#     blank,
#     ax_hy,
#     ax_cn - ax_hy,
#     sa_hy,
#     sa_cn - sa_hy,
#     co_hy,
#     co_cn - co_hy,
# ]
input_images = [
    ax_cn,
    blank,
    sa_cn,
    blank,
    co_cn,
    blank,
    blank,
    blank,
    blank,
    blank,
    blank,
    blank,
    blank,
]

image_list = [input_images, []]

# %%
print("Making images list")
for model in model_list.keys():
    if model != "Input" and model != "empty":
        maps_path = f"{models_dir}/MAPS_{model}"
        # out_cn_img_path = path.join(
        #     maps_path, "split-0/best-loss/test_CN/tensors", output_file_name
        # )
        # out_hy_img_path = path.join(
        #     maps_path, "split-0/best-loss/test_hypo_ad_30/tensors", output_file_name
        # )
        out_cn_img_path = path.join(
            maps_path, "split-0/best-loss/validation/tensors", output_file_name
        )

        out_cn_img = torch.load(out_cn_img_path).detach().numpy()[0]
        # out_hy_img = torch.load(out_hy_img_path).detach().numpy()[0]

        # imgs = (
        #     get_images(input_cn_img, out_cn_img)
        #     + [blank]
        #     + get_images(input_hy_img, out_hy_img)
        # )
        imgs = (
            get_images(input_cn_img, out_cn_img) + [blank] + [blank for _ in range(6)]
        )

        image_list.append(imgs)

# labels_x = [
#     "",
#     "",
#     "                Reconstruction from CN subject of the test set",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "                  Reconstruction from the same subject after\n                   simulating AD (30%)",
#     "",
#     "",
#     "",
# ]

labels_x = [
    "",
    "",
    "                Reconstruction from CN subject of the validation set",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
]

# %%
print("Making figure")
fig, axes = plt.subplots(
    len(model_list),
    13,
    figsize=(27, 45),
    gridspec_kw={
        "wspace": 0,
        "hspace": -0.152,
        "width_ratios": [1] * 13,
        "height_ratios": [1, 0.35] + [1] * (len(model_list) - 2),
    },
)

for i, model_name in enumerate(model_list.values()):
    axes[i][0].set_ylabel(model_name, rotation=0, fontsize=20, ha="right")
    for j in range(13):
        # Remove axis ticks
        axes[i][j].get_xaxis().set_ticks([])
        axes[i][j].get_yaxis().set_ticks([])

        # Set colorm map and norm
        if j in [1, 3, 5, 8, 10, 12]:
            cmap = "seismic"
            norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)
        else:
            cmap = "nipy_spectral"
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
        if i == 1 or j == 6:
            axes[i][j].set_visible(False)
        else:
            axes[i][j].imshow(image_list[i][j], cmap=cmap, norm=norm)

        if i == 0:
            axes[i][j].set_xlabel(labels_x[j], fontsize=24, labelpad=5.0)
            axes[i][j].xaxis.set_label_position("top")

cax1, kw1 = mpl.colorbar.make_axes(
    [ax for ax in axes.flat], location="bottom", shrink=1.0, aspect=60, pad=-0.1
)
cbar1 = plt.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap="nipy_spectral"
    ),
    cax=cax1,
    **kw1,
)
cbar1.ax.tick_params(labelsize=20)

cax2, kw2 = mpl.colorbar.make_axes(
    cax1, location="bottom", shrink=1.0, aspect=60, pad=0.08
)
cbar2 = plt.colorbar(
    mpl.cm.ScalarMappable(
        norm=MidpointNormalize(vmin=-1, vmax=1, midpoint=0), cmap="seismic"
    ),
    cax=cax2,
    **kw2,
)
cbar2.ax.tick_params(labelsize=20)

plt.savefig(
    f"/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/plots/reconstruction_plot_{sub}.png",
    bbox_inches="tight",
    pad_inches=0,
)

plt.show()

# %%
