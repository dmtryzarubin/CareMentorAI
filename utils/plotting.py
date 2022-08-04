from typing import Dict, Iterable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchio as tio
from matplotlib.patches import Patch

LABEL_MAPPER = {
    1: "Left hemisphere | C (Caudate)",
    2: "Left hemisphere | L (Lentiformis)",
    3: "Left hemisphere | IC (Internal capsule)",
    4: "Left hemisphere | I (Island)",
    5: "Left hemisphere | M4",
    6: "Left hemisphere | M5",
    7: "Left hemisphere | M6",
    8: "Left hemisphere | M1",
    9: "Left hemisphere | M2",
    10: "Left hemisphere | M3",
    11: "Right hemisphere | C (Caudate)",
    12: "Right hemisphere | L (Lentiformis)",
    13: "Right hemisphere | IC (Internal capsule)",
    14: "Right hemisphere | I (Island)",
    15: "Right hemisphere | M4",
    16: "Right hemisphere | M5",
    17: "Right hemisphere | M6",
    18: "Right hemisphere | M1",
    19: "Right hemisphere | M2",
    20: "Right hemisphere | M3",
}

PALETTE = {
    k: v
    for k, v in zip(
        LABEL_MAPPER, sns.color_palette("pastel", 10) + sns.color_palette("bright", 10)
    )
}


class Visualizer:
    def __init__(
        self,
        subject: tio.Subject,
        label_mapper: Dict[int, str],
        palette: Dict[int, Tuple[int]],
        plot_shape: Tuple[int] = (512, 512),
    ) -> None:
        """
        Class for one volume visualisation.

        :param subject: TorchIO subject with image, mask and Optional[prediction]
        :param label_mapper: Dict that maps int mask labels to strings
        :param palette: Dict with colors for each int label
        :param plot_shape: Resize images to that shape before plotting, defaults to (512, 512)
        """
        self.plot_shape = plot_shape
        self.palette = palette
        self.image = subject.image.numpy().squeeze()
        self.mask = subject.mask.numpy().squeeze().astype("uint8")
        self.classes = list(label_mapper)
        self.total_vol = np.count_nonzero(self.mask > 0)
        self.mask_classes_vol = {
            cls: np.count_nonzero(self.mask == cls) for cls in self.classes
        }
        if "pred" in subject.keys():
            self.pred = subject.pred.numpy().squeeze().astype("uint8")
            self.pred_classes_vol = {
                cls: np.count_nonzero(self.pred == cls) for cls in self.classes
            }
            self.nrows = 3
        else:
            self.pred = None
            self.nrows = 2

    @staticmethod
    def getSlice(axis: int = 0, idx: int = 0, ndim: int = 3) -> tuple:
        """
        Calculates array slice for getting an image slice on given axis and index.
        """
        return tuple(
            [slice(idx, idx + 1) if i == axis else slice(None) for i in range(ndim)]
        )

    def plotAnnotationSlice(self, axis: int, idx: int = 0, **kwargs) -> None:
        """
        Plots image and mask Optional[prediction] slice from given subject volumes.

        :param axis: H or W or D axis to get slice from
        :param idx: Index of a slice from give axis, defaults to 0
        """
        figsize = kwargs.get("figsize", (14, 9))
        fig, axs = plt.subplots(1, self.nrows, figsize=figsize)
        if self.pred is not None:
            items = ["image", "mask", "pred"]
        else:
            items = ["image", "mask"]

        sl = self.getSlice(axis, idx)

        legend_elements = [
            Patch(facecolor=self.palette[l], label=LABEL_MAPPER[l])
            for l in self.classes
        ]
        cmap = kwargs.get("cmap", "gray")
        loc = kwargs.get("loc", "upper right")
        bbox_to_anchor = kwargs.get("bbox_to_anchor", (1.8, 0.98))
        legend_fontsize = kwargs.get("legend_fontsize", 8)

        for idx, item in enumerate(items):
            cur_ = cv2.resize(
                getattr(self, item)[sl].squeeze(),
                self.plot_shape,
                interpolation=cv2.INTER_AREA,
            )
            if item != "image":
                cur_ = self.getColorMask(cur_, self.palette, self.classes)
            axs[idx].imshow(cur_, cmap=cmap)
            axs[idx].set_title(f"{item}-Axis:{axis}")

        axs[idx].legend(
            handles=legend_elements,
            loc="upper right" if loc is None else loc,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_fontsize,
        )
        plt.show()

    @staticmethod
    def padArr(arr: np.ndarray, tile_size=(512, 512)) -> np.ndarray:
        """
        Padds 2d-array to a given shape
        """
        y_, x_ = tile_size
        y, x = arr.shape
        y_pad = y_ - y
        x_pad = x_ - x
        img_padded = np.pad(
            arr,
            ((y_pad // 2, y_pad // 2), (x_pad // 2, x_pad // 2)),
            mode="constant",
            constant_values=0,
        )
        return img_padded

    @staticmethod
    def getColorMask(
        mask: np.ndarray,
        palette: Dict[int, Tuple[int]],
        labels: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """
        Creates colorful mask from given mask and palette.

        :param mask: Mask, where [i, j] pixel corresponds to one of the labels
        :param palette: Palette that encodes int label to Tuple[int].
        :param labels: Unique mask labrls, defaults to None
        :return: 3dim mask with shape [H, W, 3] for plotting
        """
        assert mask.ndim == 2  # src mask expected to be 2 dimensional slice
        if labels is None:
            labels = np.unique(mask)[1:]  # skip background
        mask_ = np.dstack([mask] * 3).astype("float")
        for label in labels:
            mask_[(mask_ == label).all(-1)] = palette[label]
        return mask_
