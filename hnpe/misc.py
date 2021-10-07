
import os


def make_label(meta_parameters):
    "This functions creates the basic label used when saving all files."
    keys = [
        "n_rd",
        str(meta_parameters["n_rd"]),
        "n_sr",
        str(meta_parameters["n_sr"]),
        "n_sf",
        str(meta_parameters["n_sf"]),
    ]

    label = meta_parameters["case"] + os.sep + "_".join(keys)

    return label
