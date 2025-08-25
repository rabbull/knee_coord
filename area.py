import os

import numpy as np
import trimesh
from matplotlib import pyplot as plt

import config
from bone import BoneCoordination


def plot_contact_area_curve(
        frame_contact_components: list[list[trimesh.Trimesh]],
        frame_coordinates: dict[config.BoneType, list[BoneCoordination]],
):
    res = {}
    for base in frame_coordinates:
        res[base] = do_plot_contact_area_curve(base.value, frame_contact_components, frame_coordinates[base])
    return res


def do_plot_contact_area_curve(
        name: str,
        frame_contact_components: list[list[trimesh.Trimesh]],
        frame_coordinates: list[BoneCoordination],
):
    medial, lateral = [], []
    for frame_index, (contact_components, coordination) \
            in enumerate(zip(frame_contact_components, frame_coordinates)):
        if len(contact_components) == 0:
            medial.append(0)
            lateral.append(0)
            continue

        medial_area, lateral_area = 0, 0
        for contact_component in contact_components:
            area = contact_component.area
            x_pos = coordination.project(contact_component.centroid)[0]
            if (x_pos > 0 and config.KNEE_SIDE == config.KneeSide.LEFT) \
                    or (x_pos < 0 and config.KNEE_SIDE == config.KneeSide.RIGHT):
                medial_area += area
            else:
                lateral_area += area
        medial.append(medial_area)
        lateral.append(lateral_area)

    fig, ax = plt.subplots()
    ax.plot(medial, label='Medial')
    ax.plot(lateral, label='Lateral')
    ax.legend()
    ax.set_title(f'Contact Area Curve - Base {name.capitalize()}')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'contact_area_base_{name}.jpg'))
    plt.close(fig)

    return medial, lateral

