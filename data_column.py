from abc import ABC, abstractmethod
import csv
import os
from typing import Generic, TypeVar

import config

DataType = TypeVar('DataType')


class DataColumn(ABC, Generic[DataType]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def names(self, base: config.BoneType) -> list[str]:
        ...

    @abstractmethod
    def values(self, base: config.BoneType, i: int) -> list[DataType]:
        ...


class DOFDataColumn(DataColumn[float]):
    def __init__(self, dof) -> None:
        super().__init__()
        self.y_tx, self.y_ty, self.y_tz, self.y_rx, self.y_ry, self.y_rz = dof

    def names(self, base: config.BoneType) -> list[str]:
        return ['Translation X', 'Translation Y', 'Translation Z',
                'Rotation X', 'Rotation Y', 'Rotation Z']

    def values(self, base: config.BoneType, i: int) -> list[float]:
        _ = base  # unused
        return [self.y_tx[i],
                self.y_ty[i],
                self.y_tz[i],
                self.y_rx[i],
                self.y_ry[i],
                self.y_rz[i]]


class AreaCurveDataColumn(DataColumn[float]):
    def __init__(self, area_curve) -> None:
        super().__init__()
        self.area_curve = area_curve

    def names(self, base: config.BoneType) -> list[str]:
        return ['Area Medial', 'Area Lateral']

    def values(self, base: config.BoneType, i: int) -> list[float]:
        area_medial, area_lateral = self.area_curve[base]
        return [area_medial[i], area_lateral[i]]


class DeepestPointDataColumn(DataColumn[float | str]):
    def __init__(self, deepest_points) -> None:
        super().__init__()
        self.deepest_points = deepest_points

    def names(self, base: config.BoneType) -> list[str]:
        return ['Deepest Point Medial X', 'Deepest Point Medial Y', 'Deepest Point Medial Z',
                'Deepest Point Lateral X', 'Deepest Point Lateral Y', 'Deepest Point Lateral Z',]

    def values(self, base: config.BoneType, i: int) -> list[float | str]:
        dp = self.deepest_points[base]
        ans: list[str | float] = ['None'] * 6
        if dp[i] is None:
            return ans
        for j in range(2):
            if dp[i][j] is not None:
                for k in range(3):
                    ans[j * 3 + k] = dp[i][j][k]
        return ans


class DeepestPoint2DDataColumn(DataColumn[float | str]):
    def __init__(self, deepest_points_2d) -> None:
        super().__init__()
        self.deepest_points_2d = deepest_points_2d

    def names(self, base: config.BoneType) -> list[str]:
        return ['Deepest Point 2D Medial X', 'Deepest Point 2D Medial Y',
                'Deepest Point 2D Lateral X', 'Deepest Point 2D Lateral Y']

    def values(self, base: config.BoneType, i: int) -> list[float | str]:
        dp2d = self.deepest_points_2d[base]
        ans: list[str | float] = ['None'] * 4
        for j, side in enumerate(['Medial', 'Lateral']):
            if dp2d[side] is not None and dp2d[side][i] is not None:
                for k in range(2):
                    ans[j * 2 + k] = dp2d[side][i][k]
        return ans


class MaxDepthDataColumn(DataColumn[float]):
    def __init__(self, max_depth_curve) -> None:
        super().__init__()
        self.max_depth_curve = max_depth_curve

    def names(self, base: config.BoneType) -> list[str]:
        return ['Depth', 'Depth Medial', 'Depth Lateral']

    def values(self, base: config.BoneType, i: int) -> list[float]:
        curve = self.max_depth_curve.get(
            base) if self.max_depth_curve is not None else None
        if curve is None:
            return [0.0, 0.0, 0.0]
        depth, depth_medial, depth_lateral = curve
        return [depth[i] if depth[i] is not None else 0.0,
                depth_medial[i] if depth_medial[i] is not None else 0.0,
                depth_lateral[i] if depth_lateral[i] is not None else 0.0]


class DeformityDataColumn(DataColumn[float]):
    def __init__(self, deformity_curve, label: str = '') -> None:
        super().__init__()
        self.deformity_curve = deformity_curve
        self.label = label.strip()

    def names(self, base: config.BoneType) -> list[str]:
        prefix = f'{self.label} ' if self.label else ''
        return [f'{prefix}Deformity Medial', f'{prefix}Deformity Lateral']

    def values(self, base: config.BoneType, i: int) -> list[float]:
        curve = self.deformity_curve.get(
            base) if self.deformity_curve is not None else None
        if curve is None:
            return [0.0, 0.0]
        medial, lateral = curve
        return [medial[i] if medial is not None and medial[i] is not None else 0.0,
                lateral[i] if lateral is not None and lateral[i] is not None else 0.0]


class FixedPoints2DDataColumn(DataColumn[float | str]):
    def __init__(self, fixed_points_2d) -> None:
        super().__init__()
        self.fixed_points_2d = fixed_points_2d

    def names(self, base: config.BoneType) -> list[str]:
        if self.fixed_points_2d is None or base not in self.fixed_points_2d:
            return []
        res: list[str] = []
        for name in sorted(self.fixed_points_2d[base].keys()):
            res.append(f"{name} X")
            res.append(f"{name} Y")
        return res

    def values(self, base: config.BoneType, i: int) -> list[float | str]:
        points = self.fixed_points_2d.get(
            base, {}) if self.fixed_points_2d is not None else {}
        ans: list[float | str] = []
        for base_name in sorted(points.keys()):
            arr = points.get(base_name)
            if arr is not None and i < len(arr) and arr[i] is not None:
                ans.append(arr[i][0])
                ans.append(arr[i][1])
            else:
                ans.extend(['None', 'None'])
        return ans


def dump_all_data(
        task_dof_data_raw, task_dof_data_smoothed, task_area_curve, task_frame_deepest_points, task_max_depth_curve,
        task_femur_deformity_curve, task_tibia_deformity_curve, task_deformity_curve, task_plot_deepest_points, task_plot_fixed_points):
    dof = task_dof_data_smoothed if config.MOVEMENT_SMOOTH else task_dof_data_raw
    n = len(dof[0]) if dof is not None else 0
    columns = [
        DOFDataColumn(dof),
        AreaCurveDataColumn(task_area_curve),
        DeepestPointDataColumn(task_frame_deepest_points),
        DeepestPoint2DDataColumn(task_plot_deepest_points),
        MaxDepthDataColumn(task_max_depth_curve),
        DeformityDataColumn(task_femur_deformity_curve, label='Femur'),
        DeformityDataColumn(task_tibia_deformity_curve, label='Tibia'),
        DeformityDataColumn(task_deformity_curve, label=''),
        FixedPoints2DDataColumn(task_plot_fixed_points),
    ]
    for base in [config.BoneType.FEMUR, config.BoneType.TIBIA]:
        header = ['Frame']
        for col in columns:
            header.extend(col.names(base))
        csv_path = os.path.join(config.OUTPUT_DIRECTORY,
                                f'{base.value}_all_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(
                csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')
            writer.writerow(header)
            for i in range(n):
                row = [i + 1]
                for col in columns:
                    row.extend(col.values(base, i))
                writer.writerow(row)
