from torch import Tensor
import torch
import numpy as np
from typing import Sequence, Union
import importlib
if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')

def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)


def to_tensor(
    data: Union[Tensor, np.ndarray, Sequence, int,
                float]) -> Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')