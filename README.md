# arirang_detect
dacon arirang objecte detection

1. download data from dacon
2. notebook/read_json.ipynb 를 실행하여 라벨을 csv로 바꿔준다.
3. pretrained model download
4. python src/main.py

## albumentation image augmentation 문제 발생

<p>bbox가 0 ~ 1.0 구간을 벗어나면 error가 발생하는 문제<br>
albumentations/augmentations/bbox_utils.py의 check_bbox를 아래와 같이 수정</p>

```python
def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1:
            bbox = np.clip(bbox, 0., 1.)
            # raise ValueError(
            #     "Expected {name} for bbox {bbox} "
            #     "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            # )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))

```

## Models

| Variant | Download | mAP (val2017) | mAP (test-dev2017) | mAP (TF official val2017) | mAP (TF official test-dev2017) |
| --- | --- | :---: | :---: | :---: | :---: |
| lite0 | [tf_efficientdet_lite0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_lite0-f5f303a9.pth) | 32.0 | TBD | N/A | N/A |
| D0 | [efficientdet_d0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth) | 33.6 | TBD | 33.5 | 33.8 |
| D0 | [tf_efficientdet_d0.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-1851dfed.pth) | 34.2 | TBD | 34.3 | 34.6 |
| D1 | [efficientdet_d1.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth) | 39.4 | 39.5 | 39.1 | 39.6 |
| D1 | [tf_efficientdet_d1.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_40-a30f94af.pth) | 40.1 | TBD | 40.2 | 40.5 |
| D2 | [tf_efficientdet_d2.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_43-8107aa99.pth) | 43.4 | TBD | 42.5 | 43 |
| D3 | [tf_efficientdet_d3.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth) | 47.1 | TBD | 47.2 | 47.5 |
| D4 | [tf_efficientdet_d4.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth) | 49.2 | TBD | 49.3 | 49.7 |
| D5 | [tf_efficientdet_d5.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth) | 51.2 | TBD | 51.2 | 51.5 |
| D6 | [tf_efficientdet_d6.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth) | 52.0 | TBD | 52.1 | 52.6 |
| D7 | [tf_efficientdet_d7.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth) | 53.1 | 53.4 | 53.4 | 53.7 |
| D7X | [tf_efficientdet_d7x.pth](https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7x-f390b87c.pth) | 54.3 | TBD | 54.4 | 55.1 |

_NOTE: Official scores for all modules now using soft-nms, but still using normal NMS here._

