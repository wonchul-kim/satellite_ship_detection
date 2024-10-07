from ultralytics.data.split_dota import split_test, split_trainval

# split train and val set, with labels.
gap = 100
crop_size = 640
split_trainval(
    data_root="/HDD/datasets/public/dota/ship/v1_ship_yolo_obb",
    save_dir=f"/HDD/datasets/public/dota/ship/v1_ship_yolo_obb_split_{crop_size}_{gap}",
    crop_size=crop_size,
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=gap,
)
