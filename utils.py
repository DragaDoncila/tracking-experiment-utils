import yaml

def get_scale(
    ds_name, 
    scale_config_path='/home/ddon0001/PhD/data/cell_tracking_challenge/scales.yaml'
):
    with open(scale_config_path, 'r') as f:
        scales = yaml.load(f, Loader=yaml.FullLoader)
    return scales[ds_name]['pixel_scale']