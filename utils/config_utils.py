import yaml
import os


def load_config(path, default_path=None, visited=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
        visited (set, optional): set of visited paths to detect cycles. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # Initialize visited set to detect cycles
    if visited is None:
        visited = set()
    
    # Normalize path to detect cycles correctly
    abs_path = os.path.abspath(path)
    
    # Detect cycle
    if abs_path in visited:
        raise ValueError(f"Circular inheritance detected: {abs_path} is already in the inheritance chain: {visited}")
    
    visited.add(abs_path)
    
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path, visited)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # merge per dataset cfg. and main cfg.
    # Remove inherit_from from cfg_special before merging to prevent it from being included in final config
    inherit_from_value = cfg_special.pop("inherit_from", None)
    update_recursive(cfg, cfg_special)
    # Restore inherit_from if needed (though it shouldn't be in final config)
    if inherit_from_value is not None:
        cfg_special["inherit_from"] = inherit_from_value

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively. dict1 get masked by dict2, and we retuen dict1.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
