import os
from shutil import copyfile

def overwrite_config(source_name, target_name = "config"):
    # Overwrite old config file with new one on the fly
    # Source and target name does not use ".yaml" at the end
    
    src = "configs/" + source_name + ".yaml"
    dst = "configs/" + target_name + ".yaml"
    os.remove(dst)
    copyfile(src, dst)

    return