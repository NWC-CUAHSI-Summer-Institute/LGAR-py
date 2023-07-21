import argparse
from omegaconf import OmegaConf
from pathlib import Path

cwd = Path.cwd()
main_cfg_path = cwd / "dpLGAR"
data_cfg_path = cwd / "dpLGAR/flat_files/config"
models_cfg_path = cwd / "dpLGAR/models/config"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--forcing_file", type=str, help="Path to the forcing file")
    parser.add_argument("--soil_data_file", type=str, help="Path to the soil flat_files file")
    parser.add_argument("--num_soil_layers", type=int, help="Number of soil layers")
    parser.add_argument("--texture_per_layer", type=str, help="Texture for each soil layer (comma-separated)")
    parser.add_argument("--thickness_per_layer", type=str, help="Thickness for each soil layer (comma-separated)")

    args = parser.parse_args()

    texture_per_layer = args.texture_per_layer.split(',')
    thickness_per_layer = args.thickness_per_layer.split(',')

    main_config = OmegaConf.create()
    data_config = OmegaConf.create()
    model_config = OmegaConf.create()

    data_config["forcing_file"] = args.forcing_file
    data_config["soil_params_file"] = args.args.soil_data_file
    data_config["num_soil_layers"] = args.args.num_soil_layers
    data_config["forcing_file"] = args.forcing_file
    data_config["layer_thickness"] = thickness_per_layer
    print(f"Texture per layer: {texture_per_layer}")


if __name__ == "__main__":
    main()