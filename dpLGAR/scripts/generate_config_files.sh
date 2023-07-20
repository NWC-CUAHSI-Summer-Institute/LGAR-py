#!/bin/bash

echo "Please select an option:"
echo "1. Import from LGAR-C config file"
echo "2. Manually create configs"
echo "3. Exit"
read option

if [ $option -eq 1 ]
then
  echo "Please enter the path to the LGAR-C config file:"
  read config_path

  # Import the configuration from the provided file
  python import_config.py --config_path $config_path

elif [ $option -eq 2 ]
then
  # Manually create new configurations
  echo "Please enter the path to the forcing file:"
  read forcing_file
  echo "Please enter the path to the soil data file:"
  read soil_data_file
  echo "Please enter the number of soil layers:"
  read num_soil_layers

  # Assume the texture and thickness for each layer are entered as comma-separated lists
  echo "Please enter the texture for each soil layer (comma-separated):"
  read -a texture_per_layer
  echo "Please enter the thickness for each soil layer (comma-separated):"
  read -a thickness_per_layer

  # Convert arrays to comma-separated strings
  texture_per_layer=$(IFS=','; echo "${texture_per_layer[*]}")
  thickness_per_layer=$(IFS=','; echo "${thickness_per_layer[*]}")

  # Call the Python script with the arguments
  python my_script.py --forcing_file $forcing_file --soil_data_file $soil_data_file \
    --num_soil_layers $num_soil_layers --texture_per_layer $texture_per_layer \
    --thickness_per_layer $thickness_per_layer

elif [ $option -eq 3 ]
then
  echo "Exiting..."
  exit 0

else
  echo "Invalid option. Please enter 1, 2, or 3."
fi
