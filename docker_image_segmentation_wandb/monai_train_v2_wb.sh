pip list
unzip ALTS.zip
unzip WORD.zip
rm WORD.zip
rm ALTS.zip
mv WORD ./ALTS/data_dir/
cd ALTS

# Run the Python script with the arguments passed to this shell script
python monai_train_v2_wb.py --model_depth $1 --model_start_channels $2 --model_num_res_units $3

tar_filename="proj_"$1"_"$2"_"$3"_"$4".tar.gz"

# Tar the project directory and name it uniquely
tar -czvf $tar_filename "ALTS/proj_dir/UNet_depth_${1}_channels_${2}_resunits_${3}"

# Cleanup
rm -rf ./data_dir/WORD
cd ..
rm -rf ALTS
