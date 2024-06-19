unzip ALTS.zip
unzip WORD.zip
rm WORD.zip
rm ALTS.zip
mv WORD ./ALTS/data_dir/
cd ALTS
python monai_train_v1.py
rm -rf ./data_dir/WORD
cd ../
tar -czvf proj_d5f48r5e20000.tar.gz ALTS/proj_dir/
rm -rf ALTS
