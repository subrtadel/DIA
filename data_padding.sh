#!/bin/bash
raw_data_list=($(ls -1 ./dataset/raw_data/))
clean_data_list=($(ls -1 ./dataset/data/))

# TODO: asi lepsi najit maximum a pak jen pridavat.....
image_counter=${#clean_data_list[@]}
echo $image_counter

for im_name in "${raw_data_list[@]}" 
do
  new_im_name=$(printf %06d $image_counter)
  echo $new_im_name
  convert "./dataset/raw_data/${im_name}" -virtual-pixel black -set option:distort:viewport \
     "%[fx:max(w,h)]x%[fx:max(w,h)]-%[fx:max((h-w)/2,0)]-%[fx:max((w-h)/2,0)]" \
     -filter point -distort SRT 0  +repage  ./dataset/raw_data/${im_name}
  convert "./dataset/raw_data/${im_name}" -resize 512x512 ./dataset/data/${im_name}
  image_counter=$((image_counter+1))
done
