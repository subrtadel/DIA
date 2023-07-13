import os
import re

path2raw = './dataset/raw_data/'
path2clean = './dataset/data/'

os.makedirs(path2clean, exist_ok=True)

def separate_filenames(path):
    all_raw_files = os.listdir(path)
    raw_files_to_rename = [fn for fn in all_raw_files if re.match('[0-9]{5}.*', fn) is None]
    raw_files_okay = [fn for fn in all_raw_files if not re.match('[0-9]{5}.*', fn) is None]
    return raw_files_to_rename, raw_files_okay

def determine_file_count(raw_files_okay):
    okay_file_names_numbers = [int(rfn.split('.')[0]) for rfn in raw_files_okay]
    old_file_count = 0
    if len(okay_file_names_numbers) != 0:
        old_file_count = max(okay_file_names_numbers)
        file_count = 1 + old_file_count
    return file_count

raw_files_to_rename, raw_files_okay = separate_filenames(path2raw)

file_count = determine_file_count(raw_files_okay)

for fn in raw_files_to_rename:
    # rename new files to predefined format
    suffix = fn.split('.')[-1]
    new_file_name = f'{file_count:05d}.{suffix}'
    os.rename(os.path.join(path2raw,fn), os.path.join(path2raw, new_file_name))
    file_count += 1

    # pad images
    os.system(f'convert {os.path.join(path2raw, new_file_name)} -virtual-pixel black -set option:distort:viewport "%[fx:max(w,h)]x%[fx:max(w,h)]-%[fx:max((h-w)/2,0)]-%[fx:max((w-h)/2,0)]" -filter point -distort SRT 0  +repage  {os.path.join(path2clean, new_file_name)}')
    # resize
    os.system(f'convert {os.path.join(path2clean, new_file_name)} -resize 512x512   {os.path.join(path2clean, new_file_name)}')


print(f'Done. {len(raw_files_to_rename)} new files were added.')