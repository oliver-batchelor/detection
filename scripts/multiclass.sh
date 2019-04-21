
base="/home/oliver/export/"
cmd="python -m main"

common="--lr 0.01  --input  'voc --path /local/storage/voc' --no_load --train_epochs 40 --crop_boxes --image_size 512 --batch_size 16 --epoch_size 8192 --log_dir /local/storage/logs/multiclass/" 
prefix="/local/storage/export"

subset1="cow,sheep,cat,dog"
subset2="motorbike,bicycle,car,bus"

bash -c "$cmd  $common --run_name subset1 --subset $subset1"
bash -c "$cmd  $common --run_name cow --subset $subset1 --keep_classes cow"
bash -c "$cmd  $common --run_name sheep --subset $subset1 --keep_classes sheep"
bash -c "$cmd  $common --run_name cat --subset $subset1 --keep_classes cat"
bash -c "$cmd  $common --run_name dog --subset $subset1 --keep_classes dog"

bash -c "$cmd  $common --run_name subset2 --subset $subset2"
bash -c "$cmd  $common --run_name motorbike --subset $subset2 --keep_classes motorbike"
bash -c "$cmd  $common --run_name bicycle --subset $subset2 --keep_classes bicycle"
bash -c "$cmd  $common --run_name car --subset $subset2 --keep_classes car"
bash -c "$cmd  $common --run_name bus --subset $subset2 --keep_classes bus"
