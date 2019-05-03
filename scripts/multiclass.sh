
base="/home/oliver/export/"
cmd="python -m main"

common="--lr 0.01 --model 'fcn --features 128'  --input  'voc --path /local/storage/voc' --no_load --train_epochs 40 --crop_boxes --image_size 512 --batch_size 16 --epoch_size 8192 --log_dir /local/storage/logs/multiclass_128/" 
prefix="/local/storage/export"

subset1="cow,sheep,cat,dog"
subset2="motorbike,bicycle,car,bus"

for subset in $subset1 $subset2 $subset3 $subset4; 
do 
  bash -c "$cmd  $common --run_name $subset  --subset $subset"
  for i in $(echo $subset | sed "s/,/ /g")
  do 
    bash -c "$cmd  $common --run_name $i --subset $subset --keep_classes $i"
  done
done