
base="/home/oliver/export/"
cmd="python -m main"

common="--lr 0.001  --input  'coco --path /storage/storage/coco' --model 'fcn --features 128' --no_load --train_epochs 60 --crop_boxes --image_size 512 --batch_size 16 --epoch_size 8192 --log_dir /local/storage/logs/multiclass_coco/" 
prefix="/local/storage/export"

subset1="cow,sheep,cat,dog"
subset2="zebra,giraffe,elephant,bear"
subset3="hotdog,pizza,donut,cake"
subset4="cup,fork,knife,spoon"


for subset in $subset1 $subset2 $subset3 $subset4; 
do 
  bash -c "$cmd  $common --run_name $subset  --subset $subset"
  for i in $(echo $subset | sed "s/,/ /g")
  do 
    bash -c "$cmd  $common --run_name $i --subset $subset --keep_classes $i"
  done
done


