base="/home/oliver/export/"
cmd="python -m main --input"

root="/local/storage"
prefix="$root/export"


for noise in 0 2 4 8 16;
do
  for offset in 0 2 4 8 16;
  do

      common="$common --box_noise $noise --box_offset $offset --log_dir $root/logs/noise/$noise/$offset --image_size 1024 --scale 0.5 --train_epochs 40 --no_load"

      $cmd "json --path $prefix/scott_base.json" --model "fcn --square --first 2" $common   --run_name scott_base --scale 1 --image_size 400
      $cmd "json --path $prefix/branches.json" --model "fcn" $common   --run_name branches --scale 1 --image_size 320
      $cmd "json --path $prefix/seals.json" --model "fcn --square"  $common --run_name seals
      $cmd "json --path $prefix/apples_lincoln.json" --model "fcn --square"  $common  --run_name apples_lincoln
      $cmd "json --path $prefix/penguins.json" --model "fcn" $common   --run_name penguins
#      $cmd "json --path $prefix/scallops.json" --model "fcn" $common   --run_name scallops
  done
done




