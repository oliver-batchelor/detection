base="/home/oliver/export/"
cmd="python -m main --input"

root="/local/storage"
prefix="$root/export"


for noise in 0 2 4 8 16;
do
  for offset in 0 2 4 8 16;
  do

      common="$common --box_noise $noise --box_offset $offset --log_dir $root/logs/noise/$noise/$offset --image_size 1024 --scale 0.5 --train_epochs 80 --no_load"

      $cmd "json --path $prefix/seals.json" --model "fcn --square"  $common --run_name seals
      $cmd "json --path $prefix/apples.json" --model "fcn --square"  $common  --run_name apples
      $cmd "json --path $prefix/penguins.json" --model "fcn" $common   --run_name penguins
      $cmd "json --path $prefix/scott_base.json" --model "fcn" $common   --run_name scallops
  done
done

