base="/home/oliver/export/"
cmd="python -m main --input"


prefix="/home/oliver/storage/export"


for scale in 0.5 0.25 0.125;
do
  for image_size in 512 768 1024;
  do
      first=3
      if [ scale == 0.25 ] || [ scale == 0.125 ] 
      then
        first=2
      fi

      common="$common --image_size $image_size --log_dir /local/storage/logs/scales/$scale/$image_size --train_epochs 1 --no_load"

      $cmd "json --path $prefix/penguins.json" --model "fcn --first $first" $common   --run_name penguins
      $cmd "json --path $prefix/scallops_niwa.json" --model "fcn --first $first" $common   --run_name scallops
  done
done

for scale in 0.5 0.25 0.125;
do
  for image_size in 512 768 1024 2048;
  do
      first=3
      if [ scale == 0.25 ] || [ scale == 0.125 ] 
      then
        first=2
      fi

      common="$common --image_size $image_size --log_dir /local/storage/logs/scales/$scale/$image_size --train_epochs 1 --no_load"

      $cmd "json --path $prefix/seals.json" --model "fcn --square --first $first"  $common --run_name seals
      $cmd "json --path $prefix/apples.json" --model "fcn --square --first $first"  $common --run_name apples
  done
done