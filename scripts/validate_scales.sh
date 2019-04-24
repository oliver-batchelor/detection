base="/home/oliver/export/"
cmd="python -m main --input"

root="/local/storage"
prefix="$root/export"


for scale in 1 2 4 8;
do
  for image_size in 512 768 1024;
  do
      first=3
      if [ $scale -gt 2 ] 
      then
        first=2
      fi

      factor=`bc -l <<< 1/$scale`

      common="$common --scale $factor --image_size $image_size --log_dir $root/logs/scales2/$scale/$image_size --epoch_size 1024 --train_epochs 80 --no_load"

      $cmd "json --path $prefix/seals.json" --model "fcn --square --first $first"  $common --run_name seals
      $cmd "json --path $prefix/apples.json" --model "fcn --square --first $first"  $common --run_name apples
      $cmd "json --path $prefix/penguins.json" --model "fcn --first $first" $common   --run_name penguins
      $cmd "json --path $prefix/scallops_niwa.json" --model "fcn --first $first" $common   --run_name scallops
      $cmd "json --path $prefix/fisheye.json" --model "fcn --first $first" $common   --run_name fisheye
  done
done

