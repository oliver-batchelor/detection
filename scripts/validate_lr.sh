base="/home/oliver/export/"
cmd="python -m main --input"

base="--no_load" 
root="/local/storage"
prefix="$root/export"

for method in cosine step log;
do
  for cycle in 1 2 4;
  do

    len=$((80/cycle))
    epoch=$((1024*cycle))
    lr_step=$((40/cycle))

    common="$base --scale 0.5 --log_dir $root/logs/lr/$method/$epoch/ --lr_step $lr_step --lr_decay $method --epoch_size $epoch --train_epochs $len"

    $cmd "json --path $prefix/oliver/combined.json" --model "fcn --square --first 2" --image_size 400 $common --run_name oliver_penguins
    $cmd "json --path $prefix/apples.json" --model "fcn --square" --image_size 1024 $common --run_name apples
    $cmd "json --path $prefix/scallops_niwa.json" $common  --image_size 800  --run_name scallops
    $cmd "json --path $prefix/branches.json" $common --image_size 320   --run_name branches
  done
done

