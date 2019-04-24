base="/home/oliver/export/"
cmd="python -m main --input"

root="/local/storage"
prefix="$root/export"

train_on() {
  run=$1
  method=$2
  cycle=$3
  incremental=$4

  len=$((80/cycle))
  epoch=$((1024*cycle))
  lr_step=$((40/cycle))

  common="--no_load --scale 0.5 --log_dir $root/logs/lr/$run/$incremental/$method/$epoch/ --seed $run --lr_step $lr_step --lr_decay $method --epoch_size $epoch --train_epochs $len"
  if [ $incremental = "incremental" ]; then common="$common --incremental"; fi

  $cmd "json --path $prefix/apples.json" --model "fcn --square" --image_size 1024 $common --run_name apples
  $cmd "json --path $prefix/scallops_niwa.json" $common  --image_size 800  --run_name scallops
  $cmd "json --path $prefix/branches.json" $common --image_size 320   --run_name branches

}


for run in {1..8}
do
  for incremental in incremental full;
    do
      for method in cosine log;
      do
        for cycle in 1 4;
        do
          train_on $run $method $cycle $incremental
        done
      done
    train_on $run step 1 $incremental
  done
done
