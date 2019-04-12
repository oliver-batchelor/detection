
base="/home/oliver/export/"
cmd="python -m main --input"

base="--run_name validate --no_load  --epoch_size 1024 --lr_decay cosine --lr_min 0.01" 
prefix="/home/oliver/storage/export"

for method in cosine step log;
do
  for cycle in 1 2 4;
  do

    len=$((80/cycle))
    epoch=$((1024*cycle))
    lr_step=$((40/cycle))

    echo "--log_dir /home/oliver/logs/lr/$method$epoch/"

    common="$base --log_dir /home/oliver/logs/lr/$method/$epoch/ --lr_step $lr_step --lr_decay $method --epoch_size $epoch --train_epochs $len"

    

    #$cmd "json --path $prefix/oliver/combined.json" --model "fcn --square --first 2" --image_size 400 $common --run_name oliver_penguins
    #$cmd "json --path $prefix/apples.json" --model "fcn --square" --image_size 1024 $common --run_name apples
    echo $cmd "json --path $prefix/scallops_niwa.json" $common  --image_size 800  --run_name scallops
    #$cmd "json --path $prefix/branches.json" $common --image_size 320   --run_name branches
  done
done

