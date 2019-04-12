
base="/home/oliver/export/"
cmd="python -m main --input"

common="--no_load --train_epochs 80 --eval_split --epoch_size 1024 --image_size 400   --log_dir /home/oliver/logs/penguins"
prefix="/home/oliver/storage/export"

$cmd "json --path $prefix/oliver/penguins_royds.json"  $common  --run_name oliver_royds --model 'fcn --square --first 2'
$cmd "json --path $prefix/oliver/penguins_cotter.json" $common  --run_name oliver_cotter --model 'fcn --square --first 2'
$cmd "json --path $prefix/oliver/penguins_hallett.json" $common --run_name oliver_hallett --model 'fcn --square --first 2'

$cmd "json --path $prefix/dad/penguins_royds.json"  $common  --run_name dad_royds --model 'fcn --square --first 2'
$cmd "json --path $prefix/dad/penguins_cotter.json" $common  --run_name dad_cotter --model 'fcn --square --first 2'
$cmd "json --path $prefix/dad/penguins_hallett.json" $common --run_name dad_hallett  --model 'fcn --square --first 2'



