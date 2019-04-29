
base="/home/oliver/export/"
cmd="python -m main --input"

common="--no_load --train_epochs 80 --epoch_size 1024 --log_dir /local/storage/logs/validate/" 
prefix="/local/storage/export"

$cmd "json --path $prefix/oliver/combined.json" --model "fcn --square --first 2" --image_size 400 $common --run_name aerial_penguins
$cmd "json --path $prefix/scott_base.json" --model "fcn --square --first 2"  --image_size 400 $common --run_name scott_base

$cmd "json --path $prefix/apples.json" --model "fcn --square"  --image_size 1024 $common --run_name apples
$cmd "json --path $prefix/apples_lincoln.json" --model "fcn --square" --image_size 1024 $common --run_name apples_lincoln
$cmd "json --path $prefix/seals.json" --model "fcn --square"  --image_size 1024 $common --run_name seals
$cmd "json --path $prefix/seals_shanelle.json" --model "fcn --square"  --image_size 1024 $common --run_name seals_shanelle
 
$cmd "json --path $prefix/victor.json" --image_size 1024 $common --vertical_flips --transposes --run_name victor
$cmd "json --path $prefix/scallops_niwa.json" $common  --image_size 800 --epoch_size 2048 --run_name scallops

$cmd "json --path $prefix/mum/buoys.json" $common --image_size 800  --run_name buoys
$cmd "json --path $prefix/penguins.json" $common  --run_name penguins
$cmd "json --path $prefix/branches.json" $common --image_size 320 --run_name branches


