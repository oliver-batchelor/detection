python -m main --input "json --path /home/oliver/export/penguins_royd.json" --model "fcn --square --first 2" --image_size 300 --run_name validate_royd  --no_load
python -m main --input "json --path /home/oliver/export/penguins_hallett.json" --model "fcn --square --first 2" --image_size 300 --run_name validate_hallett  --no_load
python -m main --input "json --path /home/oliver/export/penguins_cotter.json" --model "fcn --square --first 2" --image_size 300 --run_name validate_cotter  --no_load
python -m main --input "json --path /home/oliver/export/penguins_combined.json" --model "fcn --square --first 2" --image_size 300 --run_name validate_combined --auto_pause 128  --tests "test_hallett,test_royd,test_cotter" --no_load

