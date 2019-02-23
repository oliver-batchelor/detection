for i in ~/storage/indexes/*.db; do filename=$(basename "$i" .db); ./server $i --export ~/storage/export/$filename.json ; done
