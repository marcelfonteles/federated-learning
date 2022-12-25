#!/bin/bash
"" > pids.txt
port=3000
for i in {1..20}; do
  flask --app app.py run -h 0.0.0.0 -p $(( $port + $i)) & echo $! >> pids.txt
done
