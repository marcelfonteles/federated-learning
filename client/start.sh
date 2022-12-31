#!/bin/bash
rm pids.txt
port=3000
for i in {1..10}; do
  flask --app app.py run -h 0.0.0.0 -p $(( $port + $i)) & echo $! >> pids.txt
  sleep 5
done
