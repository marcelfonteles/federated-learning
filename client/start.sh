#!/bin/bash
# Calculate the sum of two integers with pre initialize values
# in a shell script
port=3000
for i in {1..3}; do
  flask --app app.py run -h 0.0.0.0 -p $(( $port + $i)) & >> pids.txt
done
