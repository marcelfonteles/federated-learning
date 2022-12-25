#!/bin/bash
#!usr/bin/env bash

file="pids.txt"

while read -r line; do
    kill -9 $line
done <$file