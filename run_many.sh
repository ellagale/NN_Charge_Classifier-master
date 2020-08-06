#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
	current_time=$(date "+%Y.%m.%d-%H.%M.%S")
	python modelv2.py 2>&1 | tee 1K_NN_log.$current_time.txt
done

