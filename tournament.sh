#!/bin/bash

# This script is used to run the tournament
# It will run the code of individual players on different maps in maps/tournament folder
# and generate the results in results/tournament folder

# Check if the player number is between 1 and 10
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <player_number>"
    exit 1
fi

if [ "$1" -lt 1 ] || [ "$1" -gt 10 ]; then
    echo "Player number should be between 1 and 10"
    exit 1
fi

# Check if the tournament folder exists
if [ ! -d "requests/tournament" ]; then
    echo "requests/tournament folder does not exist"
    exit 1
fi

# Check if the results folder exists for this player
if [ ! -d "results/tournament" ]; then
    mkdir results/tournament
fi

# Create a map for parameters to use with different maps
declare -A map
map[1]="requests/tournament/g1.json"
map[2]="requests/tournament/g2.json"
map[3]="requests/tournament/g3.json"
map[4]="requests/tournament/g4.json"
map[5]="requests/tournament/g5.json"
map[6]="requests/tournament/g6.json"
map[7]="requests/tournament/g7.json"
map[8]="requests/tournament/g8.json"
map[9]="requests/tournament/g9.json"
map[10]="requests/tournament/g10.json"
map[11]="requests/tournament/g11.json"
map[12]="requests/tournament/g12.json"
map[13]="requests/tournament/g13.json"
map[14]="requests/tournament/g14.json"
map[15]="requests/tournament/g15.json"
map[16]="requests/tournament/g16.json"

declare -A tolerance
tolerance[1]=25
tolerance[2]=12
tolerance[3]=5
tolerance[4]=1

# Run the player code on each map and save the results in results/tournament folder
for i in {1..16}
do
    for t in {1..4}
    do
      echo "Running player $1 on requests $i with tolerance $t"
      python3.10 main.py -p $1 -rq ${map[$i]} -d ${tolerance[$t]} -ng --disable_logging > results/tournament/p${1}/p${1}_g${i}_t${t}.json
    done
done