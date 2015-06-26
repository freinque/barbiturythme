#! /bin/bash
csound --logfile=$1.csdlog -d -o $1.wav $1.orc $1.sco
#####csound --format=24bit d -o $1.wav $1.orc $1.sco
echo wav generated
#sleep 0.5s
#aplay /home/freinque/Dropbox/ugur_stuff_drop/barbiturythme/bin/beat_lear.wav
