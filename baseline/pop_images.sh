#!/bin/bash
# Written by Aleksei Wan on 22.03.2020
for e in $(ls); do
        cd $e
        for f in $(ls); do
                mv $f ../$f
        done
        cd ..
done

