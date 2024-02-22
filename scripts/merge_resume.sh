#!/usr/bin/env bash

RS_DIR=./random_search/random_search_${1}

for MAPS_RESUME in $(find ${RS_DIR}/maps -name "*_resume" -type d); do
    echo $MAPS_RESUME
    cp -r ${MAPS_RESUME}/split-* ${MAPS_RESUME//"_resume"/}/
done
