#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./recode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

if [ ! -d "${outdir}" ]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

for c in $(ls ${indir})
do
	for inname in $(ls ${indir}/${c}/*avi)
	do
		class_path="$(dirname "$inname")"
		class_name="${class_path##*/}"

		outname="${outdir}/${class_name}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
    # you might want to try libx264 (need to compile FFMPEG with libx264)
		ffmpeg -hide_banner -loglevel panic -i ${inname} -c:v mpeg4 -q:v 1 ${outname}

	done
done
