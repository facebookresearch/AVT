#!/usr/bin/env bash

mkdir rars && mkdir videos
unrar x hmdb51_org.rar rars/
for rar in $(ls rars); do unrar x "rars/${rar}" videos/; done;
