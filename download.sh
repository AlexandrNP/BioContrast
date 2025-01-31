#!/bin/bash

hsa=$(curl -s https://rest.kegg.jp/list/pathway/hsa)
while IFS=$'\t' read -ra line; do
  #for i in "${line[@]}"; do
    echo "${line[0]}"
    curl -s "https://rest.kegg.jp/get/${line[0]}/kgml" -o "KGML/${line[0]}.xml"
    #echo $cmd
    #curl "https://rest.kegg.jp/get/${line[0]}/kgml -o KGML/${line[0]}.xml"
  #done
done <<< "$hsa"
#echo $hsa