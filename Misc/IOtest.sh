#!/bin/bash
# test functionalities of bash scripts

strvar='hello world!'
echo $strvar
cmdvar=$( ls -h )
echo $cmdvar
python ComdArgParse.py -i $1 -o $2 -p $3

# IFS='' (or IFS=) prevents leading/trailing whitespace from being trimmed.
# -r prevents backslash escapes from being interpreted.
# || [[ -n $line ]] prevents the last line from being ignored if it doesn't end with a \n (since read returns a non-zero exit code when it encounters EOF)

while IFS='' read -r line || [[ -n "$line" ]]
do
    echo "Text read from file: $line"
done < params.txt

for file in $( ls *.py )
do
    name=$( echo $file | sed -E 's/([A-Za-z_]+)\.py$/\1/' )
    echo $name
done