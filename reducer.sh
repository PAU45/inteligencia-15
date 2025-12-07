#!/bin/bash
current_word=""
current_count=0

while IFS=$'\t' read -r word count; do
  if [ "$current_word" = "$word" ]; then
    current_count=$((current_count + count))
  else
    if [ -n "$current_word" ]; then
      echo "$current_word	$current_count"
    fi
    current_word="$word"
    current_count="$count"
  fi
done

if [ -n "$current_word" ]; then
  echo "$current_word	$current_count"
fi
