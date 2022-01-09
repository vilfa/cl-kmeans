#!/usr/bin/env zsh

echo Cleaning and making project...
make clean && make

if [ ! -d images ]; then
  echo Missing images directory. Exiting.
  exit
fi

if [ ! -d out ]; then
  echo Missing output directory. Creating.
  mkdir out
fi

images=("480p.png" "600p.png" "720p.png" "900p.png" "1080p.png" "2k.png" "4k.png" "5k.png" "6k.png")
x_args=("-t1" "-t2" "-t4" "-t8" "-g")
k_count=(8 16 16 16 32 32 64 64 128 256)
n_count=(8 8 16 32 32 64 64 128 128 128)
n_test=0

for img in "${images[@]}"; do
  printf "#################\n"
  printf "#Image: %s#\n" $img
  printf "#################\n"
  for x in "${x_args[@]}"; do
    for ((i = 1; i <= $#k_count; i++)); do
      printf "\n---Run %d---\n" $n_test
      printf "Image: %s; Clusters: %d; Iter: %d; Xcmd=%s\n" $img ${k_count[i]} ${n_count[i]} $x
      echo "./build/compress $x -k${k_count[i]} -n${n_count[i]} -iimages/$img -oout/${k_count[i]}k_${n_count[i]}n_$img"
      time ./build/compress $x -x -k${k_count[i]} -n${n_count[i]} -iimages/$img -oout/${k_count[i]}k_${n_count[i]}n_$img
      ((n_test = n_test + 1))
    done
  done
  printf "#################\n"
  printf "#################\n"
  printf "#################\n"
done

echo Testing done. Run "$n_test" tests.
