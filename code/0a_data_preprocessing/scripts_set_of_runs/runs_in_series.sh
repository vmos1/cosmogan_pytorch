
## Script to run extraction on a set of foldes in series
#for i in $1/*; do prefix=$(basename $i); full_path=$(realpath $i); echo $prefix; echo $full_path; done
for i in $1/*; do prefix=$(basename $i); full_path=$(realpath $i); echo $prefix; echo $full_path; python 1_slice_parallel.py --cores 4 -p $prefix --smoothing --mode full --splice 16 -d $full_path; done

