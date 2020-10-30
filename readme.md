# mo445 first task

# to run
docker run -it --rm \
  --env DISPLAY=$DISPLAY --env NUMIMAGES=5 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) \
  adnrv/opencv \
  /bin/bash -c \
  "\
  python3 data_set.py
  "