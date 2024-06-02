# argument version

set -eux

while getopts "v:" opt; do
  case $opt in
    v)
      version=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

if [ -z "$version" ]; then
  echo "version is required"
  exit 1
fi

export PYTHONPATH=$PWD pytest tests

docker build
docker build -f Docker/Evaluate.Dockerfile . -t terryzho/bigcodebench-evaluate:$version
docker tag terryzho/bigcodebench-evaluate:$version terryzho/bigcodebench-evaluate:latest
docker push terryzho/bigcodebench-evaluate:$version
docker push terryzho/bigcodebench-evaluate:latest

docker build -f Docker/Generate_Cuda11.Dockerfile . -t terryzho/bigcodebench-generate-cu11:$version
docker tag terryzho/bigcodebench-generate-cu11:$version terryzho/bigcodebench-generate-cu11:latest
docker push terryzho/bigcodebench-generate-cu11:$version
docker push terryzho/bigcodebench-generate-cu11:latest

docker build -f Docker/Generate_Cuda12.Dockerfile . -t terryzho/bigcodebench-generate-cu12:$version
docker tag terryzho/bigcodebench-generate-cu12:$version terryzho/bigcodebench-generate-cu12:latest
docker push terryzho/bigcodebench-generate-cu12:$version
docker push terryzho/bigcodebench-generate-cu12:latest