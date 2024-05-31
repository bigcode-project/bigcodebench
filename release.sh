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

git tag $version

# docker build
docker build . -t terryzho/bigcodebench:$version
docker tag terryzho/bigcodebench:$version terryzho/bigcodebench:latest
docker push terryzho/bigcodebench:$version
docker push terryzho/bigcodebench:latest

rm -rf dist
python3 -m build
python3 -m twine upload dist/*

git push
git push --tags