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
docker build . -t terryzho/openeval:$version
docker tag terryzho/openeval:$version terryzho/openeval:latest
docker push terryzho/openeval:$version
docker push terryzho/openeval:latest

rm -rf dist
python3 -m build
python3 -m twine upload dist/*

# git push
git push
git push --tags
