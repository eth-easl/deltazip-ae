cd serving && docker build -f Dockerfile -t ghcr.io/xiaozheyao/deltaserve:test .
cd ../compression && docker build -f meta/docker/Dockerfile.deltaserve -t ghcr.io/xiaozheyao/deltazip:test .