# reference: https://grpc.io/docs/protoc-installation/
protoc --python_out=. ./payload.proto
mv payload_pb2.py payload.py