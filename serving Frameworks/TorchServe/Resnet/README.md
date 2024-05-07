Sample commands to create a resnet-18 eager mode model archive, register it on TorchServe and run image prediction
Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve


      wget https://download.pytorch.org/models/resnet18-f37072fd.pth
    torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json
    mkdir model_store
    mv resnet-18.mar model_store/
    torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
    curl http://127.0.0.1:8080/predictions/resnet-18 -T ./examples/image_classifier/kitten.jpg
