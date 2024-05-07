
# Getting started

Install TorchServe and torch-model-archiveror

1.Install dependencies

### For Conda
Python >=3.8 is required to run Torchserve.

Install torchserve, torch-model-archiver and torch-workflow-archiver


            
    Install torchserve, torch-model-archiver and torch-workflow-archiver

For Conda Note: Conda packages are not supported for Windows. Refer to the documentation here.

conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch

### For Pip

     pip install torchserve torch-model-archiver torch-workflow-archiver



# Serve a model
This section shows a simple example of serving a model with TorchServe.you must have already installed TorchServe and the model archiver.

To run this example, clone the TorchServe repository:

    git clone https://github.com/pytorch/serve.git

Then run the following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path.

# Store a Model
To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model. You can also create model stores to store your archived models.

Create a directory to store your models.

    mkdir model_store
Download a trained model.

    wget https://download.pytorch.org/models/densenet161-8d451a50.pth

Archive the model by using the model archiver. The extra-files param uses a file from the TorchServe repo, so update the path if necessary.

    torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier

For more information about the model archiver, see Torch Model archiver for TorchServe

### Start TorchServe to serve the model

After you archive and store the model, use the torchserve command to serve the model.

    torchserve --start --ncs --model-store model_store --models densenet161.mar


#### produces the output as :
 For the kitten Image

          {
          "tabby": 0.40966343879699707,
          "tiger_cat": 0.346704363822937,
          "Egyptian_cat": 0.13002890348434448,
          "lynx": 0.023919545114040375,
          "bucket": 0.011532172560691833
        }


## Using REST APIs

as an example we’ll download the below cute kitten with

kitten


    curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

And then call the prediction endpoint


    curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg

Which will return the following JSON object
        
        {
          "tabby": 0.40966343879699707,
          "tiger_cat": 0.346704363822937,
          "Egyptian_cat": 0.13002890348434448,
          "lynx": 0.023919545114040375,
          "bucket": 0.011532172560691833
        }

All interactions with the endpoint will be logged in the logs/ directory, so make sure to check it out!


### Stop TorchServe

            torchserve --stop

## Inspect the logs

All the logs you’ve seen as output to stdout related to model registration, management, inference are recorded in the /logs folder.
