from hpe.modelinvoker import ModelInvoker

serving_engine_choice = "torchserve"

model_invoker = ModelInvoker(serving_engine_choice)
model_invoker.invoke("resnet18", "resnet-18.mar")
