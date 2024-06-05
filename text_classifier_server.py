import grpc
from concurrent import futures
import text_classifier_pb2
import text_classifier_pb2_grpc
from transformers import pipeline
import time

# --- gRPC Server ---

class TextClassifierServicer(text_classifier_pb2_grpc.TextClassifierServicer):
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
    def ClassifyText(self, request, context):
        start_time = time.time()  # Record start time
        text = request.text
        try:
            result = self.classifier(text)[0]
            predicted_category = result["label"]
            confidence = result["score"]
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An error occurred while processing your request.")
            return text_classifier_pb2.ClassificationResponse()
        end_time = time.time()   # Record end time
        handler_time = end_time - start_time
        return text_classifier_pb2.ClassificationResponse(
            category=predicted_category,
            confidence=confidence,
            handler_time=handler_time
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    text_classifier_pb2_grpc.add_TextClassifierServicer_to_server(TextClassifierServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

