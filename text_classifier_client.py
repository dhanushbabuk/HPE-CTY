import grpc
import text_classifier_pb2
import text_classifier_pb2_grpc
import sys
import os

class TextClassifierClient:
    # ... (rest of the TextClassifierClient class remains the same)
def run():
    if len(sys.argv) < 2:  # Check if text is provided as an argument
        print("Error: Text input is required as a command-line argument.")
        return
    
    # Establish connection to the server
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = text_classifier_pb2_grpc.TextClassifierStub(channel)

        while True:
            text = sys.argv[1]
            if text.lower() == 'q':
                break

            request = text_classifier_pb2.TextRequest(text=text)
            response = stub.ClassifyText(request)

            # Print the classification result
            print("Category:", response.category)
            print("Confidence:", response.confidence)

            # Exit after processing the input once
            break

if __name__ == '__main__':
    run()

