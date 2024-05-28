import re

def extract_metrics(lines):
    metrics = {'HandlerTime.ms': None, 'PredictionTime.ms': None}
    for line in lines:
        handler_time_match = re.search(r'HandlerTime.ms:([\d.]+)', line)
        prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
        if handler_time_match:
            metrics['HandlerTime.ms'] = float(handler_time_match.group(1))
        if prediction_time_match:
            metrics['PredictionTime.ms'] = float(prediction_time_match.group(1))
    return metrics

# Read the content of the model metrics log file
log_file_path = 'logs/model_metrics.log'  # Update this with the path to your log file
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Parse lines in pairs and extract metrics
model_metrics = []
for i in range(0, len(lines), 2):
    metrics = extract_metrics(lines[i:i+2])
    model_metrics.append(metrics)

# Print the extracted model metrics
for metrics in model_metrics:
    handler_time = metrics.get('HandlerTime.ms', 'N/A')
    prediction_time = metrics.get('PredictionTime.ms', 'N/A')
    print(f"Handler Time: {handler_time} ms | Prediction Time: {prediction_time} ms")
