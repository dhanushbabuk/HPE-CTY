import os
from datetime import datetime

class displayMetrics:
    def extract_resnet_metrics(lines):
        metrics = {'HandlerTime.ms': None, 'PredictionTime.ms': None, 'Timestamp': None}
        for line in lines:
            handler_time_match = re.search(r'HandlerTime.ms:([\d.]+)', line)
            prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
            timestamp_match = re.search(r'timestamp:(\d+)', line)
            if handler_time_match:
                metrics['HandlerTime.ms'] = float(handler_time_match.group(1))
            if prediction_time_match:
                metrics['PredictionTime.ms'] = float(prediction_time_match.group(1))
            if timestamp_match:
                timestamp = int(timestamp_match.group(1))
                metrics['Timestamp'] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return metrics

    def extract_bert_metrics(line):
        metrics = {'PredictionTime.ms': None, 'Timestamp': None}
        prediction_time_match = re.search(r'PredictionTime.ms:([\d.]+)', line)
        timestamp_match = re.search(r'timestamp:(\d+)', line)
        if prediction_time_match:
            metrics['PredictionTime.ms'] = float(prediction_time_match.group(1))
        if timestamp_match:
            timestamp = int(timestamp_match.group(1))
            metrics['Timestamp'] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return metrics

    def display_metrics(cls):
        log_file_path = 'logs/model_metrics.log'
        if not os.path.exists(log_file_path):
            st.error(f"Log file not found: {log_file_path}")
            return

        with open(log_file_path, 'r') as file:
            lines = file.readlines()

        resnet_metrics = []
        count_resnet = 0
        for i in range(len(lines) - 1, -1, -1):
            if "resnet18" in lines[i]:
                metrics = cls.extract_resnet_metrics(lines[i-1:i+1])
                resnet_metrics.append(metrics)
                count_resnet += 1
                if count_resnet == 5:
                    break

        bert_metrics = []
        count_bert = 0
        for i in range(len(lines) - 1, -1, -1):
            if "bert" in lines[i]:
                metrics = cls.extract_bert_metrics(lines[i])
                bert_metrics.append(metrics)
                count_bert += 1
                if count_bert == 5:
                    break

        st.write("Metrics for resnet18:")
        for metrics in reversed(resnet_metrics):
            handler_time = metrics.get('HandlerTime.ms', 'N/A')
            prediction_time = metrics.get('PredictionTime.ms', 'N/A')
            timestamp = metrics.get('Timestamp', 'N/A')
            st.write(f"Timestamp: {timestamp} | Handler Time: {handler_time} ms | Prediction Time: {prediction_time} ms")

        st.write("Prediction Times for bert:")
        for metrics in reversed(bert_metrics):
            prediction_time = metrics.get('PredictionTime.ms', 'N/A')
            timestamp = metrics.get('Timestamp', 'N/A')
            if prediction_time != 'N/A':
                st.write(f"Timestamp: {timestamp} | Prediction Time: {prediction_time} ms")
