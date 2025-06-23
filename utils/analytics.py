import os
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import json
import traceback
import re

class AnalyticsEngine:
    def __init__(self):
        self.log_file = self._get_log_file_path()
        self.ensure_log_directory_exists()
        self.verdict_types = ["VERIFIED", "PARTIALLY_VERIFIED", "FAKE", "SUSPICIOUS", "UNVERIFIED"]

    def _get_log_file_path(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(base_dir, 'log', 'predictions.log'),
            os.path.join(os.getcwd(), 'log', 'predictions.log'),
            os.path.join(os.path.expanduser('~'), 'fake_news_logs', 'predictions.log')
        ]
        for path in possible_paths:
            if os.path.exists(os.path.dirname(path)):
                return path
        return possible_paths[0]

    def ensure_log_directory_exists(self):
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            if not os.path.exists(self.log_file):
                open(self.log_file, 'a').close()
        except Exception as e:
            print(f"Could not initialize logging directory: {str(e)}")
            traceback.print_exc()

    def parse_logs(self):
        stats = {
            "total": 0,
            "daily": {},
            "hourly": {},
            "verdicts": {verdict: 0 for verdict in self.verdict_types},
            "accuracy": None,
            "error_rate": None
        }

        if not os.path.exists(self.log_file):
            return stats

        daily_counter = Counter()
        hourly_counter = Counter()

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        parts = line.strip().split("|")
                        if len(parts) < 3:
                            continue
                        timestamp = parts[0].strip()
                        verdict = parts[1].strip().upper()
                        text = parts[2].strip()

                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        date_str = dt.strftime("%Y-%m-%d")
                        hour_str = dt.strftime("%H:00")

                        stats["total"] += 1
                        daily_counter[date_str] += 1
                        hourly_counter[hour_str] += 1

                        if verdict in stats["verdicts"]:
                            stats["verdicts"][verdict] += 1
                        else:
                            stats["verdicts"]["UNVERIFIED"] += 1  # fallback

                    except Exception as e:
                        print(f"Error parsing log line: {line} - {str(e)}")

            stats["daily"] = dict(sorted(daily_counter.items()))
            stats["hourly"] = dict(sorted(hourly_counter.items()))

            verified = stats["verdicts"]["VERIFIED"]
            total = stats["total"]
            stats["accuracy"] = round(verified / total, 2) if total else 0
            stats["error_rate"] = round(1 - stats["accuracy"], 2)

        except Exception as e:
            print(f"Failed to parse logs: {str(e)}")
            traceback.print_exc()

        return stats

    def generate_report(self, output_format='json'):
        data = self.parse_logs()
        if output_format == 'json':
            return json.dumps(data, indent=2)
        elif output_format == 'text':
            lines = [
                "=== Fake News Detection Report ===",
                f"Total Predictions: {data['total']}",
                f"Accuracy: {data['accuracy'] * 100:.2f}%" if data['accuracy'] is not None else "Accuracy: N/A",
                f"Error Rate: {data['error_rate'] * 100:.2f}%" if data['error_rate'] is not None else "Error Rate: N/A",
                "\n-- Verdict Distribution --"
            ]
            for verdict, count in data["verdicts"].items():
                percentage = (count / data['total'] * 100) if data['total'] else 0
                lines.append(f"{verdict}: {count} ({percentage:.1f}%)")
            lines.append("\n-- Daily Activity --")
            for day, count in data["daily"].items():
                lines.append(f"{day}: {count} predictions")
            return "\n".join(lines)
        else:
            return "Unsupported report format"

    def visualize_data(self, save_path=None):
        data = self.parse_logs()
        if data["total"] == 0:
            print("No data available to visualize.")
            return

        plt.figure(figsize=(14, 6))

        # Verdict Pie Chart
        plt.subplot(1, 2, 1)
        values = [v for v in data['verdicts'].values()]
        labels = [k.replace("_", " ").title() for k in data['verdicts'].keys()]
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Verdict Distribution")

        # Daily Activity Bar Chart
        plt.subplot(1, 2, 2)
        dates = list(data['daily'].keys())
        counts = list(data['daily'].values())
        plt.bar(dates, counts, color='skyblue')
        plt.xticks(rotation=45)
        plt.title("Daily Prediction Activity")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Singleton Export
analytics = AnalyticsEngine()
parse_logs = analytics.parse_logs
