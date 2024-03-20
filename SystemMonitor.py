import logging
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        # Setup logging
        logging.basicConfig(filename="system_monitor.log", level=logging.INFO)
        self.start_time = datetime.now()

    def log_activity(self, activity):
        """
        Logs system activities for monitoring.
        """
        logging.info(f"{datetime.now()} - {activity}")

    def log_error(self, error):
        """
        Logs system errors for debugging and monitoring.
        """
        logging.error(f"{datetime.now()} - ERROR: {error}")

    def report_system_health(self):
        """
        Reports on system health based on performance metrics.
        """
        # Placeholder for calculating and logging system health metrics
        uptime = datetime.now() - self.start_time
        logging.info(f"System Uptime: {uptime}")
        # Add more detailed health and performance metrics as needed

system_monitor = SystemMonitor()
