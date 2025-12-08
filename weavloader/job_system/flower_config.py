'''
Flower Configuration for Weavloader
see https://flower.readthedocs.io/en/latest/config.html for more details
'''
import os

#redis
broker = 'redis://localhost:6379/0'

# Basic authentication
basic_auth = 'admin:admin'

# Flower settings
port = 5555
address = '0.0.0.0'

# Enable events
enable_events = True

# Logging configuration
logging = os.getenv('LOG_LEVEL', 'INFO')

# Task monitoring
max_workers = 5000
max_tasks = 10000

# Enable task events
purge_offline_workers = 0  # never remove offline workers

# Natural time display
natural_time = True

# Task columns to display
tasks_columns = 'name,uuid,state,args,kwargs,result,received,started,runtime,worker,retries'

# Enable debug mode (set to False in production)
debug = True if os.getenv('LOG_LEVEL', 'INFO') == 'DEBUG' else False

# Enable xheaders for proxy support
xheaders = True
# persist the state of the flower
# persistent = True
# db = '/tmp/flower.db'
# state_save_interval = 5000  # saving the Flower state every 5 sec in milliseconds
