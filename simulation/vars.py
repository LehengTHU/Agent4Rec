import threading

global global_k_tokens
global global_start_time
global global_steps
global global_last_tokens_record
global global_interval
global global_finished_users
global global_finished_pages
global global_error_cast
global lock

global_k_tokens = 0
global_start_time = 0
global_steps = 0
global_last_tokens_record = 0
global_interval = 10
global_finished_users = 0
global_finished_pages = 0
global_error_cast = 0

lock = threading.Lock() # global lock for threads