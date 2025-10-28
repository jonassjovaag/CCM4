import threading
from typing import Dict, Callable

class ThreadController:
    def __init__(self):
        self.threads = {}
        self.running = threading.Event()
        
    def start_thread(self, name: str, target: Callable, daemon: bool = True):
        thread = threading.Thread(target=target, name=name, daemon=daemon)
        self.threads[name] = thread
        thread.start()
        
    def start_all_threads(self, thread_map: Dict[str, Callable]):
        self.running.set()
        for name, target in thread_map.items():
            self.start_thread(name, target)
            print(f"Started thread: {name}")
        active_threads = self.get_active_threads()
        print(f"Running threads: {', '.join(active_threads.keys())}")
        
    def stop_all(self):
        self.running.clear()
        for thread in self.threads.values():
            thread.join(timeout=1.0)
            
    def is_thread_alive(self, name: str) -> bool:
        return name in self.threads and self.threads[name].is_alive()
        
    def get_active_threads(self) -> Dict[str, threading.Thread]:
        return {name: thread for name, thread in self.threads.items() 
                if thread.is_alive()}
                
    def restart_thread(self, name: str, target: Callable):
        if name in self.threads:
            self.threads[name].join(timeout=0.5)
        self.start_thread(name, target)
