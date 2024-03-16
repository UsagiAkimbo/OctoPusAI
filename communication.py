import asyncio
from typing import Callable, Dict

class CommunicationBus:
    def __init__(self):
        self.listeners = dict()
        self.event_loop = asyncio.get_event_loop()

    async def send_message(self, message_type: str, data: Dict):
        if message_type in self.listeners:
            for callback in self.listeners[message_type]:
                self.event_loop.create_task(callback(data))

    def register_listener(self, message_type: str, callback: Callable):
        if message_type not in self.listeners:
            self.listeners[message_type] = []
        self.listeners[message_type].append(callback)

    def deregister_listener(self, message_type: str, callback: Callable):
        if message_type in self.listeners:
            self.listeners[message_type].remove(callback)
            if not self.listeners[message_type]:
                del self.listeners[message_type]

communication_bus = CommunicationBus()

async def start_communication_bus():
    await communication_bus.distribute_messages()

