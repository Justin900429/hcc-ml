import socket
from threading import Lock

from tellosrc.base import ResourceThread


class StateReceiver(ResourceThread):
    def __init__(self, ip="0.0.0.0", port=8890):
        super().__init__()
        self.ip = ip
        self.port = port
        self.state_lock = Lock()
        self.state = dict()
        self.id = None

    def run(self):
        sck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sck.bind((self.ip, self.port))
        sck.settimeout(1.0)
        while not self.stopped():
            try:
                response, _ = sck.recvfrom(1024)
                with self.state_lock:
                    new_result = False
                    for item in response.decode().strip().split(';'):
                        item = item.strip()
                        if len(item) == 0:
                            continue
                        try:
                            key, value = item.split(':')
                            if key == 'mpry':
                                value = tuple(map(float, value.split(',')))
                            else:
                                value = float(value)
                            self.state[key] = value
                            new_result = True
                        except Exception:
                            # Ignore invalid state item. This may be caused by
                            # the UDP packet loss.
                            print("Invalid state item: '%s'" % item)
                    if new_result:
                        if self.id is None:
                            self.id = 0
                        else:
                            self.id += 1
            except socket.timeout:
                # Ignore timeout.
                # This behavior allows the thread to check whether it should
                # stop.
                pass
        sck.close()

    def get_result(self):
        with self.state_lock:
            return self.id, (self.state.copy(),)
