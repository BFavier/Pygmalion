import os
import socket
import pickle
from select import select
from time import sleep


PREFIX_SIZE: int = 64
CONNECTION_BROKEN_ERROR = RuntimeError("socket connection broken")


def send(connexion: socket.socket, message: bytes):
    """
    Send bytes to the other side of the connexion.
    This functions is non blocking and writes to a buffer.
    """
    global PREFIX_SIZE, CONNECTION_BROKEN_ERROR
    size = int(len(message)).to_bytes(PREFIX_SIZE, byteorder="little", signed=False)
    connexion.sendall(size)
    connexion.sendall(message)


def receive(connexion: socket.socket) -> bytes:
    """
    Receive bytes sent from the other side of the connexion.
    This function is blocking until some data can be read from the buffer.
    """
    global PREFIX_SIZE, CONNECTION_BROKEN_ERROR
    size = b""
    while len(size) < PREFIX_SIZE:
        s = connexion.recv(PREFIX_SIZE)
        if s == b"":
            raise CONNECTION_BROKEN_ERROR
        size += s
    size = int.from_bytes(size, byteorder="little", signed=False)
    message = b""
    while len(message) < size:
        msg = connexion.recv(size - len(message))
        if msg == b"":
            raise CONNECTION_BROKEN_ERROR
        message += msg
    return message


def send_object(connexion: socket.socket, obj: object):
    """
    send the given object at the bottom of the sent objects pile
    """
    byt = pickle.dumps(obj)
    send(connexion, byt)


def receive_object(connexion: socket.socket) -> object:
    """
    return the object at the top of the sent pile, or wait for one to be sent
    """
    byt = receive(connexion)
    return pickle.loads(byt)


class Server:
    """
    A server is centralizing the work of several workers
    """

    def __init__(self, n_workers: int, port: int=3834, address: str="127.0.0.1"):
        """
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((address, port))
        self.socket.settimeout(None)
        self.socket.listen(n_workers)
        self.workers = []
        for _ in range(n_workers):
            conn, addr = self.socket.accept()
            self.workers.append(conn)

    def get_work(self) -> object:
        """
        get the work of the first worker ready
        """
        ready, _, _ = select(self.workers, [], [])
        connexion = ready[0]
        obj = receive_object(connexion)
        send(connexion, b"")  # tell the worker that we receptionned his work and to prepare another work
        return obj


class Client:
    """
    A Client is a process that works asynchronously to send some work to the server
    """
 
    def __init__(self, port: int=3834, address: str="127.0.0.1"):
        """
        """
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(None)
                self.socket.connect((address, port))
            except ConnectionRefusedError as e:
                sleep(1.0)
            else:
                break


    def send_work(self, obj: object):
        """
        send work and wait until server signals it receptionned before exiting
        """
        send_object(self.socket, obj)
        receive(self.socket)


if __name__ == "__main__":
    import IPython
    IPython.embed()