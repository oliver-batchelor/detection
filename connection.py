
import websockets
from websockets.exceptions import ConnectionClosed

import asyncio
from asyncio.streams import IncompleteReadError

from concurrent.futures import ThreadPoolExecutor
from torch.multiprocessing import Process, Pipe
import time


def connection(conn, url, reconnect_time=0.5):
    async def recv_loop(socket):
        try:
            while True:
                in_str = await socket.recv()
                conn.send(in_str)                   
        except (ConnectionClosed, IncompleteReadError) as e:
            conn.send(None)                   



    async def send_loop(socket):
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            while True:
                msg = await loop.run_in_executor(executor, conn.recv)
                if msg is None: 
                    break
                await socket.send(msg)
        except (ConnectionClosed, IncompleteReadError) as e:
            pass

    async def run():
        async with websockets.connect(url) as socket:
            await asyncio.wait([recv_loop(socket), send_loop(socket)], return_when=asyncio.FIRST_COMPLETED)

    def run_reconnecting():
        while True:
            try:
                asyncio.get_event_loop().run_until_complete(run())
                time.sleep(reconnect_time)

            except (EOFError, ConnectionClosed, ConnectionRefusedError):
                pass

    try:
        run_reconnecting()
    except (KeyboardInterrupt, SystemExit):
        pass



def connect(url):
    (conn1, conn2) = Pipe()
    p = Process(target=connection, args=(conn2, url))
    p.start()

    return p, conn1
