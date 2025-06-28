
# import asyncio
# from websockets.asyncio.client import connect


# async def hello():
#     async with connect("ws://localhost:8765") as websocket:
#         await websocket.send("Hello world!")
#         message = await websocket.recv()
#         print(message)


# if __name__ == "__main__":
#     asyncio.run(hello())

import asyncio
import websockets
import json
import base64

CHUNK_SIZE = 4096  # You can adjust this

async def send_file(uri, filename, message):
    async with websockets.connect(uri) as websocket:
        with open(filename, 'rb') as file:
            chunk_number = 0
            while True:
                chunk = file.read(CHUNK_SIZE)
                if not chunk:
                    break

                encoded_chunk = base64.b64encode(chunk).decode('utf-8')

                data = {
                    "type": "file_chunk",
                    "filename": filename,
                    "message": message,
                    "chunk_number": chunk_number,
                    "data": encoded_chunk,
                    "is_last": False
                }

                await websocket.send(json.dumps(data))
                chunk_number += 1

            # Send a final message indicating the file transfer is complete
            await websocket.send(json.dumps({
                "type": "file_chunk",
                "filename": filename,
                "message": message,
                "chunk_number": chunk_number,
                "data": "",
                "is_last": True
            }))

        print(f"File {filename} sent successfully in {chunk_number} chunks.")


if __name__ == "__main__":
    filename = "Only_Jesus.flac" # Replace with your file
    asyncio.run(send_file("ws://localhost:8765", filename, "test"))
