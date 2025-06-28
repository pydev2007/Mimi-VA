
# import asyncio
# from websockets.asyncio.server import serve


# async def echo(websocket):
#     async for message in websocket:
#         print(f"Received message from client: {message}")
#         await websocket.send(f"Hello, Client! You said: {message}")

# async def main():
#     async with serve(echo, "localhost", 8765) as server:

#         await server.serve_forever()


# if __name__ == "__main__":
#     asyncio.run(main())

# Client (client.py)
import asyncio
import json
import websockets
import base64

async def handle_connection(websocket):
    try:
        received_chunks = {}
        async for message in websocket:


            data = json.loads(message)

            if data.get("type") == "file_chunk":
                filename = data["filename"]
                chunk_number = data["chunk_number"]
                is_last = data["is_last"]
                chunk_data = base64.b64decode(data["data"]) if data["data"] else b''

                if filename not in received_chunks:
                    received_chunks[filename] = {}

                received_chunks[filename][chunk_number] = chunk_data

                if is_last:
                    print(f"Received final chunk of {filename}. Assembling...")
                    chunks = [received_chunks[filename][i] for i in sorted(received_chunks[filename])]
                    with open(f"{filename}", 'wb') as out_file:
                        for chunk in chunks:
                            out_file.write(chunk)
                    print(f"File {filename} saved as received_{filename} with a message of: {json.loads(message)["message"]}")
                    del received_chunks[filename]

    except websockets.exceptions.ConnectionClosedError:
        print("Client disconnected unexpectedly.")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
