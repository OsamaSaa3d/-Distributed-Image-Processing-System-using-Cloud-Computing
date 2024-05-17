import socket
import threading
import cv2
import pickle

class ClientThread(threading.Thread):
    def __init__(self, client_socket, address):
        threading.Thread.__init__(self)
        self.client_socket = client_socket
        self.address = address
        print("[+] New thread started for ", address)

    def run(self):
        try:
            print("Connection from : ", self.address)
            data = b""
            while True:
                self.client_socket.settimeout(2)  # Set timeout to 2 seconds
                try:
                    packet = self.client_socket.recv(4096)
                except socket.timeout:
                    print("Timeout occurred. No further data received.")
                    break
                data += packet
            message = pickle.loads(data)
            image_data = message["image_data"]
            operation_type = message["operation_type"]
            
            if operation_type == "Invert":
                processed_image = cv2.bitwise_not(image_data)
            elif operation_type == "Grayscale":
                processed_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            elif operation_type == "Blur":
                processed_image = cv2.GaussianBlur(image_data, (5, 5), 0)
            else:
                processed_image = image_data  # If operation_type is unknown, return the original image
            
            message_reply = {"operation_type": operation_type, "image_data": processed_image}
            x = pickle.dumps(message_reply)

            self.client_socket.sendall(x)
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            self.client_socket.close()
            print("Client at ", self.address, " disconnected...")

def main():
    SERVER_IP_ADDRESS = "192.168.88.132"  # Change this to your server's IP address
    SERVER_PORT = 1550

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_IP_ADDRESS, SERVER_PORT))
    server_socket.listen(5)
    print("Server is listening...")
    try:
        while True:
            client_socket, address = server_socket.accept()
            print(f"Connection from {address} established.")
            client_thread = ClientThread(client_socket, address)
            client_thread.start()
    except KeyboardInterrupt as e:
        server_socket.close()

if __name__ == "__main__":
    main()
