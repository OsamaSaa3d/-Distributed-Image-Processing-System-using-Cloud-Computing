import tkinter as tk
from tkinter import filedialog
import socket
import pickle
import cv2

# server details
SERVER_IP_ADDRESS = "192.168.88.132"
SERVER_PORT =  1550

def upload_image():
    filename = filedialog.askopenfilename()
    if filename:
        # Load the image using OpenCV
        image_data = cv2.imread(filename)
        if image_data is None:
            print("Error: Unable to load image")
            return None, None
        else:
            return image_data, filename.split("/")[-1]

def send_image_and_operation(image_data, operation_type):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP_ADDRESS, SERVER_PORT))
        message = {"operation_type": operation_type, "image_data": image_data}
        x = pickle.dumps(message)
        client_socket.sendall(x)
        
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        message_reply = pickle.loads(data)
        image = message_reply["image_data"]
        # Display the processed image
        cv2.imshow("Processed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        client_socket.close()
        return message_reply["response"]
    except Exception as e:
        return str(e)

def process_image():
    image_data, _ = upload_image()
    operation_type = operation_var.get()
    response = send_image_and_operation(image_data, operation_type)
    response_label.config(text=response)

def main():
    root = tk.Tk()
    root.title("Image Processing Client")
    
    upload_button = tk.Button(root, text="Upload Image", command=process_image)
    upload_button.pack()
    
    global response_label
    response_label = tk.Label(root, text="")
    response_label.pack()
    
    global operation_var
    operation_var = tk.StringVar()
    operation_var.set("Invert")  # Default operation
    operation_menu = tk.OptionMenu(root, operation_var, "Invert", "Grayscale", "Blur")  # Add more operations as needed
    operation_menu.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()
