import socket

while True:
    # Send user input to Webots controller
    task_in = input("Enter task: ")
    print(task_in)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 5000))
        s.sendall(task_in.encode('utf-8'))