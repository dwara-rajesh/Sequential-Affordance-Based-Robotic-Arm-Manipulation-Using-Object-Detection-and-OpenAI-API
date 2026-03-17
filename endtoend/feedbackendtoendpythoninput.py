import socket, base64, re, json, time
from openai import OpenAI
import numpy as np

message = []

def GPTCall(msg):
    global message
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=msg
    )
    reply = response.choices[0].message.content
    message.append({"role": "assistant", "content": reply})

    data = reply.strip("```json").strip('```')
    # print("Before RegEx")
    # print(data)
    data = re.sub(r'//.*?\n', '', data) #RegEX Implementation
    # print("After RegEx")
    # print(data)
    jsondata = json.loads(data)
    affordancedata = jsondata['affordance_map']
    taskdata = jsondata['task_sequence']

    return affordancedata, taskdata

#Encode image to base64 fpr GPT Input
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == '__main__':
    imagefilepath = r"YoloOutput.png"
    depthjsonpath = r"depth_list.json"

    # Send user input to Webots controller
    task_in = input("Enter task: ")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 5000))
        s.sendall(task_in.encode('utf-8'))

    rdata = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as r:
        r.bind(("localhost", 5001))  # Use a different port for listening
        r.listen()
        while True:
            try:
                r.settimeout(0.1)
                conn, addr = r.accept()
                received = conn.recv(4096).decode('utf-8')

                if received:
                    start = received.find("[")
                    end = received.find("]")
                    if start != -1 and end != -1:
                        rdata = list(map(float, received[start+1:end].split(',')))
                    break
            except socket.timeout:
                continue

    camera_intrinsics = [{'width': rdata[0]}, {'height': rdata[1]}, {'cx': rdata[2]}, {'cy': rdata[3]},
                         {'hfov': rdata[4]}, {'vfov': rdata[5]}, {'fx': rdata[6]}, {'fy': rdata[7]}]
    camera_position = [rdata[8],rdata[9],rdata[10]]

    print(f"Camera Position Relative to the World: {camera_position}")
    with open(depthjsonpath, "r") as json_file:
        depthimarr = json.load(json_file)

    depthimarr = np.array(depthimarr)

    base64_image = encode_image(imagefilepath)

    message = [
        {"role": "system", "content": "You are an assistant that generates affordance maps to aid robot learning and generates task sequences for robots. You are an expert and obtaining object's world positions based on image, depth data and camera intrinsics and camera position"},
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"Based on the image, task in hand: {task_in} and the depth data: {depthimarr}. Generate an affordance map of each object in the scene called affordance_map and output it as a JSON format containing object_name and affordances. The output should be a JSON format only!!. Generate the task sequence too in JSON format, called task_sequence, any form of transformation would be referred to as move (MOVE actions should have a target position in the form of [x,y,z] based on the depth data and image, and the intrinsic parameters of the camera {camera_intrinsics} and camera position {camera_position}) and any action related to grippers will be pick-up and place.No comments please",
            },
            {
                "type": "image_url",
                "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
                },
            },
            ],
        }
    ]

    affordance_data, task_sequence_data = GPTCall(message)
    print("Affordance O/P by GPT: ", affordance_data)
    print("TaskSequence O/P by GPT: ", task_sequence_data)
    feedback = input("Type 'run' if you are happy with output and want to continue to simulation or provide feedback on output: ")
    while feedback.lower() != "run":
        message.append({"role": "user", "content": feedback})
        affordance_data, task_sequence_data = GPTCall(message)
        print("Affordance O/P by GPT after feedback: ", affordance_data)
        print("TaskSequence O/P by GPT after feedback: ", task_sequence_data)
        feedback = input("Type 'run' if you are happy with output and want to continue to simulation or provide feedback on output: ")

    for task in task_sequence_data:
        if task['action'].lower() == "move":
            data = f"move to {task['target_position']}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 5000))
                s.sendall(data.encode('utf-8'))

        if task['action'] == "pick-up":
            data = "gripper on"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 5000))
                s.sendall(data.encode('utf-8'))

        if task['action'] == "place":
            data = "gripper off"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 5000))
                s.sendall(data.encode('utf-8'))
        time.sleep(5)

    data = "All sub tasks complete"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('localhost', 5000))
                s.sendall(data.encode('utf-8'))