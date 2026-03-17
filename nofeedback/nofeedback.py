from ikpy.chain import Chain
import numpy as np
from controller import Supervisor,Camera,RangeFinder,VacuumGripper
import warnings,cv2,json,base64,math,socket,re,time
from openai import OpenAI

# warnings.filterwarnings("ignore", category=UserWarning, module="ikpy.chain")

#Initialize Robot Chain
ur5e_chain = Chain.from_urdf_file(r"robot_model.urdf", active_links_mask=[False, True, True, True, True, True, True, False])
# print(ur5e_chain)

#Initialize Robotic Arm
robot = Supervisor()

timestep = int(robot.getBasicTimeStep())

#Get joint names & sensor names
devices = robot.getNumberOfDevices()
joints = []
sensornames = []
for i in range(devices):
    device = robot.getDeviceByIndex(i)
    if "joint" in device.getName() and "sensor" not in device.getName():
        joints.append(device.getName())
    if "sensor" in device.getName():
        sensornames.append(device.getName())

#Get joint motors
motors = []
for joint in joints:
    motors.append(robot.getDevice(joint))
for motor in motors:
    motor.setPosition(float(0))

sensors = []
for sensorname in sensornames:
    sensor = robot.getDevice(sensorname)
    sensor.enable(timestep)
    sensors.append(sensor)


#Set up Gripper
gripper = VacuumGripper("vacuum gripper")

#Set up Depth Camera
camera = Camera("Astra rgb")
camera.enable(timestep)
depthimager = RangeFinder("Astra depth")

#Camera Supervisor
cam = robot.getFromDef("CAMERA")

#Get intrinsic camera values
width = camera.getWidth()
height = camera.getHeight()
cx = width / 2
cy = height / 2
hfov = camera.getFov()
vfov = 2 * math.atan((height / width) * math.tan(hfov / 2))
fx = width / (2 * math.tan(hfov/2))
fy = height / (2 * math.tan(vfov/2))

#Set up YOLOv4 for OBJ Detection
net = cv2.dnn.readNet(r"yolov4.weights",r'yolov4.cfg')
with open(r"coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#Image save path after Object Detection
filepath = r"YoloOutput.png"

#Detect Object
def obj_detect(image, depthimage):
    height, width = image.shape[:2]

    # Set up inference
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputlayers = net.getUnconnectedOutLayersNames()
    detections = net.forward(outputlayers)

    positions = []
    boxes = []
    confidences = []
    class_ids = []

    # Collect all bounding box predictions
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{label}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Get depth value of the detected object
            center_x = x + w // 2
            center_y = y + h // 2
            depth_value = depthimage[int(center_y), int(center_x)]

            # 3D position of the object relative to the camera
            x_camera = (center_x - cx) * depth_value / fx
            y_camera = (center_y - cy) * depth_value / fy
            z_camera = depth_value
            position = [x_camera, z_camera, y_camera]
            objname = classes[class_ids[i]]
            currentdict = {"obj": objname, "position": position}
            positions.append(currentdict)

    cv2.imwrite(filepath, image)
    return positions


#Encode image to base64 fpr GPT Input
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#Call ChatGpt API
def GPTCallCohesion(file,task_in,depthim,poses):
    client = OpenAI(api_key="x")
    base64_image = encode_image(file)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": "You are an assistant that generates affordance maps to aid robot learning and generates task sequences for robots."},
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"Based on the image, task in hand: {task_in} and the depth data: {depthim}. Generate an affordance map of each object in the scene called affordance_map and output it as a JSON format containing object_name and affordances. The output should be a JSON format only!!. Generate the task sequence too in JSON format, called task_sequence, any form of transformation would be referred to as move (MOVE actions should have a target position based on the {poses},depth data and image) and any action related to grippers will be pick-up and place.No comments please",
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
    )
    data = response.choices[0].message.content.strip("```json").strip('```')
    # print("Before RegEx")
    # print(data)
    data = re.sub(r'//.*?\n', '', data) #RegEX Implementation
    # print("After RegEx")
    # print(data)
    jsondata = json.loads(data)
    affordancedata = jsondata['affordance_map']
    taskdata = jsondata['task_sequence']
    print(f"Generated Affordance Map: {affordancedata}")
    print(f"Generated Task Sequence: {taskdata}")
    return affordancedata, taskdata
    
        
def CommandExecution(taskcohesion):
    for task in taskcohesion:
        if task['action'] == "pick-up":
            if not gripper.isOn():
                gripper.turnOn()
            print("turned grippper on")
            desired_delay = 20000
            num_steps = int(desired_delay / timestep)
            
            for _ in range(num_steps):
                robot.step(timestep)
                
        elif task['action'] == "place":
            if gripper.isOn():
                gripper.turnOff()
            print("turned grippper off")
            desired_delay = 20000
            num_steps = int(desired_delay / timestep)
            
            for _ in range(num_steps):
                robot.step(timestep)
                
        else:
            targetpos_in_cameraframe = task['target_position']

            #Kinematics Method
            initialjointpos = [0]
            for sensor in sensors:
                initialjointpos.append(sensor.getValue())

            fk_result = ur5e_chain.forward_kinematics(initialjointpos)
            targetpos_in_cameraframe.append(1.0)
            targetpos_in_cameraframe = np.array(targetpos_in_cameraframe).reshape(4, 1)
            targetpos_in_baseframe = fk_result @ targetpos_in_cameraframe
            targetpos = targetpos_in_baseframe[:3].flatten().tolist()
            print(f"Moving to {targetpos}")

            ik_results = ur5e_chain.inverse_kinematics(targetpos)

            for i, motor in enumerate(motors):
                motor.setPosition(ik_results[i+1])
            
            desired_delay = 20000
            num_steps = int(desired_delay / timestep)
            
            for _ in range(num_steps):
                robot.step(timestep)
                
    for motor in motors:
        motor.setPosition(float(0))
    print("Completed task, give me another!")



#Set up task listener
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 5000))
s.listen()
print("Give me a task to do!")
while robot.step(timestep) != -1:
    try:
        s.settimeout(0.1)
        conn, addr = s.accept()
        task_in = conn.recv(1024).decode('utf-8')
        print(f"Task: {task_in}")
    except socket.timeout:
        continue

    #Get RGB Image
    image = camera.getImage()
    rgbimarr = np.frombuffer(image,dtype = np.uint8).reshape((height,width,4))[:,:,:3].copy()
    rgbimarr.setflags(write=1)

    #Get Depth Image
    depthim = depthimager.getRangeImage()
    depthimarr = np.array(depthim).reshape((height,width))

    #OBJ Detection
    objposes = obj_detect(rgbimarr, depthimarr)

    #GPT Task Cohesion
    affordance, taskcohesion = GPTCallCohesion(filepath,task_in,depthimarr,objposes)

    #Execute Commands
    CommandExecution(taskcohesion)
