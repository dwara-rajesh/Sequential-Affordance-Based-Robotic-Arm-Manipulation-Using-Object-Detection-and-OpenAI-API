from ikpy.chain import Chain
import numpy as np
from controller import Robot,Camera,RangeFinder,VacuumGripper
import cv2,json,math,socket,re

#Initialize Robot Chain
ur5e_chain = Chain.from_urdf_file(r"robot_model.urdf", active_links_mask=[False, True, True, True, True, True, True, False])
# print(ur5e_chain)

#Initialize Robotic Arm 
robot = Robot()

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

#Get joint sensors
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

#Set up task listener
r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
r.bind(('localhost', 5000))
r.listen() 

initialtask = True
taskgiven = False

print("Give me a task to do!")
while robot.step(timestep) != -1:
    try:
        r.settimeout(0.1)
        conn, addr = r.accept()
        
        if not taskgiven and initialtask:
            task_in = conn.recv(4096).decode('utf-8')
            print(f"Task: {task_in}")
            if task_in:
                taskgiven = True
                initialtask = False
        else:
            command = conn.recv(4096).decode('utf-8')
            print(f"Command: {command}")
            if command == "All sub tasks complete":
                initialtask = True
                for motor in motors:
                    motor.setPosition(float(0))
                print("Waiting for next task")
    except socket.timeout:
        continue
    
    if taskgiven:       
        #Get RGB Image
        image = camera.getImage()   
        rgbimarr = np.frombuffer(image,dtype = np.uint8).reshape((height,width,4))[:,:,:3].copy()
        rgbimarr.setflags(write=1)
        
        #Get Depth Image
        depthim = depthimager.getRangeImage()    
        depthimarr = np.array(depthim).reshape((height,width))

        depth_list = depthimarr.tolist()
        
        depthjson_path = r"depth_list.json"
        with open(depthjson_path, "w") as json_file:
            json.dump(depth_list, json_file)
        
        # #OBJ Detection
        objposes = obj_detect(rgbimarr, depthimarr)
        
        posjson_path = r"pos_list.json"
        with open(posjson_path, "w") as json_file:
            json.dump(objposes, json_file)
        
        data = "data transferred"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
           s.connect(('localhost', 5001))
           s.sendall(data.encode('utf-8')) 
           print("Sent")
        taskgiven = False                             
    else:
        if command == "gripper on":
            if not gripper.isOn():
                gripper.turnOn()
            print("turned grippper on")
        elif command == "gripper off":
            if gripper.isOn():
                gripper.turnOff()
            print("turned grippper off")
        else:
            #move to position
            start = command.find("[")
            end = command.find("]")
            if start != -1 and end != -1:
                targetposition_in_camera_frame = list(map(float, command[start+1:end].split(',')))
                #Kinematics Method
                initialjointpos = [0]
                for sensor in sensors:
                    initialjointpos.append(sensor.getValue())
                fk_result = ur5e_chain.forward_kinematics(initialjointpos)
                targetposition_in_camera_frame.append(1.0)
                targetposition_in_camera_frame = np.array(targetposition_in_camera_frame).reshape(4, 1)
                targetposition_in_base_frame = fk_result @ targetposition_in_camera_frame
                targetpos = targetposition_in_base_frame[:3].flatten().tolist()
                print(f"Moving to {targetpos}") 
            
                ik_results = ur5e_chain.inverse_kinematics(targetpos)
                for i, motor in enumerate(motors):
                    motor.setPosition(ik_results[i+1])