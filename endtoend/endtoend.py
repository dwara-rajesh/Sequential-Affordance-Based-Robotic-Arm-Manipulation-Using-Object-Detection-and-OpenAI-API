from ikpy.chain import Chain
import numpy as np
from controller import Supervisor,Camera,RangeFinder,VacuumGripper
import json,math,socket,re,cv2

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

filepath = r"CameraFeed.png"
        
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
        cv2.imwrite(filepath, rgbimarr)
        
        #Get Depth Image
        depthim = depthimager.getRangeImage()    
        depthimarr = np.array(depthim).reshape((height,width))

        depth_list = depthimarr.tolist()
        
        depthjson_path = r"depth_list.json"
        with open(depthjson_path, "w") as json_file:
            json.dump(depth_list, json_file)
        
        camerapos = cam.getPosition()
        tosend = [width,height,cx,cy,hfov,vfov,fx,fy,camerapos[0],camerapos[1],camerapos[2]]
        data = f"data {tosend}"
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
                targetpos = list(map(float, command[start+1:end].split(',')))
                print(f"Moving to {targetpos}") 
            
                ik_results = ur5e_chain.inverse_kinematics(targetpos)
                for i, motor in enumerate(motors):
                    motor.setPosition(ik_results[i+1])