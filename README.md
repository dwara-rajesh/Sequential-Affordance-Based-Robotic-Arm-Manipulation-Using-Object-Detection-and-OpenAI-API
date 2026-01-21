# Simulating a Robotic Arm for Sequential Affordance-Based Manipulation Using Object Detection and OpenAI API  

This repository contains the implementation and simulation files for the paper:  

**Haroon Muhammed, Dwarakesh Rajesh**  
*Simulating a Robotic Arm for Sequential Affordance-Based Manipulation Using Object Detection and OpenAI API*  

The project investigates how **Large Language Models (LLMs)**, specifically ChatGPT, can be integrated with **object detection** and **affordance reasoning** to enable robotic arms to perform **multi-step manipulation tasks** in response to natural language commands.  

Simulation & Implementation files: https://drive.google.com/drive/folders/1n3EUZbuvptDlwyPfofB9im6LaRWsYxmC?usp=drive_link

---

## 🧩 Research Objectives  
1. Can ChatGPT correctly infer **object affordances** from detected objects and task descriptions?  
2. How reliable are LLM-generated **task sequences** in ensuring successful completion of multi-step tasks?  
3. What challenges arise when integrating **object detection**, **LLMs**, and **robotic control**?  
4. Which **robot learning paradigm** (end-to-end vs modular pipeline) is most effective for LLM-based manipulation?  

---

## 📖 Methodology  

### Simulation Environment  
- **Platform**: Webots R2023b  
- **Robot**: UR5e robotic arm  
- **Sensors**: Astra RGB-D camera (for perception)  
- **End-Effector**: Robotiq EPick vacuum gripper  
- **Setup**: Task table with manipulable objects (e.g., soccer ball)  

### Framework Overview  
The system consists of three core modules:  

1. **Object Detection**  
   - YOLOv4 (trained on COCO dataset)  
   - RGB + depth data from Astra camera  
   - Non-Maxima Suppression (NMS) to reduce duplicate detections  

2. **Affordance & Task Sequence Generation**  
   - Input: natural language command + object detections + depth data  
   - GPT-based affordance mapping + JSON-formatted task sequences  
   - User feedback loop for refinement  

3. **Task Execution**  
   - Inverse kinematics with [ikpy](https://github.com/Phylliade/ikpy)  
   - Action primitives: `move`, `pick-up`, `place`  
   - Frame transformations from camera → robot base  

---

## 🔬 Experiments  

Six experimental setups were tested:  

1. **Baseline Modular Pipeline** (no feedback, vague prompt) → *5% success*  
2. **Prompt Engineering** (no feedback) → *40% success*  
3. **Regex Cleaning of GPT Output** (no feedback) → *80% success*  
4. **Regex + Prompt Engineering** (no feedback) → *85% success*  
5. **Feedback Loop Integration** → *Most reliable task execution*  
6. **End-to-End GPT Framework** (object detection + affordance + sequencing) → *Failed due to inaccurate object localization*  

**Key Finding**: Modular pipeline + prompt engineering + regex + user feedback yielded the highest reliability.  

---

## 📊 Results  

- Success rate improved from **5% → 85%** with modular pipeline refinements.  
- Feedback loop significantly reduced unnecessary or malformed actions.  
- Limitations observed in **inverse kinematics accuracy** due to URDF precision in Webots.  

---

## 📌 Future Work  

- Transition to **CoppeliaSim** or **Isaac Sim** for more accurate manipulator dynamics.  
- Automate user feedback for continuous self-correction.  
- Extend affordance reasoning to **multi-object, cluttered environments**.  
- Explore **reinforcement learning + LLM hybrid frameworks**.  
