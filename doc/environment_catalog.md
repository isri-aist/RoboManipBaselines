# Environment catalog

You can teleoperate a robot in each environment by giving `<env_name>` (e.g. `MujocoUR5eCable`) as an argument to `Teleop.py` as follows:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Teleop.py <env_name>
```

## MuJoCo environments
### UR5e
#### MujocoUR5eCable
Task to pass the cable between two poles.

https://github.com/user-attachments/assets/006e7337-2cfa-41d7-80a9-18cb40c013c4

#### MujocoUR5eRing
Task to pick a ring and put it around the pole.

https://github.com/user-attachments/assets/29c4001e-5d6e-4414-ad70-ebb7a25bf69e

#### MujocoUR5eParticle
Task to scoop up particles.

https://github.com/user-attachments/assets/759cddeb-8089-4e44-907c-2e8405ea026c

#### MujocoUR5eCloth
Task to roll up the cloth.

https://github.com/user-attachments/assets/0b7a52d5-5ce3-49f2-b574-a78e729865cc

#### MujocoUR5eDoor
Task to open a door.

https://github.com/user-attachments/assets/738a1b13-0b32-4d82-87ea-f7c60ae46849

#### MujocoUR5eToolbox
Task to pick up a toolbox and place it.

https://github.com/user-attachments/assets/fb3e1e5b-db10-475a-8263-57e0fda2cfaf

#### MujocoUR5eCabinet
Task to open a cabinet with a sliding door and a hinged door.

https://github.com/user-attachments/assets/ff0d8be0-1d6c-4f8d-aaca-de451c13df7d

#### MujocoUR5eInsert
Task to insert a pin into a hole.

https://github.com/user-attachments/assets/6435a06d-fcd2-45ec-88da-5a26852dd4d8

#### MujocoUR5ePick
Task to pick up objects (9 types) and place them into baskets (2 types).

https://github.com/user-attachments/assets/77ad9c36-4cda-4a09-a799-205386ed682c

### Dual UR5e
#### MujocoUR5eDualCable
Task to manipulate a cable with two grippers.

https://github.com/user-attachments/assets/6cfbafe6-1072-419d-892f-d66b7da8709f

### xArm7
#### MujocoXarm7Cable
Task to pass the cable between two poles.

https://github.com/user-attachments/assets/450d04e8-7154-4c01-849c-d1f9b50be0ae

#### MujocoXarm7Ring
Task to pick a ring and put it around the pole.

https://github.com/user-attachments/assets/da83db94-7fcb-4690-9b73-41af0c1394a8

#### MujocoXarm7Pusht
Task to push the T-shaped block to align the pose.

https://github.com/user-attachments/assets/000ae494-8ed5-423a-a26e-652b7817c372

### ALOHA
#### MujocoAlohaCable
Task to manipulate a cable with two grippers.

https://github.com/user-attachments/assets/dfa154ab-4ca6-42bf-82aa-d45f189dbcc2

#### MujocoAlohaHandover
Task to transfer an object grasped by the left gripper to the right gripper and place it.

https://github.com/user-attachments/assets/4ceef03a-11ac-47de-b3cc-e5ec222a1191

### HSR
#### MujocoHsrTidyup
Task to tidy up objects on the floor.

https://github.com/user-attachments/assets/2e5aef49-10da-4cf5-ba8a-1c9240672eff

### Unitree G1
#### MujocoG1Bottles
Task to grasp and manipulate bottles.

https://github.com/user-attachments/assets/67b18ba1-c35f-460e-b44e-0c39eaff8dc9

## Isaac environments
### UR5e
#### IsaacUR5eChain
Task to pick a chain and hang it on a hook.

https://github.com/user-attachments/assets/803759a5-be70-4530-b9ba-b5d381b1bb7a

#### IsaacUR5eCabinetVec
Task to open the top lid of the box.  
The parallel simulation feature of Isaac Gym allows multiple robots to be controlled in parallel while adding random noise.

https://github.com/user-attachments/assets/4f171bd3-7572-41a4-9e81-9df5dfec575a

## Real-world environments
### UR5e
#### RealUR5eDemo
Various manipulation tasks with UR5e in the real world.

https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

See [here](./real_ur5e.md) for instructions on how to operate real robot.

### xArm7
#### RealXarm7Demo
Various manipulation tasks with xArm7 in the real world

https://github.com/user-attachments/assets/ab0c9830-5b33-48e8-9dac-272460a51a39
