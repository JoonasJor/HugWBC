from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
import unitree_legged_const as h1

H1_NUM_MOTOR = 20

class H1JointIndex:
    # Right leg
    kRightHipYaw = 8
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kRightAnkle = 11

    # Left leg
    kLeftHipYaw = 7
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kLeftAnkle = 10

    # Torso
    kWaistYaw = 6
    kNotUsedJoint = 9

    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

class LowLevelControl:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.01  
        self.duration_ = 10.0    
        self.counter_ = 0
        self.kp_low_ = 60.0
        self.kp_high_ = 200.0
        self.kd_low_ = 1.5
        self.kd_high_ = 5.0
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  
        self.InitLowCmd()
        self.low_state = None 
        self.crc = CRC()

    def Init(self):
        # create publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def is_weak_motor(self,motor_index):
        return motor_index in {
            H1JointIndex.kLeftAnkle,
            H1JointIndex.kRightAnkle,
            H1JointIndex.kRightShoulderPitch,
            H1JointIndex.kRightShoulderRoll,
            H1JointIndex.kRightShoulderYaw,
            H1JointIndex.kRightElbow,
            H1JointIndex.kLeftShoulderPitch,
            H1JointIndex.kLeftShoulderRoll,
            H1JointIndex.kLeftShoulderYaw,
            H1JointIndex.kLeftElbow,
        }

    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(H1_NUM_MOTOR):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01 
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A 
            self.low_cmd.motor_cmd[i].q= h1.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = h1.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    #def Start(self):
    #    self.lowCmdWriteThreadPtr = RecurrentThread(
    #        interval=self.control_dt_, target=self.LowCmdWrite, name="control"
    #    )
    #    self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

    def ReportRPY(self):
        print("rpy: [",self.low_state.imu_state.rpy[0],", "
                    ,self.low_state.imu_state.rpy[1],", "
                    ,self.low_state.imu_state.rpy[2],"]"
        )

    def LowCmdWriteJointTorques(self, torques, stiffness, damping):
        for i in range(H1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = torques[i]     
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].dq = 0                
            self.low_cmd.motor_cmd[i].kp = stiffness[i]
            self.low_cmd.motor_cmd[i].kd = damping[i] 

            # tau = Joint target torque  
            # q   = Joint target position
            # dq  = Joint target speed
            # kp  = Joint stiffness coefficient
            # kd  = Joint damping coefficient

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def LowCmdWriteJointAngles(self, angles, stiffness, damping):

        for i in range(H1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].tau =  0
            self.low_cmd.motor_cmd[i].q = self.low_state.motor_state[i].q - angles[i]
            self.low_cmd.motor_cmd[i].dq = 0.0                
            self.low_cmd.motor_cmd[i].kp = stiffness[i]
            self.low_cmd.motor_cmd[i].kd = damping[i] 

            # tau = Joint target torque  
            # q   = Joint target position
            # dq  = Joint target speed
            # kp  = Joint stiffness coefficient
            # kd  = Joint damping coefficient

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)

        for i, cmd in enumerate(self.low_cmd.motor_cmd):
            print(f"Motor {i}: mode={cmd.mode}, q={cmd.q}, kp={cmd.kp}, kd={cmd.kd}")
        
        self.lowcmd_publisher_.Write(self.low_cmd)

    def LowCmdWriteZero(self):
        for i in range(H1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
