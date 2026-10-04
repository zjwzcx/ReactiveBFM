/*
This is the transition layer for Deployment on Unitree series robot

The main logic for this code is that it serves as the transition layer between the policy(high-level) and the robots(low-level).

                                                Policy (High Level)
                                                       |
                                                Transition Layer (HERE)
                                                       |
                                                Unitree Robot (Low Level)

The code is based on the prevailing LCM to build communication between different parts and support easy transfer between any unitree robots (For example, G1->H1)

You may define the robot-specific params in unitree_sdk2/assets where you can create a new file and define the params as unitree_g1_29dof.hpp

The detailed implementation of communication is:

                                                Policy (High Level)
                                                   LCM||LCM
                                                Transition Layer (HERE)
                                   (Wrapped by LCM)DDS||DDS
                                                Unitree Robot (Low Level)

The left side represents the direction from up to down and the right side is the opposite.
*/

// General Headers

// Standard Content
#include <cmath>
#include <memory>
#include <thread>
#include <lcm/lcm-cpp.hpp>

// Unitree
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include "unitree/common/thread/thread.hpp"

// LCM
#include "robot_state_lcmt.hpp"
#include "robot_cmd_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "../remote_controller/remote_controller.hpp"

// Robot-specific configs
#include "g1_29dof_dex3.hpp" // weishuai: you only have to modify this to include diverse hyper-parameters for your controller

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
#define TOPIC_SPORT_STATE "rt/odommodestate"

static const std::string HG_LEFT_HAND_CMD_TOPIC = "rt/dex3/left/cmd";
static const std::string HG_LEFT_HAND_STATE_TOPIC = "rt/lf/dex3/left/state";

static const std::string HG_RIGHT_HAND_CMD_TOPIC = "rt/dex3/right/cmd";
static const std::string HG_RIGHT_HAND_STATE_TOPIC = "rt/lf/dex3/right/state";

uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else
                CRC32 <<= 1;
            if (data & xbit)
                CRC32 ^= dwPolynomial;

            xbit >>= 1;
        }
    }
    return CRC32;
};

typedef struct {
    uint8_t id     : 4;
    uint8_t status : 3;
    uint8_t timeout: 1;
} RIS_Mode_t;

/*---------------------------Here is the main body of the controller-----------------------------*/
class RobotController {
private:
    double time_;
    double control_dt_;  // 0.002s-500HZ
    double duration_;    // time for moving to default pose
    PRorAB mode_;        // mode for control ankle
    uint8_t mode_machine_;

    /*Communication between control interface and low level humanoid robots*/
    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_; //Unitree state subscriber and publisher
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> odometer_subscriber_;

    /*Hand publisher and subscriber*/
    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> leftcmd_publisher_;
    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> rightcmd_publisher_;

    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> leftstate_subscriber_;
    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> rightstate_subscriber_;

    /*Communication between control interface and high level policy*/
    lcm::LCM _simpleLCM;

    robot_state_lcmt robot_state_simple = {0};
    robot_cmd_lcmt robot_cmd_simple = {0};
    rc_command_lcmt rc_command = {0};

    /*Multi-threads*/
    unitree::common::ThreadPtr highstateWriterThreadPtr, lowcmdWriterThreadPtr;
    std::thread highcmdReceiverThreadPtr;

    /*Indicators*/
    bool _firstRun;
    bool _firstCommandReceived;
    bool _firstLowCmdReceived;
    bool _firstHighCmdReceived;
    bool _firstOdometerMsgReceived;
    bool _firstLeftStateReceived;
    bool _firstRightStateReceived;
    // bool _firstPointLioMsgReceived;

    /*Data buffer*/
    unitree_hg::msg::dds_::LowState_ low_state{};
    unitree_hg::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::SportModeState_ odometer_state{};
    unitree_hg::msg::dds_::HandState_ left_state;
    unitree_hg::msg::dds_::HandState_ right_state;
    unitree_hg::msg::dds_::HandCmd_ left_cmd;
    unitree_hg::msg::dds_::HandCmd_ right_cmd;
    xRockerBtnDataStruct remote_key_data;

public:
        RobotController(std::string networkInterface): 
            time_(0.0),
            control_dt_(0.005), // 200HZ
            duration_(5.0), //time for moving to default pose
            mode_(PR), // ankle control mode
            mode_machine_(0)
    {
        // Init network connection
        unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

        left_state.motor_state().resize(NUM_HAND_MOTOR);
        right_state.motor_state().resize(NUM_HAND_MOTOR);
        left_cmd.motor_cmd().resize(NUM_HAND_MOTOR);
        right_cmd.motor_cmd().resize(NUM_HAND_MOTOR);
        left_state.press_sensor_state().resize(NUM_SENSOR);
        right_state.press_sensor_state().resize(NUM_SENSOR);

        set_default_state();

        /*-------Create Communication between transition layer and the low-level humanoid robots------*/
        // create publisher (transition layer -> robot)
        lowcmd_publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
        lowcmd_publisher_->InitChannel();

        // create subscriber (robot -> transition layer)
        lowstate_subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
                HG_STATE_TOPIC));
        lowstate_subscriber_->InitChannel(
            std::bind(&RobotController::lowstateHandler, this, std::placeholders::_1), 1);

        odometer_subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
                TOPIC_SPORT_STATE));
        odometer_subscriber_->InitChannel(
            std::bind(&RobotController::OdometerHandler, this, std::placeholders::_1), 1);

        leftcmd_publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_> (HG_LEFT_HAND_CMD_TOPIC));
        leftcmd_publisher_->InitChannel();

        rightcmd_publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_> (HG_RIGHT_HAND_CMD_TOPIC));
        rightcmd_publisher_->InitChannel();

        leftstate_subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_> (HG_LEFT_HAND_STATE_TOPIC));
        leftstate_subscriber_->InitChannel(
            std::bind(&RobotController::LeftstateHandler, this, std::placeholders::_1), 1
        );

        rightstate_subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_> (HG_RIGHT_HAND_STATE_TOPIC));
        rightstate_subscriber_->InitChannel(
            std::bind(&RobotController::RightstateHandler, this, std::placeholders::_1), 1
        );

        // create writer (which uses publisher) (transition layer -> robot)
        lowcmdWriterThreadPtr = unitree::common::CreateRecurrentThreadEx("dds_write_thread", UT_CPU_ID_NONE, control_dt_ * 1e6, &RobotController::lowcmdWriter, this);

        /*--------Create Comminication between transition layer and the high-level policy-------*/
        // create lcm subscriber (policy action -> transition layer); Receiver receives high-level signals and hands over to handler for processing.
        _simpleLCM.subscribe("pd_plustau_targets", &RobotController::highcmdHandler, this);
        highcmdReceiverThreadPtr = std::thread(&RobotController::highcmdReceiver, this);

        // lcm send thread (transition layer -> policy)
        highstateWriterThreadPtr = unitree::common::CreateRecurrentThreadEx("lcm_send_thread", UT_CPU_ID_NONE, control_dt_*1e6, &RobotController::highstateWriter, this);

        _firstRun = true;
        _firstCommandReceived = false;
        _firstLowCmdReceived = false;
        _firstHighCmdReceived = false;
        _firstOdometerMsgReceived = false;
        _firstLeftStateReceived = false;
        _firstRightStateReceived = false;
    }

    /*Initialization*/
    void set_default_state(){
        for(int i=0; i<NUM_MOTOR+2*NUM_HAND_MOTOR; i++){
            robot_cmd_simple.q_des[i] = default_joint_position[i];
            robot_cmd_simple.qd_des[i] = 0;
            robot_cmd_simple.tau_ff[i] = 0;
            robot_cmd_simple.kp[i] = Kp[i];
            robot_cmd_simple.kd[i] = Kd[i];
        }
        std::cout << "Default Joint Position Set!" << std::endl;
    }

    /*-----------Communication with the high-level layer--------------*/
    // High command receive(It will hand over the control signal to handler): Policy -> Transition Layer
    void highcmdReceiver(){
        while(true){
            _simpleLCM.handle();
        }
    }

    void highcmdHandler(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const robot_cmd_lcmt *msg){
        (void) rbuf;
        (void) chan;
        robot_cmd_simple = *msg;

        if (!_firstHighCmdReceived){
            _firstHighCmdReceived = true;
            std::cout<< "Communication built successfully between transition layer and policy!" << std::endl;
        }
    }

    //High state writer: Transition Layer -> Policy; You may only 
    void highstateWriter() {
        for(int i=0; i<NUM_MOTOR; i++){
            robot_state_simple.q[i] = low_state.motor_state()[i].q();
            robot_state_simple.qd[i] = low_state.motor_state()[i].dq();
            robot_state_simple.tau_est[i] = low_state.motor_state()[i].tau_est();
        }

        for(int i=0; i<NUM_HAND_MOTOR; i++){
            robot_state_simple.q[NUM_MOTOR+i] = left_state.motor_state()[i].q();
            robot_state_simple.qd[NUM_MOTOR+i] = left_state.motor_state()[i].dq();
            robot_state_simple.tau_est[NUM_MOTOR+i] = left_state.motor_state()[i].tau_est();

            robot_state_simple.q[NUM_MOTOR+NUM_HAND_MOTOR+i] = right_state.motor_state()[i].q();
            robot_state_simple.qd[NUM_MOTOR+NUM_HAND_MOTOR+i] = right_state.motor_state()[i].dq();
            robot_state_simple.tau_est[NUM_MOTOR+NUM_HAND_MOTOR+i] = right_state.motor_state()[i].tau_est();
        }

        for(int i=0; i<4; i++){
            robot_state_simple.quat[i] = low_state.imu_state().quaternion()[i];
            robot_state_simple.lquat[i] = left_state.imu_state().quaternion()[i];
            robot_state_simple.rquat[i] = right_state.imu_state().quaternion()[i];
        }

        for(int i=0; i<3; i++){
            robot_state_simple.rpy[i] = low_state.imu_state().rpy()[i];
            robot_state_simple.aBody[i] = low_state.imu_state().accelerometer()[i];
            robot_state_simple.omegaBody[i] = low_state.imu_state().gyroscope()[i];
            robot_state_simple.p[i] = odometer_state.position()[i];
            robot_state_simple.vBody[i] = odometer_state.velocity()[i];

            robot_state_simple.lrpy[i] = left_state.imu_state().rpy()[i];
            robot_state_simple.rrpy[i] = right_state.imu_state().rpy()[i];

            robot_state_simple.laBody[i] = left_state.imu_state().accelerometer()[i];
            robot_state_simple.raBody[i] = right_state.imu_state().accelerometer()[i];

            robot_state_simple.lomegaBody[i] = left_state.imu_state().gyroscope()[i];
            robot_state_simple.romegaBody[i] = right_state.imu_state().gyroscope()[i];
        }

        for(int i=0; i<9; i++){
            for(int j=0; j<12; j++){
                robot_state_simple.lpressure[i][j] = left_state.press_sensor_state()[i].pressure()[j];
                robot_state_simple.rpressure[i][j] = right_state.press_sensor_state()[i].pressure()[j];
            }
        }


        if(mode_machine_ != low_state.mode_machine()){
            if(mode_machine_ == 0)
                std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
            mode_machine_ = low_state.mode_machine();
        }

        memcpy(&remote_key_data, &low_state.wireless_remote()[0], 40);
        rc_command.left_stick[0] = remote_key_data.lx;
        rc_command.left_stick[1] = remote_key_data.ly;
        rc_command.right_stick[0] = remote_key_data.rx;
        rc_command.right_stick[1] = remote_key_data.ry;
        rc_command.right_lower_right_switch = remote_key_data.btn.components.R2;
        rc_command.right_upper_switch = remote_key_data.btn.components.R1;
        rc_command.left_lower_left_switch = remote_key_data.btn.components.L2;
        rc_command.left_upper_switch = remote_key_data.btn.components.L1;

        _simpleLCM.publish("robot_state_data", &robot_state_simple);
        _simpleLCM.publish("rc_command_data", &rc_command);
    }


    void lowstateHandler(const void *message) {
        /*
        The lowstateHandler is mainly responsible for the following things:
        1. Update the current proprioception state
        2. Obtain the remote controller state
        3. Update the signal across threads
        */

        low_state = *(const unitree_hg::msg::dds_::LowState_ *)message;

        if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1))
        {
            std::cout << "low_state CRC Error" << std::endl;
            return;
        }

        if (_firstLowCmdReceived == false)
        {
            std::cout << "Communication built successfully between transition layer and robot!" <<std::endl;
            _firstLowCmdReceived = true;
        }
    }

    void OdometerHandler(const void *message) {

        odometer_state = *(unitree_go::msg::dds_::SportModeState_ *) message;

        if(_firstOdometerMsgReceived == false)
        {
            std::cout << "Commnication built successfully between transition layer and Unitree Odometer!" << std::endl;
            _firstOdometerMsgReceived = true;
        }
    }

    void LeftstateHandler(const void *message) {
        left_state = *(unitree_hg::msg::dds_::HandState_ *) message;

        if(_firstLeftStateReceived == false)
        {
            std::cout << "Communication build successfully between translation layer and Dex-3 Left Hand!" << std::endl;
            _firstLeftStateReceived = true;
        }
    }

    void RightstateHandler(const void *message) {
        right_state = *(unitree_hg::msg::dds_::HandState_ *) message;

        if(_firstRightStateReceived == false)
        {
            std::cout << "Communication build successfully between translation layer and Dex-3 Right Hand!" << std::endl;
            _firstRightStateReceived = true;
        }
    }

    void lowcmdWriter() {

        low_cmd.mode_pr() = mode_;
        low_cmd.mode_machine() = mode_machine_;


        if(time_ < duration_){
            time_ += control_dt_;

            float ratio = time_ / duration_;
            for(int i = 0; i<NUM_MOTOR; i++){
                low_cmd.motor_cmd().at(i).mode() = 1;
                low_cmd.motor_cmd()[i].kp() = Kp[i];
                low_cmd.motor_cmd()[i].kd() = Kd[i];
                low_cmd.motor_cmd()[i].dq() = 0.f;
                low_cmd.motor_cmd()[i].tau() = 0.f;

                float q_des = default_joint_position[i];

                q_des = (q_des - robot_state_simple.q[i]) * ratio + robot_state_simple.q[i];
                low_cmd.motor_cmd()[i].q() = q_des;
            }

            for (int i = 0; i < NUM_HAND_MOTOR; i++)
            {
                RIS_Mode_t ris_mode;
                ris_mode.id = i;
                ris_mode.status = 0x01;
                ris_mode.timeout = 0;

                uint8_t hand_mode = 0;
                hand_mode |= (ris_mode.id & 0x0F);
                hand_mode |= (ris_mode.status & 0x07) << 4;
                hand_mode |= (ris_mode.timeout & 0x01) << 7;

                left_cmd.motor_cmd().at(i).mode() = hand_mode;
                right_cmd.motor_cmd().at(i).mode() = hand_mode;

                left_cmd.motor_cmd()[i].dq(0);
                right_cmd.motor_cmd()[i].dq(0);

                left_cmd.motor_cmd()[i].tau(0);
                right_cmd.motor_cmd()[i].tau(0);

                left_cmd.motor_cmd()[i].kp(Kp[NUM_MOTOR+i]);
                right_cmd.motor_cmd()[i].kp(Kp[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                left_cmd.motor_cmd()[i].kd(Kd[NUM_MOTOR+i]);
                right_cmd.motor_cmd()[i].kd(Kd[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                float lq_des = default_joint_position[NUM_MOTOR+i];
                float rq_des = default_joint_position[NUM_MOTOR+NUM_HAND_MOTOR+i];

                lq_des = (lq_des - robot_state_simple.q[NUM_MOTOR+i])*ratio + robot_state_simple.q[NUM_MOTOR+i];
                rq_des = (rq_des - robot_state_simple.q[NUM_MOTOR+NUM_HAND_MOTOR+i]) * ratio + robot_state_simple.q[NUM_MOTOR+NUM_HAND_MOTOR+i];

                left_cmd.motor_cmd()[i].q(lq_des);
                right_cmd.motor_cmd()[i].q(rq_des);
            }
        }

        else{
            if (_firstRun){
                for(int i=0; i<NUM_MOTOR; i++)
                    robot_cmd_simple.q_des[i] = robot_state_simple.q[i];
                for(int i=0; i<NUM_HAND_MOTOR; i++){
                    robot_cmd_simple.q_des[i+NUM_MOTOR] = robot_state_simple.q[i+NUM_MOTOR];
                    robot_cmd_simple.q_des[i+NUM_MOTOR+NUM_HAND_MOTOR] = robot_state_simple.q[i+NUM_MOTOR+NUM_HAND_MOTOR];
                }
                remote_key_data.btn.components.A = 0;
                remote_key_data.btn.components.B = 0;
                remote_key_data.btn.components.L2 = 0;
                _firstRun = false;
            }

            if(((int) remote_key_data.btn.components.B ==1 && (int) remote_key_data.btn.components.L2 == 1)){
                for(int i=0; i<NUM_MOTOR; i++){
                    low_cmd.motor_cmd()[i].q() = 0;
                    low_cmd.motor_cmd()[i].dq() = 0;
                    low_cmd.motor_cmd()[i].kp() = 0;
                    low_cmd.motor_cmd()[i].kd() = 10;
                    low_cmd.motor_cmd()[i].tau() = 0;
                }

                for (int i = 0; i < NUM_HAND_MOTOR; i++){
                    RIS_Mode_t ris_mode;
                    ris_mode.id = i;
                    ris_mode.status = 0x01;
                    ris_mode.timeout = 0x01;

                    uint8_t hand_mode = 0;
                    hand_mode |= (ris_mode.id & 0x0F);
                    hand_mode |= (ris_mode.status & 0x07) << 4;
                    hand_mode |= (ris_mode.timeout & 0x01) << 7;

                    left_cmd.motor_cmd().at(i).mode() = hand_mode;
                    right_cmd.motor_cmd().at(i).mode() = hand_mode;

                    left_cmd.motor_cmd()[i].q(0);
                    right_cmd.motor_cmd()[i].q(0);

                    left_cmd.motor_cmd()[i].dq(0);
                    right_cmd.motor_cmd()[i].dq(0);

                    left_cmd.motor_cmd()[i].tau(0);
                    right_cmd.motor_cmd()[i].tau(0);

                    left_cmd.motor_cmd()[i].kp(0);
                    right_cmd.motor_cmd()[i].kp(0);

                    left_cmd.motor_cmd()[i].kd(0);
                    right_cmd.motor_cmd()[i].kd(0);
                }

                std::cout << "Switched to Damping Mode!" << std::endl;

                low_cmd.crc() = Crc32Core((uint32_t *)&low_cmd, (sizeof(low_cmd)>>2)-1);
                lowcmd_publisher_->Write(low_cmd);
                leftcmd_publisher_->Write(left_cmd);
                rightcmd_publisher_->Write(right_cmd);

                sleep(1.5);

                while(true){

                    if((int) remote_key_data.btn.components.B ==1 && (int) remote_key_data.btn.components.L2 == 1) {
                        std::cout << "L2+B is pressed again, Exit!" << std::endl;
                        exit(0);
                    }

                    else{
                        std::cout<<"Press L2+B again to exit!" <<std::endl;
                        sleep(0.01);
                    }
                }
            }

            else{
                for(int i=0; i<NUM_MOTOR; i++){
                    low_cmd.motor_cmd()[i].q() = robot_cmd_simple.q_des[i];
                    low_cmd.motor_cmd()[i].dq() = robot_cmd_simple.qd_des[i];
                    low_cmd.motor_cmd()[i].kp() = robot_cmd_simple.kp[i];
                    low_cmd.motor_cmd()[i].kd() = robot_cmd_simple.kd[i];
                    low_cmd.motor_cmd()[i].tau() = robot_cmd_simple.tau_ff[i];
                }

                for (int i = 0; i < NUM_HAND_MOTOR; i++){
                    RIS_Mode_t ris_mode;
                    ris_mode.id = i;
                    ris_mode.status = 0x01;

                    uint8_t hand_mode = 0;
                    hand_mode |= (ris_mode.id & 0x0F);
                    hand_mode |= (ris_mode.status & 0x07) << 4;
                    hand_mode |= (ris_mode.timeout & 0x01) << 7;

                    left_cmd.motor_cmd().at(i).mode() = hand_mode;
                    right_cmd.motor_cmd().at(i).mode() = hand_mode;

                    left_cmd.motor_cmd()[i].q(robot_cmd_simple.q_des[NUM_MOTOR+i]);
                    right_cmd.motor_cmd()[i].q(robot_cmd_simple.q_des[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                    left_cmd.motor_cmd()[i].dq(robot_cmd_simple.qd_des[NUM_MOTOR+i]);
                    right_cmd.motor_cmd()[i].dq(robot_cmd_simple.qd_des[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                    left_cmd.motor_cmd()[i].kp(robot_cmd_simple.kp[NUM_MOTOR+i]);
                    right_cmd.motor_cmd()[i].kp(robot_cmd_simple.kp[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                    left_cmd.motor_cmd()[i].kd(robot_cmd_simple.kd[NUM_MOTOR+i]);
                    right_cmd.motor_cmd()[i].kd(robot_cmd_simple.kd[NUM_MOTOR+NUM_HAND_MOTOR+i]);

                    left_cmd.motor_cmd()[i].tau(robot_cmd_simple.tau_ff[NUM_MOTOR+i]);
                    right_cmd.motor_cmd()[i].tau(robot_cmd_simple.tau_ff[NUM_MOTOR+NUM_HAND_MOTOR+i]);
                }
            }
        }

        low_cmd.crc() = Crc32Core((uint32_t *)&low_cmd, (sizeof(low_cmd)>>2)-1);
        lowcmd_publisher_->Write(low_cmd);

        leftcmd_publisher_->Write(left_cmd);
        rightcmd_publisher_->Write(right_cmd);
    }

};

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " networkInterface"<< std::endl;
        exit(-1);
    }

    std::cout << "Make sure the robot is hung up!" << std::endl
              << "You should not run the deploy code until the robot has moved to default positions!" <<std::endl
              << "Remote-control safety actions:" << std::endl
              << "  Single press [L2 + B]: switch the robot to damping mode." << std::endl
              << "  Double press [L2 + B]: terminate the controller after entering damping mode." << std::endl
              << "Press Enter to continue ..." <<std::endl;
    std::cin.ignore(); // Press Enter to continue

    std::string networkInterface = argv[1];
    RobotController custom(networkInterface);
    while (true) usleep(20000); // 0.02s
    return 0;
}
