#pragma once

#include <array>
#include <stdint.h>

// Dof number
const int NUM_MOTOR = 29; 

// Stiffness for all G1 Joints
std::array<float, NUM_MOTOR> Kp{
    100, 100, 100, 200, 20, 20,      // legs
    100, 100, 100, 200, 20, 20,      // legs
    400, 400, 400,                   // waist
    90, 60, 20, 60,  20, 20, 20,  // arms
    90, 60, 20, 60,  20, 20, 20,  // arms
};

// Damping for all G1 Joints
std::array<float, NUM_MOTOR> Kd{
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,     // legs
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,     // legs
    5.0, 5.0, 5.0,              // waist
    2.0, 1.0, 0.4, 1.0, 0.5, 0.5, 0.5,  // arms
    2.0, 1.0, 0.4, 1.0, 0.5, 0.5, 0.5   // arms
};

//Default joint position
std::array<float, NUM_MOTOR> default_joint_position = {
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,    //legs
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,    //legs
    0.0, 0.0, 0.0,                      // waist
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,  // arms
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0   // arms
};

// Joint index
enum JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13,           // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14,           // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
  LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,  // NOTE INVALID for g1 23dof
  RightWristYaw = 28     // NOTE INVALID for g1 23dof
};

// Ankle control mode; Specific to G1
enum PRorAB { PR = 0, AB = 1 };
