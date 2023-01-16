#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include "Reward.h"
#include "COPReward.h"
#include "COMReward.h"
#include "torqueReward.h"
#include "Fixedeque.h"

#include <queue>
#include <deque>
#include <iostream>

#define HISTORY_BUFFER_LEN 3


namespace MASS
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};


class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetUseExo(bool use_Exo){mUseExo = use_Exo;}
	void SetUseMuscleNN(bool use_muscleNetWork){mUseMuscleNN = use_muscleNetWork;}
	void SetUseHumanNN(bool use_humanNetwork){mUseHumanNN = use_humanNetwork;}
	void SetSymmetry(bool symmetry){mSymmetry = symmetry;}
	void SetUseCOP(bool use_COP){mUseCOP = use_COP;}
    void SetCOMindependent(bool COM_independent){mCOMindependent = COM_independent;}
	void Settargetmotion_visual(bool target_motion_visualization){mUsetarget_visual = target_motion_visualization;}
	void Sethumanobj_visual(bool human_obj_visualization) {mhuman_obj_visual = human_obj_visualization;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}
	void SetTerminalTime (double terminal_time) {mterminal_time = terminal_time;}
	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}
	void SetWalkingSkill(bool walking) {walk_skill = walking;}
	void SetSquattingSkill(bool squatting) {squat_skill = squatting;}
	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com,double w_torque,double w_root){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;this->w_torque = w_torque;this->w_root=w_root;}
	void SetPDParameters(double kp) {this->kp=kp;}
	void SetSmoothRewardParameters(double w_sroot, double w_saction, double w_storque,double w_sjoint_vel){this->w_sroot = w_sroot;this->w_saction = w_saction;this->w_storque = w_storque;this->w_sjoint_vel = w_sjoint_vel;}
	void SetFootClearanceRewardParameter(double w_footclr) {this->W_footclr = W_footclr;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
    void SetJointRewardParameters(double w_hip,double w_knee,double w_ankle){this->w_hip = w_hip;this->w_knee = w_knee;this->w_ankle = w_ankle;}
	void SetFootClearance(double a, double b) {high_clearance = a; low_clearance =b; }
	void SetFootTolerances(double a) {foot_tolerance = a;}
	void SetPlotRelatedMuscle(bool plot_related_muscle){mPlotRelatedMuscle = plot_related_muscle;}

public:
	void ProcessAction(int j, int num); 
	void Step();
	void Step_test(); 
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState(); // get p and v
	Eigen::VectorXd GetHumanState();
	Eigen::VectorXd GetControlState(); // get delayed states 
	Eigen::VectorXd GetControlHumanState(); // get delayed states 
	Eigen::VectorXd GetFullObservation(); // get total observation
	Eigen::VectorXd GetControlCOP(); 

	void SetAction(const Eigen::VectorXd& a);
	void SetHumanAction(const Eigen::VectorXd& a);
	double GetReward();
	double GetHumanReward();
	std::tuple<double,double,double,double,double,double,Eigen::VectorXd,Eigen::VectorXd,double,double,double,double,double,double> GetRenderReward_Error();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	MuscleTuple GetCurrentMuscleTuple(){return mCurrentMuscleTuple;}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumFullObservation(){return mNumFullObservation;}
	int GetNumHumanObservation() {return mNumHumanObservation;}

	int GetNumAction(){return mNumExoActiveDof-3;}
	int GetNumHumanAction() {return mNumHumanActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a);
	bool GetUseMuscle(){return mUseMuscle;}
	bool GetWalkSkill() {return walk_skill;}
	bool GetSquatSkill() {return squat_skill;}
	bool GetUseMuscleNN(){return mUseMuscleNN;}
	bool GetUseHumanNN() {return mUseHumanNN;}
	bool GetUseSymmetry(){return mSymmetry;}
	bool GetUsetargetvisual() {return mUsetarget_visual;}
	bool GetUsehumanobjvisual() {return mhuman_obj_visual;}
	bool GetPlotRelatedMuscle() {return mPlotRelatedMuscle;}
	std::map<std::string,Eigen::Vector3d> Get_mTargetEE_pos(){return mTargetEE_pos;}
	Eigen::Vector3d geo_center_target, geo_center_target_left, geo_center_target_right;
	std::map<std::string, Reward*> GetmReward(){return mReward;}
    Eigen::VectorXd GetTargetObservations();
	Eigen::VectorXd GetCOPRelative();
	Eigen::VectorXd GetAction() {return mExoAction;}
	Eigen::VectorXd GetHumanAction() {return mHumanAction;}
	void randomize_masses(double lower_bound, double upper_bound);
	void randomize_inertial(double lower_bound, double upper_bound);
	void randomize_motorstrength(double lower_bound, double upper_bound);
	void randomize_controllatency(double lower_bound, double upper_bound);
	void randomize_friction(double lower_bound, double upper_bound);
	void randomize_centerofmass(double lower_bound, double upper_bound);
	void randomize_muscleisometricforce(double lower_bound, double upper_bound);
	void randomize_terrain();


 	// void applyWeldJointconstraint();
	void UpdateStateBuffer();
	void UpdateActionBuffer(Eigen::VectorXd action);
	void UpdateTorqueBuffer(Eigen::VectorXd torque);
	void UpdateHumanActionBuffer(Eigen::VectorXd humanaction);
	Eigen::VectorXd CopyVector(Eigen::VectorXd v){ Eigen::VectorXd out(v.size()); out << v; return out;}
	//dart::dynamics::SkeletonPtr createFloor();
private:
	
	std::vector<Eigen::Vector3d> contact_pos_left; 
	std::vector<Eigen::Vector3d> contact_pos_right; 
	std::vector<Eigen::Vector3d> contact_force_left; 
	std::vector<Eigen::Vector3d> contact_force_right;
	FixedQueue<Eigen::VectorXd> history_buffer_true_state;
	FixedQueue<Eigen::VectorXd> history_buffer_true_human_state;
	FixedQueue<Eigen::VectorXd> history_buffer_control_state;
	FixedQueue<Eigen::VectorXd> history_buffer_control_human_state;
	FixedQueue<Eigen::VectorXd> history_buffer_true_COP;
	Eigen::VectorXd control_COP; 
	FixedQueue<Eigen::VectorXd> history_buffer_action;
	FixedQueue<Eigen::VectorXd> history_buffer_human_action;
	
	FixedQueue<Eigen::VectorXd> history_buffer_torque;
    dart::constraint::WeldJointConstraintPtr mWeldJoint;
	std::map<std::string, Reward*> mReward;
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	bool mUseExo;
	bool mUseMuscleNN;
	bool mSymmetry;
	bool mUseCOP;
	bool mUsehuman;
	bool mUseHumanNN;
	bool mUseExoinitialstate;
	bool mUseHumaninitialstate;
	bool mCOMindependent;
	bool mUsejointconstraint;
	bool mUsetarget_visual;
	bool mhuman_obj_visual;
	bool mPlotRelatedMuscle;
	bool squat_skill, walk_skill;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mCurrentExoAction, mPrevExoAction; 
	Eigen::VectorXd mCurrentHumanAction, mPrevHumanAction; 
	Eigen::VectorXd dynamic_torque;
	Eigen::VectorXd mExoAction;
	Eigen::VectorXd mHumanAction;
	Eigen::VectorXd  mHumanAction_des;
	Eigen::VectorXd mTargetPositions,mTargetVelocities,mHuman_initial,mExo_initial;
	std::map<std::string,Eigen::Vector3d> mTargetEE_pos;
    Eigen::VectorXd Initial_masses;
	Eigen::MatrixXd Initial_inertia;
	Eigen::VectorXd Initial_centerofmass;
	int mNumState;
	int mNumExoActiveDof;
	int mRootJointDof;
	int mNumHumanActiveDof;
	int mNumFullObservation;
	int mNumHumanState;
	int mNumHumanObservation;
	int cnt_step;
	Eigen::VectorXd randomized_strength_ratios;
	double randomized_latency;
	double observation_latency;
	double mterminal_time;
	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	Eigen::VectorXd mLastDesiredTorque;
	Eigen::VectorXd mcur_joint_vel;
	Eigen::VectorXd mlast_joint_vel;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;
	double lastUpdateTimeStamp; 
	double w_q,w_v,w_ee,w_com,w_COP,w_torque,w_root;
	double w_sroot,w_saction, w_storque,w_sjoint_vel;
	double W_footclr;
	double w_hip, w_knee, w_ankle;
	double kp,kv;
	double high_clearance, low_clearance;
	double foot_tolerance;
    std::vector<size_t> const_index_Exo;
	std::vector<size_t> const_index_human;
	std::vector<size_t> const_index_withoutroot;
	Eigen::VectorXd mpos_diff;
	Eigen::VectorXd mvel_diff;
	Eigen::VectorXd mee_diff;
};
};

#endif