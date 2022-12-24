#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "Force.h"
#include "BodyForce.h"
#include "SpringForce.h"
#include "Reward.h"
#include "COPReward.h"
#include "COMReward.h"
#include "torqueReward.h"
#include<cmath>
#include <algorithm> 
#include <random>
//#include "Fixedeque.h"

#include "dart/collision/bullet/bullet.hpp"
#include "dart/constraint/ContactConstraint.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;


Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mterminal_time(10.0),mWorld(std::make_shared<World>()),mUseMuscle(true),mUseHumaninitialstate(false),mUseExoinitialstate(false),
	mUseMuscleNN(false),mUseHumanNN(false), mSymmetry(true), mCOMindependent(true), mUseCOP(false), mUsehuman(false), observation_latency(0.04), w_q(0.7),w_v(0.1),w_ee(0.5),w_com(0.4),w_torque(0.00),w_root(0.4),w_hip(0.3),w_knee(0.7),w_ankle(0.3),w_sroot(0.0),w_saction(0.0),w_storque(0.0),w_sjoint_vel(0.0),W_footclr(0.0)
{
	// mRewards = RewardFactory::mRewards;	
	history_buffer_true_state.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_control_state.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_true_human_state.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_control_human_state.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_true_COP.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_action.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_human_action.setMaxLen(HISTORY_BUFFER_LEN);
	history_buffer_torque.setMaxLen(HISTORY_BUFFER_LEN);

}
// initialize ： read metal_file，load use_muscle, con_hz, sim_hz, skel_file, map_file, bvh_file and reward_param
void
Environment::
Initialize(const std::string& meta_file,bool load_obj)
{	
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}
		std::cout <<"hreerwre11111111 " << std::endl;
	std::string str;        // define str
	std::string index;      // define index
	std::stringstream ss;   // define 


	for(int _i = 0; _i++; _i<2)
	{
		MASS::Character* character = new MASS::Character();   //
		character ->SetEnvironment(*this);

		while(!ifs.eof())
		{
			str.clear();
			index.clear();
			ss.clear();

			std::getline(ifs,str);
			ss.str(str);
			ss>>index;
			if(!index.compare("skill"))
			{	
				std::string str2;
				ss>>str2;    
				if(!str2.compare("walking")) 
				{                       
					this->SetWalkingSkill(true);
					character->walk_skill=true;
				}
				else if(!str2.compare("squatting"))  
				{
					this->SetSquattingSkill(true);
					character->squat_skill=true;
				}
			}
			if(!index.compare("use_muscle"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetUseMuscle(true);           
				else                                     //mUseMuscle = true
					this->SetUseMuscle(false);
			}
			if(!index.compare("use_muscleNetWork"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetUseMuscleNN(true);           
				else                                     //mUseMuscle = true
					this->SetUseMuscleNN(false);
			}
			if(!index.compare("use_humanNetwork"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetUseHumanNN(true);           
				else                                     //mUseMuscle = true
					this->SetUseHumanNN(false);
			}
			else if(!index.compare("symmetry"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetSymmetry(true);           
				else                                     //mSymmetry = true
					this->SetSymmetry(false);
			}
			else if(!index.compare("use_COP"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetUseCOP(true);           
				else                                     //mUseCOP = true
					this->SetUseCOP(false);
			}
			else if(!index.compare("com_independent"))
			{	
				std::string str2;
				ss>>str2;
				if(!str2.compare("true"))
					this->SetCOMindependent(true);           
				else                                     //mCOMindependent = true
					this->SetCOMindependent(false);
			}
			else if(!index.compare("con_hz")){
				int hz;
				ss>>hz;                              // control_hz = 30hz = mControlHz
				this->SetControlHz(hz);
			}
			else if(!index.compare("sim_hz")){
				int hz;
				ss>>hz;                             // sim_hz = 600hz  = mSimulationHz
				this->SetSimulationHz(hz);
			}
			else if(!index.compare("terminal_time")){	
				double terminal_time;	
				ss>>terminal_time;                            	
				this->SetTerminalTime(terminal_time);	
			}
			else if(!index.compare("PD_param")){
				double a;
				ss>>a;
				this->SetPDParameters(a);  // reward parameters w_q, w_v, w_ee, w_com, w_torque
			}
			// else if(!index.compare("foot_clearance_termination")){
			// 	double a,b;
			// 	ss>>a>>b;
			// 	this->SetFootClearance(a,b);  // reward parameters w_q, w_v, w_ee, w_com, w_torque
			// }
			// else if(!index.compare("foot_tolerance_termination")){
			// 	double a;
			// 	ss>>a;
			// 	this->SetFootTolerances(a);  // reward parameters w_q, w_v, w_ee, w_com, w_torque
			// }
			else if(!index.compare("target_motion_visualization"))	
			{		
				std::string str2;	
				ss>>str2;	
				if(!str2.compare("true"))	
					this->Settargetmotion_visual(true);           	
				else                                    	
					this->Settargetmotion_visual(false);	
			}
			else if(!index.compare("exo_file")){
				std::string str2;                                                               //read skel_file
				ss>>str2;
				character->LoadExofromUrdf(std::string(MASS_ROOT_DIR)+str2,load_obj);
				mUseExo= true;
				character->MergeHumanandExo();
			}
			else if(!index.compare("human_obj_visualization")){
				std::string str2;                                                               //read skel_file
				ss>>str2;
				if(!str2.compare("true"))	
					this->Sethumanobj_visual(true);
				if(!str2.compare("false"))	
					this->Sethumanobj_visual(false);
			}
			else if(!index.compare("human_file")){
				std::string str2,str3;                                                               //read skel_file
				ss>>str2>>str3;
				if(!str3.compare("true"))
				{
					character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
					mUsehuman = true;
				}
			}
			else if(!index.compare("muscle_file")){
				std::string str2;
				ss>>str2;
				if(this->GetUseMuscle())
					character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
				if(mUsehuman)
					character->ChangeHumanframe();
			}
			else if(!index.compare("muscle_force_limit_visual")){	
				double max_visual, min_visual;	
				ss>>max_visual>>min_visual ;                            	
				character->SetMusclelimitforVisual(max_visual, min_visual);	
			}
			else if(!index.compare("plot_related_muscle")){	
				std::string str2;	
				ss>>str2;	
				if(!str2.compare("true"))	                      	
					this->SetPlotRelatedMuscle(true);	
			}
			else if(!index.compare("model_component_file")){
				// std::cout << "============" << std::endl; 
				std::string str2,str3;
				ss>>str2>>str3;
				if(!str3.compare("true"))
					character->LoadModelComponents(std::string(MASS_ROOT_DIR)+str2);
			}	
			else if(!index.compare("Human_spring_force_file")){	
				std::string str2,str3;    //str2： path, str3: true	
				ss>>str2>>str3;	
				if(!str3.compare("true"))	
					character->LoadHumanforce(std::string(MASS_ROOT_DIR)+str2);	
			}			
			else if(!index.compare("Human_bushing_force_file")){	
				std::string str2,str3;    //str2： path, str3: true	
				ss>>str2>>str3;	
				if(!str3.compare("true"))	
					character->LoadHumanforce(std::string(MASS_ROOT_DIR)+str2);	
			}	
			// else if(!index.compare("map_file")){
			// 	std::string str2;                                                             //read map_file
			// 	ss>>str2;
			// 	character->LoadMap(std::string(MASS_ROOT_DIR)+str2);
			// }
			else if(!index.compare("exo_initial_state")){ 
				std::string str2;                                                             //read map_file
				ss>>str2;
				character->LoadExoInitialState(std::string(MASS_ROOT_DIR)+str2);
				mUseExoinitialstate = true;
			}
			else if(!index.compare("human_initial_state")){ 
				std::string str2;                                                             //read map_file
				ss>>str2;
				character->LoadHumanInitialState(std::string(MASS_ROOT_DIR)+str2);
				mUseHumaninitialstate = true;
			}
			else if(!index.compare("Joint_constraint_file")){
				// std::cout << "============" << std::endl; 
				std::string str2,str3;
				ss>>str2>>str3;
				if(!str3.compare("true"))
					character->LoadConstraintComponents(std::string(MASS_ROOT_DIR)+str2);
			}
			// else if(!index.compare("bvh_file")){        
			//                       //read map_file
			// 	std::string str2,str3;    //str2： path, str3: true
			// 	ss>>str2>>str3;
			// 	bool cyclic = false;
			// 	if(!str3.compare("true"))
			// 		cyclic = true;    
			// 	character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
			// }
			else if(!index.compare("motion_file")){                                              //read map_file
				std::string str2,str3;    //str2： path, str3: true
				ss>>str2>>str3;
				bool cyclic = false;
				if(!str3.compare("true"))
					cyclic = true;    
				character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
			}

			else if(!index.compare("reward_param")){
				double a,b,c,d,e,f;
				ss>>a>>b>>c>>d>>e>>f;
				this->SetRewardParameters(a,b,c,d,e,f);  // reward parameters w_q, w_v, w_ee, w_com, w_torque
			}
			else if(!index.compare("smooth_reward_param")){	
				double a,b,c,d;	
				ss>>a>>b>>c>>d;	
				this->SetSmoothRewardParameters(a,b,c,d);  // smoothreward parameters w_sroot, w_saction, w_storque	
			}
			// else if(!index.compare("foot_clearance_reward")){	
			// 	double a;	
			// 	ss>>a;	
			// 	this->SetFootClearanceRewardParameter(a);  // smoothreward parameters w_sroot, w_saction, w_storque	
			// 	std::cout << "hrewr9 "  << std::endl;
			// }
			else if(!index.compare("observation_latency")){
				double a;
				ss>>a;
				this->observation_latency = a;
			}
			else if(!index.compare("reward")){
				std::string str2;
				ss >> str2; 
				auto reward = RewardFactory::CreateReward(str2);
				reward->SetEnvironment(*this); 
				reward->ReadFromStream(ss); 
				mReward.insert(std::pair<std::string, Reward*>(reward->GetName(),reward)); 
			}
			else if(!index.compare("joint_reward_weight")){
				double a,b,c;
				ss>>a>>b>>c;
				this->SetJointRewardParameters(a,b,c);  // reward parameters w_q, w_v, w_ee, w_com, w_torque
			}
		}
		ifs.close();      // close metafile
		
		// if (mUseExo)
		// 	character->MergeHumanandExo();

		character->SetPDParameters(kp,sqrt(2.0*kp));   // PD parameters： kp=400, kv=sqrt(2*kp)
		// this->SetCharacter(character);
		this->AddCharacter(character);
	}
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));
	randomize_terrain();
	
	this->Initialize();
	mUsejointconstraint = false;
}


// dart::dynamics::SkeletonPtr 
// Environment::
// createFloor()
// {
//   SkeletonPtr floor = Skeleton::create("floor");
  
//   // Give the floor a body
//   BodyNodePtr body =
//       floor->createJointAndBodyNodePair<WeldJoint>(nullptr).second;
  
//   // Give the body a shape
//   double floor_width = 100.0;
//   double floor_height = 0.01;
//   std::shared_ptr<BoxShape> box(
//       new BoxShape(Eigen::Vector3d(floor_width, floor_height, floor_width)));
//   auto shapeNode
//       = body->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(box);
//   shapeNode->getVisualAspect()->setColor(dart::Color::Black());
  
//   // Put the body into position
//   Eigen::Isometry3d tf(Eigen::Isometry3d::Identity());

//   tf.translation() = Eigen::Vector3d(0.0, -1.39, 0.0);
//   tf.linear() = tf.linear() * R_z(45);
//   body->getParentJoint()->setTransformFromParentBodyNode(tf);
  
//   return floor;
// }


void
Environment::
Initialize()   // define the related dofs
{	

	for(std::vector<T>::size_type i = 0; i != mCharacters.size(); i++) {
		Character* mCharacter = mCharacters[i];
	
		if(mCharacter->GetSkeleton()==nullptr){
			std::cout<<"Initialize character First"<<std::endl;
			exit(0);
		}
		if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")   //FreeJoint: 6 degree
			mRootJointDof = 6;
		else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint") //PlanarJoint:
			mRootJointDof = 3;	
		else
			mRootJointDof = 0;
			
		mNumHumanActiveDof = mCharacter->GetHumandof()-mRootJointDof; 
		if (mUseExo)    
			mNumExoActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mNumHumanActiveDof - mRootJointDof ;  // remove the root Exo dof, ground_exo_waist(prismatic joint) and hipadd
		else
			mNumExoActiveDof = 0; 
		std::cout << "human dof -----"  << mNumHumanActiveDof << "  exo dof -----"  << mNumExoActiveDof << std::endl;  // 
		std::cout << "total dof -----" << mCharacter->GetSkeleton()->getNumDofs() << std::endl;
		if(mUseMuscle)
		{
			int num_total_related_dofs = 0;
			for(auto m : mCharacter->GetMuscles()){
				m->Update();    //
				num_total_related_dofs += m->GetNumRelatedDofs();   //num_total_related_dofts = m + num_related_dofs
			}
			mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);  // define vector JtA
			mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumHumanActiveDof,mCharacter->GetMuscles().size()); //define Matrix L
			mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumHumanActiveDof); // define vector b
			mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumHumanActiveDof); // define vector tau_dex
			mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size()); // define activation levels a
			std::cout << "muscle.size():" << mActivationLevels.rows() << std::endl;
		}
		

		mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));  // set gravity
		mWorld->setTimeStep(1.0/mSimulationHz);     // set time step
		mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
		mWorld->addSkeleton(mCharacter->GetSkeleton());  //
		mWorld->addSkeleton(mGround);

		mExoAction = Eigen::VectorXd::Zero(mNumExoActiveDof-3);  // define mAction vector
		mCurrentExoAction = Eigen::VectorXd::Zero(mNumExoActiveDof-3);  // define mAction vector
		mPrevExoAction = Eigen::VectorXd::Zero(mNumExoActiveDof-3);  // define mAction vector

		mHumanAction = Eigen::VectorXd::Zero(mNumHumanActiveDof);
		mHumanAction_des = Eigen::VectorXd::Zero(mNumHumanActiveDof);;
		mCurrentHumanAction = Eigen::VectorXd::Zero(mNumHumanActiveDof);  // define mAction vector
		mPrevHumanAction = Eigen::VectorXd::Zero(mNumHumanActiveDof);  // define mAction vector

		dynamic_torque = Eigen::VectorXd::Zero(mNumExoActiveDof-3);
		mDesiredTorque = Eigen::VectorXd::Zero(mCharacter->GetSkeleton()->getNumDofs());	
		// mcur_joint_vel = Eigen::VectorXd::Zero(mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof);

		for (int i=0;i<mCharacter->GetHumandof();i++)
			const_index_human.push_back(i);
			
		for (int i=mCharacter->GetHumandof();i<mCharacter->GetSkeleton()->getNumDofs();i++)
			const_index_Exo.push_back(i);

		std::cout << "mskeletondof:    " << mCharacter->GetSkeleton()->getNumDofs() << std::endl;
		int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();

		Initial_masses = Eigen::VectorXd::Zero(Numbodynodes);
		Initial_inertia = Eigen::MatrixXd::Zero(3*Numbodynodes,3);
		Initial_centerofmass = Eigen::VectorXd::Zero(Numbodynodes*3);
		for (int i=0;i<Numbodynodes;i++)
		{
		Initial_masses(i) = mCharacter->GetSkeleton()->getBodyNode(i)->getMass();
		}
		for (int i=0;i<Numbodynodes;i++)
		{
			const Inertia& iner= mCharacter->GetSkeleton()->getBodyNode(i)->getInertia();
			Initial_inertia.block(i*3,0,3,3) = iner.getMoment();
		}

		for (int i=0;i<Numbodynodes;i++)
		{
			const Inertia& iner= mCharacter->GetSkeleton()->getBodyNode(i)->getInertia();
			Initial_centerofmass.segment<3>(i*3) = iner.getLocalCOM();
		}

		Reset(false);
		mNumState = GetState().rows();             // p.rows +v.rows +1 
		mNumHumanState = GetHumanState().rows();    // p.rows +v.rows +1 
		mNumHumanObservation = GetHumanState().rows();
		mNumFullObservation = GetFullObservation().rows();
		std::cout << "NumState:    " << mNumState << std::endl;
		std::cout << "NumHumanState:    " << mNumHumanState << std::endl;
		if (mSymmetry)
			std::cout << "Onput of NN:  " << mExoAction.rows()/2 << std::endl;
		else
			std::cout << "Onput of NN:  " << mExoAction.rows() << std::endl;
		if (mUseMuscle)
			std::cout << "--use Muscle--" << std::endl;
		else
			std::cout << "--not use Muscle--"  << std::endl;
	}

}
void 
Environment::
Reset(bool RSI)                                       	// execute the env.reset() after the terminal state is true
{
	mWorld->reset();                                    // set time = 0, Frame = 0, clear last Collision result in the world
	for(std::vector<T>::size_type i = 0; i != mCharacters.size(); i++) {
		Character* mCharacter = mCharacters[i];
		mCharacter->GetSkeleton()->clearConstraintImpulses();  
		mCharacter->GetSkeleton()->clearInternalForces();
		mCharacter->GetSkeleton()->clearExternalForces();
		double t = 0.0;
		if(RSI)
			t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);  //
		
		mWorld->setTime(t);   							// set time in the world 
		mCharacter->Reset();  							// reset mT0 = mBVH->GetT0();	 mTc.translation[1] =0;
		
		dynamic_torque.setZero(); 
		mExoAction.setZero();
		mCurrentExoAction.setZero();
		mPrevExoAction.setZero();
		mHumanAction.setZero();
		mPrevHumanAction.setZero();
		mCurrentHumanAction.setZero();
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);

		mTargetPositions = std::get<0>(pv);
		mTargetVelocities = std::get<1>(pv);
		mTargetEE_pos = std::get<2>(pv);
		mHuman_initial=Eigen::VectorXd::Zero(mCharacter->GetHumandof());
		mExo_initial=Eigen::VectorXd::Zero(mNumExoActiveDof);
		if (!RSI)
			{
				if (mUseHumaninitialstate)
				{
					// std::cout << " --Use Human Model--" << std::endl;
					mHuman_initial = mCharacter->GetHumanInitialState();
					// std::cout << mCharacter->GetHumandof() << "-----" << mHuman_initial.size() << "-----" << mHuman_initial << std::endl;
					mTargetPositions.head(mCharacter->GetHumandof()) = mHuman_initial;
				}
				if (mUseExoinitialstate)
				{
					// std::cout << "exo dof:::::::::" << mCharacter->GetExodof() << std::endl;
					mExo_initial = mCharacter->GetExoInitialState();
					mTargetPositions.tail(mNumExoActiveDof) = mExo_initial;	
				}
			}
		cnt_step =0;
		
		// testing

		randomized_latency = 0;
		randomized_strength_ratios.setOnes(mNumExoActiveDof);

		/////////randomization
		randomize_masses(0.9,1.1); 
		randomize_inertial(0.9,1.1); 
		randomize_centerofmass(0.9,1.1); 
		randomize_motorstrength(0.9,1.1); 
		randomize_friction(0.9,1.1); 
		randomize_controllatency(0, observation_latency);

		// mCharacter->GetSkeleton()->setPositions(const_index_withoutroot, Positions_withoutroot);
		
		mCharacter->GetSkeleton()->setPositions(mTargetPositions);
		mCharacter->GetSkeleton()->setVelocities(mTargetVelocities); //set velocities
		mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
		for(int i=0; i<HISTORY_BUFFER_LEN; i++)
		{
			history_buffer_true_state.push_back(this->GetState());
			history_buffer_control_state.push_back(this->GetState());
			history_buffer_action.push_back(this->GetAction());
			// history_buffer_true_human_state.push_back(this->GetHumanState());
			// history_buffer_control_human_state.push_back(this->GetHumanState());
			// history_buffer_true_COP.push_back(this->GetCOPRelative()); 
		
			// history_buffer_human_action.push_back(this->GetHumanAction());
			history_buffer_torque.push_back(this->GetDesiredTorques());
		}


		// control_COP = GetControlCOP(); 
		lastUpdateTimeStamp = 0.0; 
		if (mUsehuman)
		{
			if (!mUsejointconstraint)
			{
				for (auto ss : mCharacter->GetJointConstraints())
				{
					BodyNode* bn1 = mCharacter->GetSkeleton()->getBodyNode(std::get<0>(ss));
					BodyNode* bn2 = mCharacter->GetSkeleton()->getBodyNode(std::get<1>(ss));
					Eigen::Vector3d offset1;
					offset1 = bn1->getTransform()*std::get<2>(ss); 
					auto constraint1 = std::make_shared<dart::constraint::BallJointConstraint>(bn1, bn2, offset1);
					mWorld->getConstraintSolver()->addConstraint(constraint1);
					mUsejointconstraint = true;
				}
			}
		}
	}
}


void 
Environment::
randomize_masses(double lower_bound, double upper_bound)
{
    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
    Eigen::VectorXd sample = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_mass_ratios = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_mass = Eigen::VectorXd::Zero(Numbodynodes);
    for (int i=0;i<Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_mass_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}

	randomized_mass = randomized_mass_ratios.cwiseProduct(Initial_masses);
	// std::cout << "mass \n"  <<  randomized_mass_ratios << std::endl;
	if (mSymmetry)
	{
		mCharacter->GetSkeleton()->getBodyNode(0)->setMass(randomized_mass(0));
		mCharacter->GetSkeleton()->getBodyNode(1)->setMass(randomized_mass(1));
        for (int i=2;i<Numbodynodes;i++) 
			mCharacter->GetSkeleton()->getBodyNode(i)->setMass(randomized_mass_ratios(2)*Initial_masses(i));
	}
	else
	{
		for (int i=1;i<Numbodynodes;i++)
			mCharacter->GetSkeleton()->getBodyNode(i)->setMass(randomized_mass(i));
 	}
}


void 
Environment::
randomize_inertial(double lower_bound, double upper_bound)
{

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;

    Eigen::VectorXd sample = Eigen::VectorXd::Zero(Numbodynodes);
	Eigen::VectorXd randomized_inertial_ratios = Eigen::VectorXd::Zero(Numbodynodes);

    for (int i=0;i<Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_inertial_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}

   Eigen::Matrix3d randomized_inertial = Eigen::Matrix3d::Zero(3,3);
   // std::cout << "inertia \n"  <<  randomized_inertial_ratios << std::endl;
	for (int i=0;i<Numbodynodes;i++)
	{
		randomized_inertial = randomized_inertial_ratios(i) * Initial_inertia.block(i*3,0,3,3);  
		// Set moment of inertia defined around the center of mass.  
		mCharacter->GetSkeleton()->getBodyNode(i)->setMomentOfInertia(randomized_inertial(0,0),randomized_inertial(1,1),randomized_inertial(2,2), randomized_inertial(0,1),randomized_inertial(0,2),randomized_inertial(1,2)); 
	}
}


void 
Environment::
randomize_centerofmass(double lower_bound, double upper_bound)
{

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
	int cnt =0;
	if (mCOMindependent)
		cnt = 3;
	else
		cnt = 1;

	Eigen::VectorXd sample = Eigen::VectorXd::Zero(cnt*Numbodynodes);
	Eigen::VectorXd randomized_com_ratios = Eigen::VectorXd::Zero(cnt*Numbodynodes);
	for (int i=0;i<cnt*Numbodynodes;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_com_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}
	Eigen::Vector3d randomized_centerofmass;
	randomized_centerofmass.setZero();
	for (int i=0;i<Numbodynodes;i++)
	{
		randomized_centerofmass = randomized_com_ratios.segment(i*cnt,cnt).cwiseProduct(Initial_centerofmass.segment(i*cnt,cnt));  
		// Set center of mass.
	    // std::cout << Initial_centerofmass.segment(i*cnt,cnt) << "\n" << std::endl;
		mCharacter->GetSkeleton()->getBodyNode(i)->setLocalCOM(randomized_centerofmass); 
	}
}


void 
Environment::
randomize_motorstrength(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    Eigen::VectorXd sample= Eigen::VectorXd::Zero(mNumExoActiveDof);
	// randomized_strength_ratios = Eigen::VectorXd::Zero(mNumExoActiveDof);
    for (int i=0;i<mNumExoActiveDof;i++)
	{
		sample(i) = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
		randomized_strength_ratios(i) = (sample(i)-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	}
	// std::cout << "strength \n" << randomized_strength_ratios << std::endl;
}



void 
Environment::
randomize_controllatency(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    double sample = 0; 
	randomized_latency = 0;

	sample = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
	randomized_latency = (sample-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
	// std::cout << "latency \n" << randomized_latency << std::endl;
}



void 
Environment::
randomize_friction(double lower_bound, double upper_bound)
{
	double param_lower_bound = -1.0;
	double param_upper_bound = 1.0;
 
    double sample = 0; 
	double randomized_friction = 0;

    int Numbodynodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	sample = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
	randomized_friction = (sample-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound;
    // std::cout << "friction \n" << randomized_friction << std::endl;
	for (int i=0;i<Numbodynodes;i++)
		mCharacter->GetSkeleton()->getBodyNode(i)->setFrictionCoeff(randomized_friction);
}



// void 
// Environment::
// randomize_muscleisometricforce(double lower_bound, double upper_bound)
// {
// 	double param_lower_bound = -1.0;
// 	double param_upper_bound = 1.0;
 
//     double sample = 0; 
// 	double randomized_f0 = 0;
// 	for(auto muscle : mCharacter->GetMuscles())
// 		{
// 			sample = (float)rand()/RAND_MAX * (param_upper_bound - param_lower_bound) + param_lower_bound;
// 			randomized_f0 = muscle->f0*((sample-param_lower_bound)/(param_upper_bound-param_lower_bound)*(upper_bound-lower_bound) + lower_bound);
// 			muscle->Setf0(randomized_f0);
// 		}  
// }


void 
Environment::
randomize_terrain()
{
	Eigen::Isometry3d tf = mGround->getBodyNode(0)->getTransform();
	// tf.linear() = tf.linear() *R_z(5); 
	tf.translation()[1] = tf.translation()[1]; 
	// std::cout << tf.linear() << std::endl;
	mGround->getBodyNode(0)->getParentJoint()->setTransformFromParentBodyNode(tf);
	this->SetGround(mGround);
}


bool isSoftContact(const collision::Contact& contact)
{
  auto shapeNode1 = contact.collisionObject1->getShapeFrame()->asShapeNode();
  auto shapeNode2 = contact.collisionObject2->getShapeFrame()->asShapeNode();
  assert(shapeNode1);
  assert(shapeNode2);

  auto bodyNode1 = shapeNode1->getBodyNodePtr().get();
  auto bodyNode2 = shapeNode2->getBodyNodePtr().get();

  auto bodyNode1IsSoft =
      dynamic_cast<const dynamics::SoftBodyNode*>(bodyNode1) != nullptr;

  auto bodyNode2IsSoft =
      dynamic_cast<const dynamics::SoftBodyNode*>(bodyNode2) != nullptr;

  return bodyNode1IsSoft || bodyNode2IsSoft;
}


void
Environment::
Step()  
{	
	if(mUseMuscle)
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())
		{

			// muscle->activation = mActivationLevels[count++];
			// muscle->Update();
			// muscle->ApplyForceToBody();  // apply muscle force to body
		}
  
		if(mUseMuscleNN)
		{
			if(mSimCount == mRandomSampleIndex)
			{
				auto& skel = mCharacter->GetSkeleton();
				auto& muscles = mCharacter->GetMuscles();
				int n = skel->getNumDofs();
				int m = muscles.size();
				Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
				Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);
				int index =0;
				for(int i=0;i<muscles.size();i++)
				{
					auto muscle = muscles[i];
					// muscle->Update();
					Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
					auto Ap = muscle->GetForceJacobianAndPassive();

					JtA.block(0,i,n,1) = Jt*Ap.first;
					Jtp += Jt*Ap.second;
				}
				mCurrentMuscleTuple.JtA = GetMuscleTorques();
				mCurrentMuscleTuple.L = JtA.block(mRootJointDof, 0, mCharacter->GetHumandof()-mRootJointDof,m);
	
				mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof, mCharacter->GetHumandof()-mRootJointDof);
				mCurrentMuscleTuple.tau_des = mDesiredTorque.segment(mRootJointDof, mNumHumanActiveDof);
				mMuscleTuples.push_back(mCurrentMuscleTuple);
			}
			
		}
	
	}

	bool toUpdate = false;
	for(auto force : mCharacter->GetForces())
	{
		// std::cout << "name "  << force->GetName() << std::endl;		
		if (mWorld->getTime()-lastUpdateTimeStamp >= 0.3)
		{
			force->Update();
			// std::cout << "name1 "  << force->GetName() << std::endl;			
			toUpdate = true; 
		}
		// update spring force in the real time 
		if(force->GetName().find("springforce") != std::string::npos || force->GetName().find("bushingforce") != std::string::npos)
		{
			force->Update();
		}
		force->UpdatePos();
		force->ApplyForceToBody();
	}

	if(toUpdate)
	{
		lastUpdateTimeStamp = mWorld->getTime(); 
		toUpdate = false; 
	}

	GetDesiredTorques();
	
	if (mUseHumanNN && !mUseMuscleNN)
	{
		// std::cout << "---------mDesiredTorque-------"  << std::endl;
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);

	}
	else if (mUseMuscleNN)
	{
		// std::cout << "exo dof----" << mNumExoActiveDof << std::endl;
		// std::cout << "mDesiredTorque\n" << mDesiredTorque << std::endl;
		mCharacter->GetSkeleton()->setForces(const_index_Exo, mDesiredTorque.tail(mNumExoActiveDof));
		// std::cout << "----current positon-----\n"   << mCharacter->GetSkeleton()->getPositions().head(6) << std::endl;	
	}

	UpdateTorqueBuffer(mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof));

	mWorld->step();           
	mSimCount++;

}


Eigen::VectorXd clamp(Eigen::VectorXd x, double lo, double hi)
{
	for(int i=0; i<x.rows(); i++)
	{
		x[i] = (x[i] < lo) ? lo : (hi < x[i]) ? hi : x[i];
	}
	return x; 
}

double clamp(double x, double lo, double hi)
{

	x = (x< lo) ? lo : (hi < x) ? hi : x;

	return x; 
}

void
Environment::
ProcessAction(int substep_count, int num)
{
    double lerp = double(substep_count + 1) / num;     //substep_count: the step count should be between [0, num_action_repeat).
    mExoAction = mPrevExoAction + lerp * (mCurrentExoAction - mPrevExoAction);
	mHumanAction = mCurrentHumanAction; 
    // return proc_action;
	// if (mUseHumanNN)
	// {
	//  	mHumanAction = mPrevHumanAction + lerp * (mCurrentHumanAction - mPrevHumanAction);
	// }
}


Eigen::VectorXd
Environment::
GetDesiredTorques()
{	
	// Eigen::VectorXd p_des = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.segment(mRootJointDof,mNumHumanActiveDof) += mHumanAction;    // for human
	p_des[58] = mExoAction[0];    // for human
	p_des[60] = mExoAction[1];
	mDesiredTorque = mCharacter->GetSPDForces(p_des); 
	if (dart::math::isNan(mDesiredTorque))
	{
		// std::cout << "-----torque exist nan exo-----\n"   << mCharacter->GetSkeleton()->getPositions() << std::endl;
		// std::cout << "-----torque exist nan des-----\n"   << p_des << std::endl;
		// std::cout << "-----torque exist nan exo-----\n"   << mExoAction << std::endl;
		// std::cout << "-----torque exist human ----- \n"   << mHumanAction << std::endl;
	}
	// mDesiredTorque.head(mCharacter->GetHumandof()) = clamp(mDesiredTorque.head(mCharacter->GetHumandof()), -80, 80);
	mDesiredTorque.tail(mNumExoActiveDof) = randomized_strength_ratios.cwiseProduct(mDesiredTorque.tail(mNumExoActiveDof));
	mDesiredTorque.tail(mNumExoActiveDof) = clamp(mDesiredTorque.tail(mNumExoActiveDof), -20, 20);  //clamp torques
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}


Eigen::VectorXd
Environment::
GetTargetObservations()
{
	Eigen::Isometry3d mTc = mCharacter->GetmTc();
	// std::cout << "mTc: " << mTc.translation() << std::endl; 
	int dof = mCharacter->GetExodof();
    double time0 = mWorld->getTime();
	double dt = 1.0/mControlHz;
	Eigen::MatrixXd tar_poses(6,dof);
	Eigen::VectorXd tar_position; 
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(time0,dt);
	Eigen::VectorXd root_global = mCharacter->GetSkeleton()->getPositions().head(mRootJointDof);  //get current root global position
	for(int step=0; step < 6; step++)
	{
      	double time = time0 + (step+1)* dt;
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(time,dt);
		tar_position = std::get<0>(pv);
		Eigen::VectorXd root_rel = root_global-tar_position.head(mRootJointDof);
		tar_position.head(mRootJointDof) = root_rel;
		tar_poses.row(step) = tar_position.head(dof);
	}
	mCharacter->SetmTc(mTc);
	tar_poses.transposeInPlace();   // transpose
	Eigen::VectorXd tar_poses_v = Eigen::Map<const Eigen::VectorXd>(tar_poses.data(), tar_poses.size());  // matrix to vector
    return tar_poses_v;
}


Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index,JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}
	// std::cout << "jta.rows()" << mCurrentMuscleTuple.JtA.rows() << std::endl;
	return mCurrentMuscleTuple.JtA;
}

double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());   //L2 squareNorm()
}
double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
	return exp(-w*vec.squaredNorm());  ////L2 squareNorm()
}
double exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}


bool
Environment::
IsEndOfEpisode()    //
{
	bool isTerminal = false;
	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();
	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
    // Eigen::Vector3d pos_foot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot")->getCOM();
	// Eigen::Vector3d pos_foot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot")->getCOM();
	// double foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
	// double foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);
    Eigen::Vector6d root_pos = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	Eigen::Isometry3d cur_root_inv = mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().inverse();

	Eigen::Vector3d root_v = mCharacter->GetSkeleton()->getBodyNode(0)->getCOMLinearVelocity();
	double root_v_norm = root_v.norm();
	Eigen::Vector6d root_pos_diff = mTargetPositions.segment<6>(0) - root_pos;
	if(walk_skill==true)
	{
		if (root_y<1.25 || root_y >1.6)      //prevent falling down
			isTerminal =true;
		else if (dart::math::isNan(p) || dart::math::isNan(v))
			isTerminal =true;

	}	
	else if(squat_skill=true)
	{
		if (root_y<1.19 || root_y >1.42)      //prevent falling down
			isTerminal =true;
		else if (dart::math::isNan(p) || dart::math::isNan(v))
			isTerminal =true;
		else if(mWorld->getTime()>mterminal_time)     // input 
			isTerminal =true;
	}
	return isTerminal;
}



// states of the skeleton model include the position, velocity and phrase variable which represents the
                                     // normalized time elapsed in the reference motion. 
Eigen::VectorXd 
Environment::   
GetState()   
{
	auto& skel = mCharacter->GetSkeleton();     
	dart::dynamics::BodyNode* root = skel->getBodyNode(0);   // get root
	Eigen::VectorXd p_cur = skel->getPositions();
	Eigen::VectorXd v_cur = skel->getVelocities();

	Eigen::VectorXd p_save = Eigen::VectorXd::Zero(2);
	Eigen::VectorXd v_save = Eigen::VectorXd::Zero(2);
	p_save[0] = p_cur[58];
	p_save[1] = p_cur[60];
	v_save[0] = v_cur[58];
	v_save[1] = v_cur[60];  
    // current joint positions and velocities
	// Eigen::VectorXd p_cur, v_cur;
	// // remove global transform of the root
	// if (mUseExo)
	// 	p_cur.resize(p_save.rows()-6);
	// v_cur = v_save/10.0;
    Eigen::VectorXd state(p_save.rows()+v_save.rows()); //+tar_poses.rows());
	state<<p_save,v_save; //tar_poses;
	return state;
}


Eigen::VectorXd 
Environment::   
GetHumanState()   
{
	auto& skel = mCharacter->GetSkeleton();     
	Eigen::VectorXd p_human, v_human;
	p_human = skel->getPositions().head(mCharacter->GetHumandof());
	v_human = skel->getVelocities().head(mCharacter->GetHumandof());
	Eigen::VectorXd p_cur_human, v_cur_human;
	p_cur_human.resize(p_human.rows()-6);
	p_cur_human = p_human.tail(p_human.rows()-6);
	v_cur_human = v_human/10.0;
	Eigen::VectorXd human_state(p_cur_human.rows()+v_cur_human.rows());
	human_state << p_cur_human, v_cur_human;
	return human_state;
}


void 
Environment:: 
UpdateStateBuffer()
{
	history_buffer_true_state.push_back(this->GetState());
	history_buffer_control_state.push_back(this->GetControlState());
	// history_buffer_true_human_state.push_back(this->GetHumanState());
	// history_buffer_control_human_state.push_back(this->GetControlHumanState());
	// history_buffer_true_COP.push_back(this->GetCOPRelative());
	// store the delayed observation 
	// control_COP = GetControlCOP(); 
}



Eigen::VectorXd 
Environment:: 
GetControlState()
{
	double dt = 1.0/mControlHz;
	Eigen::VectorXd observation; 
	if((randomized_latency <= 0) || (history_buffer_true_state.size() == 1)){
    	observation = history_buffer_true_state.get(HISTORY_BUFFER_LEN-1);
	}else{
		int n_steps_ago = int(randomized_latency / dt);
		if(n_steps_ago + 1 >= history_buffer_true_state.size()){
			observation = history_buffer_true_state.get(HISTORY_BUFFER_LEN-1);
		}else{
			double remaining_latency = randomized_latency - n_steps_ago * dt; 
			double blend_alpha = remaining_latency / dt; 
			observation = (
				(1.0 - blend_alpha) * history_buffer_true_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 1)
				+ blend_alpha * history_buffer_true_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 2)); 
		}
	}

    return observation; 

}

Eigen::VectorXd 
Environment:: 
GetControlHumanState()
{
	double dt = 1.0/mControlHz;
	Eigen::VectorXd observation; 
	if((randomized_latency <= 0) || (history_buffer_true_human_state.size() == 1)){
    	observation = history_buffer_true_human_state.get(HISTORY_BUFFER_LEN-1);
	}else{
		int n_steps_ago = int(randomized_latency / dt);
		if(n_steps_ago + 1 >= history_buffer_true_human_state.size()){
			observation = history_buffer_true_human_state.get(HISTORY_BUFFER_LEN-1);
		}else{
			double remaining_latency = randomized_latency - n_steps_ago * dt; 
			double blend_alpha = remaining_latency / dt; 
			observation = (
				(1.0 - blend_alpha) * history_buffer_true_human_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 1)
				+ blend_alpha * history_buffer_true_human_state.get(HISTORY_BUFFER_LEN - n_steps_ago - 2)); 
		}
	}

    return observation; 
}

void 
Environment:: 
UpdateActionBuffer(Eigen::VectorXd action)
{
	history_buffer_action.push_back(action); 
}


void 
Environment:: 
UpdateTorqueBuffer(Eigen::VectorXd torque)
{
	history_buffer_torque.push_back(torque); 
}



void 
Environment:: 
UpdateHumanActionBuffer(Eigen::VectorXd humanaction)
{
	history_buffer_human_action.push_back(humanaction); 
}


// Eigen::VectorXd 
// Environment::  
// GetCOPRelative()
// {
// 	//use environment mEnv to calculate COP_error
// 	Eigen::Vector3d pos_foot_r = mCharacter->GetSkeleton()->getBodyNode("r_foot")->getCOM();
// 	Eigen::Vector3d pos_foot_l = mCharacter->GetSkeleton()->getBodyNode("l_foot")->getCOM();
// 	pos_foot_l(1) = -0.895985;
// 	pos_foot_r(1) = -0.895985;
// 	Eigen::Vector3d COP_target_left = pos_foot_l;
// 	Eigen::Vector3d COP_target_right = pos_foot_r;

// 	auto& results = mWorld->getConstraintSolver()->getLastCollisionResult();
//     std::vector<constraint::ContactConstraintPtr> mContactConstraints;

// 	double COP_Y_fixed_left = COP_target_left(1);
// 	double COP_Y_fixed_right = COP_target_right(1);

// 	// std::cout << "COP_Y_fixed  " << COP_Y_fixed << std::endl;

// 	std::vector<Eigen::Vector3d> all_pos_left;
// 	std::vector<Eigen::Vector3d> all_pos_right;
// 	std::vector<Eigen::Vector3d> all_force_left;
// 	std::vector<Eigen::Vector3d> all_force_right;

//     Eigen::Vector3d COP_left, COP_right; 

// 	for(int i = 0; i < results.getNumContacts(); ++i)   // store all contact forces 
// 	{
// 		auto& contact = results.getContact(i);
// 		mContactConstraints.clear();
// 		mContactConstraints.push_back(
// 				std::make_shared<constraint::ContactConstraint>(contact, mWorld->getTimeStep()));
// 		auto pos = contact.point;
// 		auto force = contact.force;
// 		// all_pos.push_back(pos);
// 		// all_force.push_back(force);
// 		auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(
// 			contact.collisionObject1->getShapeFrame());
// 		auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(
// 			contact.collisionObject2->getShapeFrame());
// 	DART_SUPPRESS_DEPRECATED_BEGIN
// 		auto body1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
// 		auto body2 = shapeFrame2->asShapeNode()->getBodyNodePtr();
// 	DART_SUPPRESS_DEPRECATED_END

		
// 		for (auto& contactConstraint : mContactConstraints)
// 		{
// 			if(body1->getName() == "l_foot_ground")
// 			{
// 				all_pos_left.push_back(pos);
// 				all_force_left.push_back(force);
// 			}
// 			else if(body1->getName() == "r_foot_ground"){
// 				all_pos_right.push_back(pos);
// 				all_force_right.push_back(force);
// 			}
// 			else
// 			{
// 				// std::cout << body1->getName() << std::endl;
// 				// std::cout << "-----Warning: contact force not on foot-------" << std::endl;
// 			}
// 		}
// 	}

// 	Eigen::Vector3d p_cross_f_left;
// 	double f_sum_left = 0; 
// 	p_cross_f_left.setZero();
// 	Eigen::Vector3d p;

// 	Eigen::Vector3d unitV;
// 	unitV << 0, 1, 0;    // unit normal vector  

// 	for(int i=0; i<all_pos_left.size(); i++){
// 		p = all_pos_left[i];
// 		double f_scalar_left = all_force_left[i].dot(unitV);
// 		f_sum_left += f_scalar_left; 
// 		p_cross_f_left += p.cross(f_scalar_left * unitV);
// 	}
// 	if (f_sum_left==0)
// 		COP_left.setZero();
// 	else
// 	{
// 		COP_left = -p_cross_f_left.cross(unitV)/f_sum_left;
// 		COP_left(1) = COP_target_left(1);
// 	}
   
// 	//
// 	Eigen::Vector3d p_cross_f_right;
// 	double f_sum_right = 0; 
// 	p_cross_f_right.setZero();
// 	for(int i=0; i<all_pos_right.size(); i++){
// 		p = all_pos_right[i];
// 		double f_scalar_right = all_force_right[i].dot(unitV);
// 		f_sum_right += f_scalar_right; 
// 		p_cross_f_right += p.cross(f_scalar_right * unitV);
// 	}
//     if (f_sum_right==0)
// 		COP_right.setZero();
// 	else
// 	{
// 		COP_right = -p_cross_f_right.cross(unitV)/f_sum_right;
// 		COP_right(1) = COP_target_right(1);
// 	}

// 	Eigen::Vector3d COP_left_rel = COP_target_left - COP_left;
// 	Eigen::Vector3d COP_right_rel = COP_target_right - COP_right;
// 	Eigen::VectorXd COP_rel(COP_left_rel.rows()+COP_right_rel.rows()); 
//     COP_rel << 0.2*COP_left_rel, 0.2*COP_right_rel;
//     return COP_rel;

// }

Eigen::VectorXd 
Environment::  
GetFullObservation()
{
	// Eigen::VectorXd tar_poses = this->GetTargetObservations();
	// --------- flatten state history 
	Eigen::MatrixXd states(mNumState, HISTORY_BUFFER_LEN);
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
		states.col(i) =  history_buffer_control_state.get(i);
	Eigen::VectorXd states_v = Eigen::Map<const Eigen::VectorXd>(states.data(), states.size());
	
	// --------- flatten action history 
	Eigen::MatrixXd actions(mNumExoActiveDof-3, HISTORY_BUFFER_LEN);
	for(int i=0; i<HISTORY_BUFFER_LEN; i++)
		actions.col(i) =  history_buffer_action.get(i);
	// matrix to vector
	Eigen::VectorXd actions_v = Eigen::Map<const Eigen::VectorXd>(actions.data(), actions.size());

	
	//get human state 
	Eigen::VectorXd humanstates_v = GetHumanState(); 

	Eigen::VectorXd observation;
	if (mUseExo)
		{
			observation.resize(states_v.rows()+actions_v.rows()+humanstates_v.rows());
			observation << states_v, actions_v, humanstates_v;
		}
	else
		{
			observation.resize(humanstates_v.rows());
			observation << humanstates_v;
		}
	return observation;
}


void 
Environment::
SetAction(const Eigen::VectorXd& a)           // execute the env.SecAction() in the GenerateTransitions process
{
	mPrevExoAction = mCurrentExoAction; 
	mCurrentExoAction = a*0.1; 
    // std::cout <<" -----mCurrentExoAction-----" << a << std::endl;
	double t = mWorld->getTime();
	// std::cout << "time:--------" << t <<std::endl;
	// std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	// mTargetPositions.head(mCharacter->GetExodof()) = std::get<0>(pv);
	// // std::cout <<  "mTargetPositions:\n" << mTargetPositions << std::endl;
	// mTargetVelocities.head(mCharacter->GetExodof()) = std::get<1>(pv);
	// mTargetEE_pos = std::get<2>(pv);
	// mSimCount = 0;
	// mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	// mAverageActivationLevels.setZero();
}


void 
Environment::
SetHumanAction(const Eigen::VectorXd& a)           // execute the env.SecAction() in the GenerateTransitions process
{
	mPrevHumanAction = mCurrentHumanAction; 
	mCurrentHumanAction = a*0.1; 
	// std::cout <<" -----mCurrentHumanAction-----" <<  mCurrentHumanAction << std::endl;
	double t = mWorld->getTime();
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = std::get<0>(pv);
	mTargetVelocities = std::get<1>(pv);
	mTargetEE_pos = std::get<2>(pv);

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();
}


void 
Environment::
SetActivationLevels(const Eigen::VectorXd& a)
{
	mActivationLevels = a*1;
	// std::cout << "before mActivationLevels:  " << mActivationLevels << std::endl;
	// mActivationLevels = clamp(mActivationLevels, 0.0, 0.5);  //clamp torques
	// std::cout << "mActivationLevels:  " << mActivationLevels(10) << std::endl;
}


double
Environment::
GetReward()
{
   	Eigen::VectorXd torque_diff = (history_buffer_torque.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_torque.get(HISTORY_BUFFER_LEN-2)+history_buffer_torque.get(HISTORY_BUFFER_LEN-3)); 
	Eigen::VectorXd torque_diff_exo =  torque_diff.tail(mNumExoActiveDof); 
	double r_torque_smooth_exo = exp_of_squared(torque_diff_exo,30.0);
 	
 	Eigen::VectorXd action_diff = (history_buffer_action.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_action.get(HISTORY_BUFFER_LEN-2)+history_buffer_action.get(HISTORY_BUFFER_LEN-3)); 
    double r_action_smooth_exo = exp_of_squared(action_diff,30.0);

	Eigen::VectorXd human_torque = GetDesiredTorques().head(mNumHumanActiveDof);
 	Eigen::VectorXd hip_torque = Eigen::VectorXd::Zero(2);
	hip_torque << human_torque(0), human_torque(9);
	double r_torque = exp_of_squared(hip_torque,0.0001);  // train

	double r = 0.8*GetHumanReward() + 0.2*r_torque + 0.2*r_torque_smooth_exo;
	return r; 
}

double
Environment::
GetHumanReward()
{	
	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd cur_pos = skel->getPositions();
	Eigen::VectorXd cur_vel = skel->getVelocities();
	// if (dart::math::isNan(cur_pos))
	// {
	// 	std::cout << "---humanreward -------------   "  << std::endl;
	// }

	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);
    // std::cout << "p_diff_all \n" << p_diff_all << std::endl;
	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
    
	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();

	for(auto ss : bvh_map)
	{
		auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
	}

   	// std::map<std::string, Eigen::Vector3d> mEEOffsetMap = mCharacter->GetBVH()->GetEEOffsetMap();
	
    int cnt = 0; 
    BodyNode* root = mCharacter->GetSkeleton()->getRootBodyNode();
		
    auto ees = mCharacter->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);

	Eigen::VectorXd com_diff;
    Eigen::Vector3d root_rotation = p_diff_all.segment<3>(0);
	Eigen::Vector2d root_XZ_position;
	root_XZ_position << p_diff_all[3],p_diff_all[5];
	Eigen::VectorXd root_diff = Eigen::VectorXd::Zero(5);
	root_diff <<root_rotation,root_XZ_position;
	
	// calucuate end_effector error
	for(int i =0;i<ees.size();i++)
	{
		ee_diff.segment<3>(i*3) = ees[i]->getCOM(); 
		// ee_diff[i*3+2] = 0;
	} 

	com_diff = skel->getCOM();

	skel->setPositions(const_index_human, mTargetPositions.head(mCharacter->GetHumandof()));
	skel->computeForwardKinematics(true,false,false);

      
	com_diff -= skel->getCOM();
	for(int i=0;i<ees.size();i++)  
	{
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;  // the error of the skeletion's COM (between the current and target）
		// ee_diff[i*3+2] = 0;
	}
	skel->setPositions(const_index_human, cur_pos.head(mCharacter->GetHumandof()));
	skel->computeForwardKinematics(true,false,false);
	/*
	// !!!!!!!!!!!!! Bug ~!!!!!!!!
	//  COP_reward and Zero_COM_momentum reward from RewardFactory
	// double r_COP_ZCM_torque = 0;
	// for(auto reward: mReward){
	// 	r_COP_ZCM_torque += reward.second->GetReward()*reward.second->GetWeight();  
	// }
    // std::cout << "p_diff \n" << p_diff << std::endl;
	*/
	Eigen::VectorXd torque_diff = (history_buffer_torque.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_torque.get(HISTORY_BUFFER_LEN-2)+history_buffer_torque.get(HISTORY_BUFFER_LEN-3)); 
	Eigen::VectorXd torque_d_human = Eigen::VectorXd::Zero(2);
	torque_d_human << torque_diff(0), torque_diff(9);
	double r_torque_smooth = exp_of_squared(torque_d_human,15);
	double r_q = exp_of_squared(p_diff,1.0);
	double r_v = exp_of_squared(v_diff,0.1);
	double r_ee = exp_of_squared(ee_diff,5.0);
	double r_com = exp_of_squared(com_diff,10.0);
	double r_root = exp_of_squared(root_diff,10);
    double r = 0;
	
	Eigen::VectorXd action_diff = (history_buffer_action.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_action.get(HISTORY_BUFFER_LEN-2)+history_buffer_action.get(HISTORY_BUFFER_LEN-3)); 
    double r_action_smooth = exp_of_squared(action_diff,20.0);
	r = r_ee*(w_q*r_q + w_v*r_v)  +0.25*r_torque_smooth;	
	return r; 
}



std::tuple<double,double,double,double,double,double,Eigen::VectorXd,Eigen::VectorXd,double,double,double,double,double,double>
Environment::
GetRenderReward_Error()
{
	auto& skel = mCharacter->GetSkeleton();
	Eigen::VectorXd cur_pos = skel->getPositions();
	Eigen::VectorXd cur_vel = skel->getVelocities();


 	int idx_hip_l_exo = mCharacter->GetSkeleton()->getJoint("exo_hip_flexion_l")->getIndexInSkeleton(0);
    int idx_hip_r_exo = mCharacter->GetSkeleton()->getJoint("exo_hip_flexion_r")->getIndexInSkeleton(0);

 	int idx_hip_l_human = mCharacter->GetSkeleton()->getJoint("FemurL")->getIndexInSkeleton(0);
    int idx_hip_r_human = mCharacter->GetSkeleton()->getJoint("FemurR")->getIndexInSkeleton(0);

	// std::cout <<  "idx---" << idx_hip_l_human <<  "idx---" << idx_hip_r_human << std::endl;   

	double p_exo_hip_l = cur_pos[idx_hip_l_exo];
	double p_exo_hip_r = cur_pos[idx_hip_r_exo];
	double p_exo_hip_l_vel = cur_vel[idx_hip_l_exo];
	double p_exo_hip_r_vel = cur_vel[idx_hip_r_exo];


	double p_human_hip_l = cur_pos[idx_hip_l_human];
	double p_human_hip_r = cur_pos[idx_hip_r_human];





	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);
    // std::cout << "p_diff_all \n" << p_diff_all << std::endl;
	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
    
	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();

	for(auto ss : bvh_map)
	{
		auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
		int idx = joint->getIndexInSkeleton(0);
		if(joint->getType()=="FreeJoint")
			continue;
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
	}

    int cnt = 0; 
    BodyNode* root = mCharacter->GetSkeleton()->getRootBodyNode();
	
  	auto ees = mCharacter->GetEndEffectors();
	Eigen::VectorXd ee_diff(ees.size()*3);

	Eigen::VectorXd com_diff;
    Eigen::Vector3d root_rotation = p_diff_all.segment<3>(0);
	Eigen::Vector2d root_XZ_position;
	root_XZ_position << p_diff_all[3],p_diff_all[5];
	Eigen::VectorXd root_diff = Eigen::VectorXd::Zero(4);
	root_diff <<root_rotation,root_XZ_position;
    
	// calucuate end_effector error
	for(int i =0;i<ees.size();i++)
	{
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();  
	    // std::cout << "ee_pos:\n" <<  ees[i]->getCOM() << std::endl;
		
	}
	com_diff = skel->getCOM();

	skel->setPositions(mTargetPositions);
	skel->computeForwardKinematics(true,false,false);

	com_diff -= skel->getCOM();                     // the error of the skeletion's COM (between the current and target)
	
	// for(int i=0;i<ees.size();i++)
	// 	std::cout << "ee_tar:\n" <<  ees[i]-> getCOM() << std::endl;

	for(int i=0;i<ees.size();i++)
	{
		ee_diff.segment<3>(i*3) -= ees[i]-> getCOM();  
		// std::cout << "ee_diff:\n" <<  ee_diff.segment<3>(i*3) << std::endl;
	}

	skel->setPositions(cur_pos);
	skel->computeForwardKinematics(true,false,false);

	//  COP_reward and Zero_COM_momentum reward from RewardFactory
	double r_COP_ZCM_torque = 0;
	// for(auto reward: mReward){
	// 	r_COP_ZCM_torque += reward.second->GetReward()*reward.second->GetWeight();  
	// 	// std::cout << reward.first << "   Reward Weight:   " << reward.second->GetWeight() << std::endl;     
	// }

    // double cop_left_reward = mReward["left_cop"]->GetReward();
    // double cop_right_reward  = mReward["right_cop"]->GetReward();
	// double torque_reward  = mReward["mimum_torque"]->GetReward();
	// double ZCM_reward = mReward["zero_com_momentum"]->GetReward();

	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,0.1);
	double r_ee = exp_of_squared(ee_diff,5.0);
	double r_com = exp_of_squared(com_diff,10.0);

	double r_root = exp_of_squared(root_diff,10);

    Eigen::VectorXd human_torque = GetDesiredTorques().head(mNumHumanActiveDof);
	Eigen::VectorXd e_torque = GetDesiredTorques().tail(mNumExoActiveDof);
	Eigen::VectorXd exo_torque = Eigen::VectorXd::Zero(2); 
	exo_torque[0] = e_torque[2];
	exo_torque[1]=  e_torque[4];

 	Eigen::VectorXd hip_torque = Eigen::VectorXd::Zero(2);

	hip_torque << human_torque(0), human_torque(9);
	double r_torque = exp_of_squared(hip_torque,0.0005);  // train
	

	Eigen::VectorXd torque_diff = (history_buffer_torque.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_torque.get(HISTORY_BUFFER_LEN-2)+history_buffer_torque.get(HISTORY_BUFFER_LEN-3)); 
	Eigen::VectorXd torque_diff_exo =  torque_diff.tail(mCharacter->GetExodof()); 
	double r_torque_smooth = exp_of_squared(torque_diff_exo,60.0);
	Eigen::VectorXd action_diff = (history_buffer_action.get(HISTORY_BUFFER_LEN-1)-2*history_buffer_action.get(HISTORY_BUFFER_LEN-2)+history_buffer_action.get(HISTORY_BUFFER_LEN-3)); 
    double r_action_smooth = exp_of_squared(action_diff,10.0);
	
	return std::make_tuple(r_q, r_v, r_ee, r_root, r_torque, r_action_smooth, human_torque, exo_torque, p_exo_hip_l,  p_exo_hip_r, p_exo_hip_l_vel, p_exo_hip_r_vel, p_human_hip_l, p_human_hip_r);

}