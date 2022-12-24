#include "Window.h"
#include "Environment.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <cmath>
#include "matplotlibcpp.h"
#include "Force.h"
#include "BodyForce.h"
namespace plt = matplotlibcpp;
#include <fstream>
#include <iterator>


using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawCollision(true),mDrawContact(true), mDrawground(true), mDrawShape(true), mDrawMuscles(true),
	mDrawSpringforce(false), mDrawBushingforce(false), mDrawEndEffectors(false), mDrawEndEffectorTargets(false), mDrawShadow(true),mMuscleNNLoaded(false),mHumanNNLoaded(false),mDrawFigure(false),mDrawCompositionforces(true)
{
	std::cout << "seed 333333" << std::endl;
	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.25;	
	mFocus = true;
	mNNLoaded = false;
	if(mEnv->GetUseSymmetry())
		for(int i; i<mEnv->GetNumAction()/2; i++)
			torque_vectors.push_back(new std::vector<double>());
	else
	    for(int i; i<mEnv->GetNumAction(); i++)
			torque_vectors.push_back(new std::vector<double>());
	std::cout << "seed 44444" << mEnv->GetNumAction() << std::endl;
	if (mEnv->GetUseHumanNN())
	{
		for(int i; i<mEnv->GetNumHumanAction(); i++)
				human_torque_vectors.push_back(new std::vector<double>());
	}
	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");
	t = 0; 	
	p::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();

	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *",mns);
	std::cout << "seed 555555" << std::endl;
}
Window::
Window(Environment* env,const std::string& nn_path)
	:Window(env)
{
	mNNLoaded = true;
	boost::python::str str;
    if(mEnv->GetUseHumanNN())
		str = ("num_state = "+std::to_string(mEnv->GetNumFullObservation()-mEnv->GetNumHumanObservation())).c_str();
	else
		str = ("num_state = "+std::to_string(mEnv->GetNumFullObservation())).c_str();
	p::exec(str,mns);
	if(mEnv->GetUseSymmetry())
		str = ("num_action = "+std::to_string(mEnv->GetNumAction()/2)).c_str();
	else
		str = ("num_action = "+std::to_string(mEnv->GetNumAction())).c_str();

	p::exec(str,mns);
	nn_module = p::eval("SimulationNN(num_state,num_action)",mns);
	p::object load = nn_module.attr("load");
	load(nn_path);
	std::cout << "here load" << std::endl;
}

Window::
Window(Environment* env,const std::string& nn_path,const std::string& human_muscle_nn_path)
	:Window(env,nn_path)
{
	if (human_muscle_nn_path.find("human") != std::string::npos  )
		mHumanNNLoaded = true;
	else if (human_muscle_nn_path.find("muscle") != std::string::npos  )
		mMuscleNNLoaded = true;
	boost::python::str str;
    if (mHumanNNLoaded)
	{
		str = ("num_state = "+std::to_string(mEnv->GetNumHumanObservation())).c_str();
		p::exec(str,mns);
		str = ("num_action = "+std::to_string(mEnv->GetNumHumanAction())).c_str();
		p::exec(str,mns);
		human_nn_module = p::eval("SimulationHumanNN(num_state,num_action)",mns);
		p::object load = human_nn_module.attr("load");
		load(human_muscle_nn_path);
		std::cout << "here human loaded" << std::endl;
	}
	else if (mMuscleNNLoaded)
	{
		str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetNumTotalRelatedDofs())).c_str();
		p::exec(str,mns);
		str = ("num_actions = "+std::to_string(mEnv->GetNumHumanAction())).c_str();
		p::exec(str,mns);
		str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetMuscles().size())).c_str();
		p::exec(str,mns);
		muscle_nn_module = p::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);
		p::object load = muscle_nn_module.attr("load");
		load(human_muscle_nn_path);
		std::cout << "here muscle loaded" << std::endl;
	}
}


Window::
Window(Environment* env,const std::string& nn_path,const std::string& human_nn_path, const std::string& muscle_nn_path)
	:Window(env,nn_path)
{
	mHumanNNLoaded = true;
	mMuscleNNLoaded = true;
	boost::python::str str;
    if (mHumanNNLoaded)
	{
		str = ("num_state = "+std::to_string(mEnv->GetNumHumanObservation())).c_str();
		p::exec(str,mns);
		str = ("num_action = "+std::to_string(mEnv->GetNumHumanAction())).c_str();
		p::exec(str,mns);
		human_nn_module = p::eval("SimulationHumanNN(num_state,num_action)",mns);
		p::object load = human_nn_module.attr("load");
		load(human_nn_path);
		std::cout << "here human loaded" << std::endl;
	}
	if (mMuscleNNLoaded)
	{
		str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetNumTotalRelatedDofs())).c_str();
		p::exec(str,mns);
		str = ("num_actions = "+std::to_string(mEnv->GetNumHumanAction())).c_str();
		p::exec(str,mns);
		str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetActiveMuscleNum())).c_str();
		p::exec(str,mns);
		muscle_nn_module = p::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);
		p::object load = muscle_nn_module.attr("load");
		load(muscle_nn_path);
		std::cout << "here muscle loaded" << std::endl;
	}


	std::map<std::string,double> musclelimitforvisualmap = mEnv->GetCharacter()->GetMusclelimitforVisualmap();
	std::map<std::string,std::vector<std::string>> relatedmusclemap =  mEnv->GetCharacter()->GetRelatedMuscleMap();
	std::vector<std::string> one_muscle_to_plot; 
	std::vector<std::string> related_muscle_to_plot; 

	for(auto ss : mEnv->GetCharacter()->GetRelatedMuscleMap())
	{
		if (ss.first.find("L_")!=std::string::npos)
		{
			if((ss.second.size()>=2) && (musclelimitforvisualmap[ss.first] >= mEnv->GetCharacter()->GetMuscleminlimitforVisual()))
			{
				related_muscle_to_plot.push_back(ss.first);
				// std::cout << ss.first << std::endl;
			}

			else
			{
				if (musclelimitforvisualmap[ss.first] >= mEnv->GetCharacter()->GetMusclemaxlimitforVisual())
				{
					one_muscle_to_plot.push_back(ss.first);
					// std::cout << ss.first << std::endl;
				}
			}
		}
	}
   
	if (mEnv->GetPlotRelatedMuscle())
	{
		std::cout << "----------------------------" << std::endl;
		one_muscle_to_plot.insert(one_muscle_to_plot.end(),related_muscle_to_plot.begin(),related_muscle_to_plot.end());
	}

	std::vector<std::string> muscle_name_to_plot = one_muscle_to_plot;
	std::vector<std::string> colors { "blue", "red", "cyan", "green", "black", "magenta"};
	for (int i = 0; i < muscle_name_to_plot.size(); i++){
		std::vector<double> vect;
		muscle_plot.insert ( std::pair<std::string, std::vector<double>>(muscle_name_to_plot[i],  vect) );
		muscle_plot_legend.insert ( std::pair<std::string, std::map<std::string, std::string>>( muscle_name_to_plot[i], 
																								{{"color",colors[i%6]}, {"linewidth","1"},{"label",muscle_name_to_plot[i]}}) );
	}
	std::cout << "----------------------------" << std::endl;
}


void
Window::
draw()
{	
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];
	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	// std::cout << "ground: raw \n" << y << std::endl;

	auto& ctresults = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();

	DrawGround(y);
	// PlotFigure();
	if(mDrawContact) DrawContactForces(ctresults);
	// if(mDrawEndEffectors) DrawEndEffectors();
    // DrawExternalforce();
	if(mDrawMuscles) DrawMuscles(mEnv->GetCharacter()->GetMuscles());
	DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());
	// DrawSkeleton(mEnv->GetGround());
	// if(mDrawSpringforce)DrawSpringforce();
	if(mDrawBushingforce)DrawBushingforce();
	// DrawJointConstraint();
	SetFocusing();
}
void
Window::
keyboard(unsigned char _key, int _x, int _y)
{
	switch (_key)
	{
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'r': this->Reset();break;
	case ' ': mSimulating = !mSimulating;break;
	case 'm': mDrawMuscles = !mDrawMuscles; break;
	case 'd': mDrawSpringforce =!mDrawSpringforce; break;
	case 'b': mDrawBushingforce =!mDrawBushingforce; break;
	case 'p': mDrawShape = !mDrawShape; break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 'n': mDrawCollision = !mDrawCollision;break;    // add collision shape key
	case 'c': mDrawContact = !mDrawContact; break;
	case 'e': mDrawEndEffectors = !mDrawEndEffectors; break;
	case 't': mDrawEndEffectorTargets = !mDrawEndEffectorTargets; break;
	case 'k': mDrawFigure = !mDrawFigure; break;
	case 'w': mDrawCompositionforces = !mDrawCompositionforces; break;
	case 'g': mDrawground =!mDrawground; break;
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}
void
Window::
displayTimer(int _val)
{
	if(mSimulating)
		Step();
	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

void
Window::
Step()
{	
	if (!mEnv->GetUsetargetvisual())
	{ 
		plt::ion();
		int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
		Eigen::VectorXd action;
		Eigen::VectorXd action_human;
		if(mNNLoaded)
			action = GetActionFromNN();
		else
			action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
		if(mHumanNNLoaded)
			action_human = GetActionFromHumanNN();
		else
			action_human = Eigen::VectorXd::Zero(mEnv->GetNumHumanAction());
	

		Eigen::VectorXd action_full = Eigen::VectorXd::Zero(action.rows()); 
		if(mEnv->GetUseSymmetry())
		{
			action_full.resize(action.rows()*2);
			action_full << action, action; 
		}
		else
		{
			action_full << action;
		} 
		// std::cout << "---numaction----" << mEnv->GetNumAction() << std::endl;
		// std::cout << "robot action:\n" << action_full << std::endl;
		// std::cout << "human action:\n" << action_human.rows() << std::endl;
		// std::cout << "human action:\n" << action_human << std::endl;
		mEnv->SetAction(action_full);
		if (mEnv->GetUseHumanNN())
			mEnv->SetHumanAction(action_human);

		mEnv->UpdateActionBuffer(action_full);
		mEnv->UpdateHumanActionBuffer(action_human);

		if (mEnv->GetWalkSkill())
			action_foot_vector.push_back(action_full(3));


		action_hip_l_exo_vector.push_back(0.1*action_full(0));
		action_hip_r_exo_vector.push_back(0.1*action_full(1));



		if(mEnv->GetUseMuscleNN())
		{
			int inference_per_sim = 2;
			for(int i=0;i<num;i+=inference_per_sim){
				Eigen::VectorXd mt = mEnv->GetMuscleTorques();
				mEnv->SetActivationLevels(GetActivationFromNN(mt));
				for(int j=0;j<inference_per_sim;j++)
				{
					mEnv->ProcessAction(j+i, num);
					mEnv->Step();
				}
			}
		}
		else
		{   
			for(int i=0;i<num;i++)
			{   
				// set mAction as the interpolation of PrevAction, Current Action; 
				mEnv->ProcessAction(i, num);
	
				mEnv->Step();
			}	
		}
	double pos0 = mEnv->GetCharacter()->GetSkeleton()->getPositions()[5];
	t = mEnv->GetWorld()->getTime();
	// std::cout << "------time-------" << t << " ----pos-----" << pos0 << std::endl;
	Character* mCharacter = mEnv->GetCharacter();
	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1]-mEnv->GetGround()->getRootBodyNode()->getCOM()[1];
	Eigen::Vector6d root_pos = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
	Eigen::Isometry3d cur_root_inv = mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().inverse();

	Eigen::Vector3d root_v = mCharacter->GetSkeleton()->getBodyNode(0)->getCOMLinearVelocity();
	double root_v_norm = root_v.norm();
	// Eigen::Vector3d foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot")->getWorldTransform().translation();
	// Eigen::Vector3d foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot")->getWorldTransform().translation();
	// double pos_foot_l =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
	// double pos_foot_r =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);
	// // std::cout << "----------" << pos_foot_l << "  " << pos_foot_r << std::endl;
	// foot_left_Forward_vector.push_back(foot_l(0));
	// foot_left_Height_vector.push_back(foot_l(1));

	// foot_right_Forward_vector.push_back(foot_r(0));
	// foot_right_Height_vector.push_back(foot_r(1));
	Eigen::VectorXd p_cur_human = mCharacter->GetSkeleton()->getPositions().tail(mCharacter->GetHumandof());
	double hip_joint_angle = p_cur_human[15];
	hip_joint_angle_vector.push_back(hip_joint_angle);
	//////////////////////plot the error 
	double plot_time = mEnv->GetWorld()->getTime();
	Eigen::Vector3d skel_COM = mCharacter->GetSkeleton()->getCOM();
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	double hip_force =0;
	double femur_force_l1=0;
	double femur_force_l2=0;
	double femur_force_r1=0;
	double femur_force_r2=0;
	double tibia_force_l1=0;
	double tibia_force_l2=0;
	double tibia_force_r1=0;
	double tibia_force_r2=0;			
	double hip_torque =0;
	double femur_torque_l1=0;
	double femur_torque_l2=0;
	double femur_torque_r1=0;
	double femur_torque_r2=0;
	double tibia_torque_l1=0;
	double tibia_torque_l2=0;
	double tibia_torque_r1=0;
	double tibia_torque_r2=0;			
	// for(int i=0; i<forces.size(); ++i)
	// {
	// 	auto& _force = forces[i];
	// 	if (_force->GetName().find("bushingforce")!=std::string::npos)
	// 	{
	// 		if (_force->GetName().find("hip")!=std::string::npos)
	// 		{
	// 			hip_force = _force->GetForce().norm();
	// 			hip_torque = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
	// 		{
	// 			femur_force_l1 = _force->GetForce().norm();
	// 			femur_torque_l1 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
	// 		{
	// 			femur_force_l2 = _force->GetForce().norm();
	// 			femur_torque_l2 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
	// 		{
	// 			femur_force_r1 = _force->GetForce().norm();
	// 			femur_torque_r1 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
	// 		{
	// 			femur_force_r2 = _force->GetForce().norm();
	// 			femur_torque_r2 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
	// 		{
	// 			tibia_force_l1 = _force->GetForce().norm();
	// 			tibia_torque_l1 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
	// 		{
	// 			tibia_force_l2 = _force->GetForce().norm();
	// 			tibia_torque_l2 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
	// 		{
	// 			tibia_force_r1 = _force->GetForce().norm();
	// 			tibia_torque_r2 = _force->GetTorque().norm();
	// 		}
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
	// 		{
	// 			tibia_force_r2 = _force->GetForce().norm();
	// 			tibia_torque_r2 = _force->GetTorque().norm();
	// 		}
	// 	}
		
	// 	if (_force->GetName().find("springforce")!=std::string::npos)
	// 	{
	// 		if (_force->GetName().find("hip")!=std::string::npos)
	// 				hip_force = _force->GetForce().norm();
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
	// 				femur_force_l1 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
	// 				femur_force_l2 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
	// 				femur_force_r1 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
	// 				femur_force_r2 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
	// 				tibia_force_l1 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
	// 				tibia_force_l2 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
	// 				tibia_force_r1 = _force->GetForce().norm();
	// 		if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
	// 				tibia_force_r2 = _force->GetForce().norm();
	// 	}
	// }

	// hip_force_vector.push_back(hip_force);
	// hip_torque_vector.push_back(hip_torque);
	// double femur_force_l = sqrt(pow(femur_force_l1,2) +pow(femur_force_l2,2));
	// double femur_force_r = sqrt(pow(femur_force_r1,2) +pow(femur_force_r2,2));
	// double tibia_force_l = sqrt(pow(tibia_force_l1,2) +pow(tibia_force_l2,2));
	// double tibia_force_r = sqrt(pow(tibia_force_r1,2) +pow(tibia_force_r2,2));
	// double femur_torque_l = sqrt(pow(femur_torque_l1,2) +pow(femur_torque_l2,2));
	// double femur_torque_r = sqrt(pow(femur_torque_r1,2) +pow(femur_torque_r2,2));
	// double tibia_torque_l = sqrt(pow(tibia_torque_l1,2) +pow(tibia_torque_l2,2));
	// double tibia_torque_r = sqrt(pow(tibia_torque_r1,2) +pow(tibia_torque_r2,2));
	
	// femur_force_vector_l.push_back(femur_force_l);
	// femur_force_vector_r.push_back(femur_force_r);
	// tibia_force_vector_l.push_back(tibia_force_l);
	// tibia_force_vector_r.push_back(tibia_force_r);
	
	// femur_torque_vector_l.push_back(femur_torque_l);
	// femur_torque_vector_r.push_back(femur_torque_r);
	// tibia_torque_vector_l.push_back(tibia_torque_l);
	// tibia_torque_vector_r.push_back(tibia_torque_r);

	// double r = mEnv->GetHumanReward();
	// double r1= mEnv->GetReward();

	// // Eigen::VectorXd muscle_activation = mEnv->GetActivationLevels();

	// // muscle_activation_vector.push_back(muscle_activation(100)); 
	std::tuple<double,double,double,double,double,double,Eigen::VectorXd,Eigen::VectorXd,double,double,double,double,double,double> tmp = mEnv->GetRenderReward_Error();
	double pos_reward = std::get<0>(tmp);
	double vel_reward = std::get<1>(tmp);
	double ee_reward = std::get<2>(tmp);
	double root_reward = std::get<3>(tmp);
	double torque_reward = std::get<4>(tmp);
	double exo_torque_smooth_reward = std::get<5>(tmp);


	Eigen::VectorXd human_torque =  std::get<6>(tmp);
	
	Eigen::VectorXd exo_torque =  std::get<7>(tmp);


	double p_exo_hip_l = std::get<8>(tmp);
	double p_exo_hip_r = std::get<9>(tmp);
	double p_exo_hip_l_vel = std::get<10>(tmp);
	double p_exo_hip_r_vel = std::get<11>(tmp);

	double p_human_hip_l = std::get<12>(tmp);
	double p_human_hip_r = std::get<13>(tmp);


	// double cop_left_error = std::get<15>(tmp);
	// double cop_right_error = std::get<16>(tmp);

	pos_reward_vector.push_back(pos_reward); 
	vel_reward_vector.push_back(vel_reward); 
	ee_reward_vector.push_back(ee_reward); 
	root_reward_vector.push_back(root_reward); 
	torque_reward_vector.push_back(torque_reward); 
	exo_torque_smooth_reward_vector.push_back(exo_torque_smooth_reward);


	for(int i; i<mEnv->GetNumAction(); i++)
	{
		torque_vectors[i]->push_back(exo_torque[i]); 
	}

	if (mEnv->GetUseHumanNN())
	{
		for(int i; i<mEnv->GetNumHumanAction(); i++)
		  	human_torque_vectors[i]->push_back(-human_torque[i]); 
	}

	hip_l_exo_vector.push_back(p_exo_hip_l*180/3.14);
	hip_r_exo_vector.push_back(p_exo_hip_r*180/3.14);
	hip_l_exo_vector_vel.push_back(p_exo_hip_l_vel*180/3.14);
	hip_r_exo_vector_vel.push_back(p_exo_hip_r_vel*180/3.14);


	hip_l_human_vector.push_back(-p_human_hip_l*180/3.14);
	hip_r_human_vector.push_back(-p_human_hip_r*180/3.14);


	time_vector.push_back(plot_time);
	

	std::map<std::string, std::string> a0 = {{"color","black"}, {"linestyle","--"},{"label","pos_reward"}};
	std::map<std::string, std::string> a1 = {{"color","magenta"}, {"linestyle",":"},{"label","ee_reward"}};
	std::map<std::string, std::string> a2 = {{"color","yellow"}, {"label","torque_reward"}};
	std::map<std::string, std::string> a3 = {{"color","red"},{"marker","+"}, {"label","torque_exo_smooth_reward"}};
	std::map<std::string, std::string> a4 = {{"color","green"}, {"label","CoP_right_reward"}};
	std::map<std::string, std::string> a_sum = {{"color","green"}, {"label","CoP_reward"}};

	std::map<std::string, std::string> a5 = {{"color","black"}, {"label","hip_l_exo"}};
	std::map<std::string, std::string> a6 = {{"color","blue"}, {"label","hip_r_exo"}};
	std::map<std::string, std::string> a7 = {{"color","red"}, {"label","hip_l_human"}};
	std::map<std::string, std::string> a8 = {{"color","magenta"}, {"label","hip_r_human"}};

	std::map<std::string, std::string> a9 = {{"color","blue"}, {"linewidth","1.5"}, {"label","hip"}};
	std::map<std::string, std::string> a10 = {{"color","red"},  {"linewidth","1.5"},{"linestyle","-."}, {"label","knee"}};
	std::map<std::string, std::string> a11 = {{"color","black"},  {"linewidth","1.5"},{"linestyle","--"}, {"label","ankle dorsi/plantar"}};
	std::map<std::string, std::string> a12 = {{"color","cyan"},  {"linewidth","1.5"},{"linestyle","--"}, {"label","ankle inversion/eversion"}};
	std::map<std::string, std::string> a13 = {{"color","magenta"}, {"label","COP_left_error"}};
	std::map<std::string, std::string> a14 = {{"color","yellow"}, {"label","COP_right_error"}};

	std::map<std::string, std::string> a15 = {{"color","black"}, {"linewidth","1"},{"label","hip"}};
	std::map<std::string, std::string> a16 = {{"color","blue"}, {"linestyle","-."}, {"label","knee"}};
	std::map<std::string, std::string> a17 = {{"color","red"}, {"linestyle","--"}, {"label","ankle dorsi/plantar"}};
	std::map<std::string, std::string> a18 = {{"color","magenta"}, {"linestyle","--"}, {"label","ankle inversion/eversion"}};

	std::map<std::string, std::string> a19 = {{"color","white"},{"linewidth","1"},{"label","hip_joint"}};
	std::map<std::string, std::string> a20 = {{"color","blue"}, {"linestyle","-."},{"label","action-knee"}};
	std::map<std::string, std::string> a21 = {{"color","red"}, {"linestyle","--"}, {"label","action-ankle"}};
	std::map<std::string, std::string> a22 = {{"color","magenta"}, {"linestyle","--"}, {"label","action-foot"}};
	

	std::map<std::string, std::string> a23 = {{"color","blue"}, {"linewidth","1.5"},{"label","hip"}};
	std::map<std::string, std::string> a24 = {{"color","red"},{"linewidth","1.5"},{"linestyle","-."},{"label","femur"}};
	std::map<std::string, std::string> a25 = {{"color","red"}, {"linewidth","1.5"},{"linestyle","-."},{"label","femur_force_r"}};
	std::map<std::string, std::string> a26 = {{"color","black"}, {"linewidth","1.5"}, {"linestyle","--"},{"label","tibia"}};
	std::map<std::string, std::string> a27 = {{"color","black"}, {"linewidth","1.5"}, {"linestyle","--"},{"label","tibia_force_r"}};

	std::map<std::string, std::string> a28 = {{"color","red"}, {"linewidth","1"},{"label","skel_COM_XY"}};
	std::map<std::string, std::string> a29 = {{"color","black"}, {"linewidth","1"},{"label","skel_COM_height"}};
	std::map<std::string, std::string> a30 = {{"color","blue"}, {"linewidth","1"},{"label","skel_COM_lateral"}};

	std::map<std::string, std::string> a31 = {{"color","blue"},{"linewidth","1.5"},{"linestyle","-."},{"label","hip_torque"}};
	std::map<std::string, std::string> a32 = {{"color","red"}, {"linewidth","1.5"},{"linestyle","-."},{"label","femur_torque"}};
	std::map<std::string, std::string> a33 = {{"color","black"}, {"linewidth","1.5"},{"label","tibia_torque"}};

	std::map<std::string, std::string> a34 = {{"color","blue"},{"linewidth","1.5"},{"linestyle","-."},{"label","hip_cur_joint_angle"}};
	std::map<std::string, std::string> a_human_hip_joint = {{"color","blue"}, {"linewidth","1"},{"label","hip_joint_human"}};

	std::map<std::string, std::string> a123 = {{"fontsize","10"}};

	// const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
	// std::map<std::string, std::vector<double>>::iterator it;
	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	it->second.push_back(0.0);
	// }
	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	int i=0;
	// 	for(auto muscle : muscles) 
	// 	{
	// 		// std::cout << muscle->GetMuscleUnitName() << std::endl;
	// 		if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
	// 		{
	// 			i=i+1;
	// 			it->second.back() += muscle->GetMuscleUnitactivation();
	// 		}
	// 	}
	// 	it->second.back() = it->second.back()/i;
	// }

	// const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
	// std::map<std::string, std::vector<double>>::iterator it;

	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	it->second.push_back(0.0);
	// }
	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	int i=0;
	// 	for(auto muscle : muscles) 
	// 	{
	// 		// std::cout << muscle->GetMuscleUnitName() << std::endl;
	// 		if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
	// 		{
	// 			i=i+1;
	// 			it->second.back() += muscle->GetMuscleUnitactivation();
	// 		}
	// 	}
	// 	it->second.back() = it->second.back()/i;
	// }

	if (mDrawFigure)
	{ 
		plt::figure(0);
		plt::clf();
		plt::subplot(4,1,1);
		// plt::title("Real-time Reward");
		// plt::xlabel("Time/s");
		plt::ylabel("Reward");
		plt::plot(time_vector,pos_reward_vector,a0);
		plt::plot(time_vector,ee_reward_vector,a1);
		plt::plot(time_vector,torque_reward_vector,a2);
		plt::plot(time_vector,exo_torque_smooth_reward_vector,a3); 
		std::map<std::string, std::string> loc = {{"loc","upper left"}};
		plt::legend(loc);



		plt::subplot(4,1,2);
		plt::title("Joint angle tracking");
		plt::xlabel("Time/s");
		plt::ylabel("Angle tracking");
		plt::plot(time_vector,hip_l_exo_vector,a5);
		plt::plot(time_vector,hip_r_exo_vector,a6);
		plt::plot(time_vector,hip_l_human_vector,a7);
		plt::plot(time_vector,hip_r_human_vector,a8);
		plt::legend(loc);


		plt::subplot(4,1,3);
		// plt::title("Joint torque");
		// plt::xlabel("Time/s");
		plt::ylabel("Joint Torque/N*m");
		// plt::plot(time_vector,*(torque_vectors[0]),a5);
		plt::plot(time_vector,*(torque_vectors[1]), a6);
		// plt::plot(time_vector,*(human_torque_vectors[9]),a7);
		plt::plot(time_vector,*(human_torque_vectors[0]),a8);
		plt::legend(loc);

		plt::subplot(4,1,4);
		plt::xlabel("Time/s");
		plt::ylabel("Action from NN/rad");
		plt::ylabel("joint angular velocity/rad");
		plt::plot(time_vector, action_hip_l_exo_vector,a5);
		plt::plot(time_vector, action_hip_r_exo_vector,a7);

	
		plt::figure(7);
		plt::clf();
		plt::xlabel("Time/s");
		plt::ylabel("joint angular velocity/rad");
		plt::plot(time_vector, 	hip_l_exo_vector_vel,a5);
		plt::plot(time_vector, 	hip_r_exo_vector_vel,a7);
		plt::legend(loc);


		if (mDrawBushingforce)
		{
			plt::figure(1);
			plt::clf();
			// plt::subplot(3,1,1);
			plt::xlabel("Time/s");
			plt::ylabel("Human strap force/N");
			plt::title("Human strap force");
			plt::plot(time_vector,hip_force_vector,a23);
			// plt::legend(loc);
			// plt::subplot(3,1,2);
			plt::plot(time_vector,femur_force_vector_r,a24);
			// plt::plot(time_vector,femur_force_vector_r,a17);
			// plt::legend(loc);
			// plt::subplot(3,1,3);
			plt::plot(time_vector,tibia_force_vector_r,a26);
			// plt::plot(time_vector,tibia_force_vector_l,a19);
			plt::legend(loc);
		 }

			plt::figure(3);
			plt::clf();
			// plt::subplot(3,1,1);
			plt::xlabel("Time/s");
			plt::ylabel("exo torque/N");
			plt::plot(time_vector,*(torque_vectors[0]),a5);
			plt::plot(time_vector,*(torque_vectors[1]), a6);
			plt::legend(loc);


		plt::legend(loc);
		plt::show();
		plt::pause(0.00001); 
	}


	// std::vector<Eigen::Vector3d> all_force;
	// std::vector<Eigen::Vector3d> all_force_left;
	// std::vector<Eigen::Vector3d> all_force_right;
	// Eigen::Vector3d left_force, right_force;
	// std::vector<constraint::ContactConstraintPtr>  mContactConstraints;
	// left_force.setZero(); right_force.setZero(); 
	// auto& results = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();

	// for(int i = 0; i < results.getNumContacts(); ++i) 
	// {
	// 	auto& contact = results.getContact(i);
	// 	mContactConstraints.clear();
	// 	mContactConstraints.push_back(
	// 			std::make_shared<constraint::ContactConstraint>(contact, mEnv->GetWorld()->getTimeStep()));
	// 	auto pos = contact.point;
	// 	auto force = contact.force;

	// 	auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(
	// 		contact.collisionObject1->getShapeFrame());
	// 	auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(
	// 		contact.collisionObject2->getShapeFrame());
	// DART_SUPPRESS_DEPRECATED_BEGIN
	// 	auto body1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
	// 	auto body2 = shapeFrame2->asShapeNode()->getBodyNodePtr();
	// DART_SUPPRESS_DEPRECATED_END

	// 	for (auto& contactConstraint : mContactConstraints)
	// 	{
	// 		if (body1->getName()=="TalusL"||body1->getName()=="FootThumbL"||body1->getName() =="FootPinkyL")
	// 		{
	// 			all_force_left.push_back(force);
	// 		}
	// 		else if(body1->getName()=="TalusR"||body1->getName()=="FootThumbR"||body1->getName() =="FootPinkyR")
	// 		{
	// 			all_force_right.push_back(force);
	// 		}
	// 		else
	// 		{

	// 		}
	// 	}
	// 	all_force.push_back(force);
	// 	for (const auto& contactConstraint : mContactConstraints)
	// 	{
	// 		if (body1->getName()=="TalusL"||body1->getName()=="FootThumbL"||body1->getName() =="FootPinkyL")
	// 		{
	// 			left_force += force;
	// 		}
	// 		else if(body1->getName()=="TalusR"||body1->getName()=="FootThumbR"||body1->getName() =="FootPinkyR")
	// 		{
	// 			right_force += force;
	// 		}
	// 		else
	// 		{
			
	// 		}
	// 	}
        
	// }

	// contact_force_vector_l_forward.push_back(left_force(0));
	// contact_force_vector_l_height.push_back(left_force(1));
	// contact_force_vector_l_lateral.push_back(left_force(2));
	// contact_force_vector_r_forward.push_back(right_force(0));
	// contact_force_vector_r_height.push_back(right_force(1));
	// contact_force_vector_r_lateral.push_back(right_force(2));

	// 	if(plot_time > 9.0){
	// 		std::cout << "plot_time:" << plot_time << std::endl;
	// 		std::vector<std::string> names;
	// 		std::vector<std::vector<double> > values;
	// 		names.push_back("time_vector"); 
	// 		values.push_back(time_vector); 

	// 		names.push_back("hip_l_exo_angle"); 
	// 		values.push_back(hip_l_exo_vector); 
	// 		names.push_back("hip_r_exo_angle"); 
	// 		values.push_back(hip_r_exo_vector);
	// 		names.push_back("hip_l_exo_vel"); 
	// 		values.push_back(hip_l_exo_vector_vel);
	// 		names.push_back("hip_r_exo_vel"); 
	// 		values.push_back(hip_r_exo_vector_vel);

	// 		names.push_back("hip_l_human_angle"); 
	// 		values.push_back(hip_l_human_vector);
	// 		names.push_back("hip_r_human_angle"); 
	// 		values.push_back(hip_r_human_vector); 

	// 		names.push_back("hip_l_exo_torque"); 
	// 		values.push_back(*(torque_vectors[0])); 
	// 		names.push_back("hip_r_exo_torque"); 
	// 		values.push_back(*(torque_vectors[1])); 

	// 		names.push_back("hip_r_human_torque"); 
	// 		values.push_back(*(human_torque_vectors[0])); 
	// 		names.push_back("hip_l_human_torque"); 
	// 		values.push_back(*(human_torque_vectors[9])); 

	// 	for(int i=0; i<names.size();i++){
	// 		std::ofstream output_file("./"+names[i]+".txt");
	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
	// 			double v = *it; 
	// 			output_file << std::to_string(v) << '\n';
	// 		}
	// 	}
	// 	std::exit(EXIT_FAILURE); 
	// }

	networkinput_vector = mEnv->GetFullObservation().head(18);
	
	

	values.push_back(networkinput_vector); 
	


	if(plot_time > 1.6){

		std::time_t now = time(0);
		// Convert now to tm struct for local timezone
		std::tm* localtm = localtime(&now);
		std::cout << values.size() << std::endl;
		std::string currenttime = std::asctime(localtm);
		currenttime.pop_back();

		std::ofstream output_file;
		output_file.open("./result2/"+currenttime+"_"+"networkinput_vector.txt"); // std::ios_base::app);

		for(int i=0; i<values.size();i++){

			std::stringstream ss;
			ss << values[i];
			//std::cout << ss.str()  << std::endl;
			output_file << ss.str() <<"\n" << contact_force_vector_l_height[i] <<"\n" << contact_force_vector_r_height[i] << '\n';
		
		}
		// std::exit(EXIT_FAILURE); 
	}


	// const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
	// std::map<std::string, std::vector<double>>::iterator it;

	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	it->second.push_back(0.0);
	// }
	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// {
	// 	int i=0;
	// 	for(auto muscle : muscles) 
	// 	{
	// 		// std::cout << muscle->GetMuscleUnitName() << std::endl;
	// 		if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
	// 		{
	// 			i=i+1;
	// 			it->second.back() += muscle->GetMuscleUnitactivation();
	// 		}
	// 	}
	// 	it->second.back() = it->second.back()/i;
	// }
	// if(plot_time>22.0)
	// {
	// 	std::vector<std::string> names;
	// 	std::vector<std::vector<double> > values;
	// 	for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
	// 	{
	// 		names.push_back(it->first); 
	// 		values.push_back(it->second); 
	// 	}
	// 	// names.push_back("cop_right_reward"); 
	// 	// values.push_back(cop_right_reward_vector); 
	// 	// std::cout << "-----------------\n"; 
	// 	// std::cout << values[1][0] << std::endl; 
	// 	std::time_t now = time(0);
	// 	// Convert now to tm struct for local timezone
	// 	std::tm* localtm = localtime(&now);

	// 	std::string currenttime = std::asctime(localtm);
	// 	for(int i=0; i<names.size();i++){
	// 		std::cout << names[i] << std::endl; 
	// 		std::ofstream output_file("./result3/"+names[i]+".txt");
	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
	// 			double v = *it; 
	// 			output_file << std::to_string(v) << '\n';
	// 		}
	// 	}
	// 	// std::exit(EXIT_FAILURE); 
	// }
	

	// if(plot_time>11)
	// {
	// 	// std::cout << "safdsafdsafdsafdaf" << std::endl;
	// 	std::vector<std::string> names;
	// 	std::vector<std::vector<double> > values;
	// 	names.push_back("time_vector"); 
	// 	values.push_back(time_vector); 
	// 	// names.push_back("t_vector"); 
	// 	// values.push_back(t_vector); 
	// 	names.push_back("ee_reward"); 
	// 	values.push_back(ee_reward_vector); 
	// 	names.push_back("pos_reward"); 
	// 	values.push_back(pos_reward_vector); 
	// 	names.push_back("cop_total_reward"); 
	// 	values.push_back(cop_total_reward_vector); 
	// 	// names.push_back("cop_right_reward"); 
	// 	// values.push_back(cop_right_reward_vector); 
	// 	// std::cout << "-----------------\n"; 
	// 	// std::cout << values[1][0] << std::endl; 

	// 	std::time_t now = time(0);

	// 	// Convert now to tm struct for local timezone
	// 	std::tm* localtm = localtime(&now);


	//     std::string currenttime = std::asctime(localtm);
	// 	for(int i=0; i<names.size();i++){
	// 		std::cout << names[i] << std::endl; 
	// 		std::ofstream output_file("./result_case4/"+currenttime+"_"+names[i]+".txt");
	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
	// 			double v = *it; 
				
	// 			output_file << std::to_string(v) << '\n';
	// 		}
	// 	}
	// 	std::exit(EXIT_FAILURE); 
	// }
		mEnv->UpdateStateBuffer();
	}
	else
	{
		plt::ion();
		Character* mCharacter = mEnv->GetCharacter();
		// t = mEnv->GetWorld()->getTime();
		// std::tuple<Eigen::VectorXd, Eigen::Vector3d> tmp = mCharacter->GetBVH()->GetMotion(t);
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mEnv->GetControlHz());
		std::cout << "time is " << t << std::endl;
		time_vector.push_back(t);
		auto targetPositions = std::get<0>(pv);
		auto targetVelocities = std::get<1>(pv);
		auto targetEE_pos = std::get<2>(pv);
		mCharacter->targetEE_pos = targetEE_pos; 
		mCharacter->GetSkeleton()->setPositions(targetPositions); // set position
		mCharacter->GetSkeleton()->setVelocities(targetVelocities); //set velocities
		mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
		// double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1]+1.4;
		double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mEnv->GetGround()->getRootBodyNode()->getCOM()[1];
 		// std::cout << "root_y:  "  << root_y << std::endl;
		// std::cout << "foot_l:  "  <<  foot_l << std::endl;
        // std::cout << "foot_r:  "  <<  foot_r << std::endl;
		Eigen::VectorXd com_diff = mCharacter->GetSkeleton()->getCOM();
		Eigen::Vector3d skel_COM = mCharacter->GetSkeleton()->getCOM();
		skel_COM_Forward_vector.push_back(skel_COM(0));
		skel_COM_Height_vector.push_back(skel_COM(1));
		skel_COM_Lateral_vector.push_back(skel_COM(2));
		com_info.push_back(com_diff);
    	auto ees = mCharacter->GetEndEffectors();
		Eigen::VectorXd ee_diff(ees.size()*3);
		// for(int i =0;i<ees.size();i++)
		// {
		// 	ee_diff.segment<3>(i*3) = ees[i]->getCOM();  
		// 	std::cout <<  ees[i]->getCOM() << std::endl;
		// }
		std::map<std::string, std::string> loc = {{"loc","upper left"}};
		std::map<std::string, std::string> a26 = {{"color","red"}, {"linewidth","1"},{"label","skel_COM_XY"}};
		std::map<std::string, std::string> a27 = {{"color","black"}, {"linewidth","1"},{"label","skel_COM_height"}};
		std::map<std::string, std::string> a28 = {{"color","blue"}, {"linewidth","1"},{"label","skel_COM_lateral"}};

		std::map<std::string, std::string> a29 = {{"color","red"}, {"linewidth","1"},{"label","COP_left"}};
		std::map<std::string, std::string> a30 = {{"color","blue"}, {"linewidth","1"},{"label","COP_right"}};

		std::map<std::string, std::string> a31 = {{"color","red"}, {"linewidth","1"},{"label","foot_left"}};
		std::map<std::string, std::string> a32 = {{"color","blue"}, {"linewidth","1"},{"label","foot_right"}};

		t += 1.0/mEnv->GetControlHz(); 
	}

}

void
Window::
DrawExternalforce()
{
	
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	
	double radius = 0.005; //2cm radius 0.005
	double heightN = 5.0e-3;//1mm per N 5.0e-3
	double coneHt = 2.0e-2; //cone height
	// std::cout << forces.size() << std::endl; 
	// std::cout << forces[0]->GetName() << forces[0]->GetForce() << std::endl; 
	// std::cout << forces[1]->GetName() << forces[1]->GetForce() << std::endl; 
	// std::cout << forces[i]->GetName() << std::endl; 
	for(int i = 0; i < forces.size(); ++i) {
		auto& _force = forces[i];
		// _force->Update(); // random Offset
		// _force->UpdatePos();
		if (_force->GetName().find("springforce")!=std::string::npos)
			continue;
		auto pos = _force->GetPos();
		auto force = _force->GetForce();
		// std::cout << "force :" << i << "\n" << force << std::endl;
		// std::cout << "pos  :" << i << "\n" << pos << std::endl;
		Eigen::Vector4d color; 
		color << 0.678431, 0.478431, 0.478431,1.0;  //red

		mRI->setPenColor(color);
		mRI->pushMatrix();
		mRI->translate(pos);
		mRI->drawSphere(radius);
		mRI->popMatrix();

		Eigen::Vector3d pos2 = pos + force * heightN;
		Eigen::Vector3d u(0, 0, 1);
		Eigen::Vector3d v = pos2 - pos;
		Eigen::Vector3d mid = 0.5 * (pos + pos2);
		double len = v.norm();
		v /= len;
		Eigen::Isometry3d T;
		T.setIdentity();
		Eigen::Vector3d axis = u.cross(v);
		axis.normalize();
		double angle = acos(u.dot(v));
		Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
		w_bracket(0, 1) = -axis(2);
		w_bracket(1, 0) = axis(2);
		w_bracket(0, 2) = axis(1);
		w_bracket(2, 0) = -axis(1);
		w_bracket(1, 2) = -axis(0);
		w_bracket(2, 1) = axis(0);

		Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
		T.linear() = R;
		T.translation() = mid;
		mRI->pushMatrix();
		mRI->transform(T);
		mRI->drawCylinder(radius, len);
		mRI->popMatrix();


		T.translation() = pos2;
		mRI->pushMatrix();
		mRI->transform(T);
		mRI->drawCone(2* radius, coneHt);
		mRI->popMatrix();
	}


}

void
Window::
DrawSpringforce()
{
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	for(int i=0; i<forces.size(); ++i)
	{
		auto& _force = forces[i];
		if (_force->GetName().find("springforce")==std::string::npos)
		{
			continue;
		}
		double a = 0;
		Eigen::Vector4d color1(1, 0.0,1.0,1.0);// purple --exoskeleton
		Eigen::Vector4d color2(0.0, 0.0, 1.0, 1.0);//blue --human
		mRI->setPenColor(color1);
		auto aps =_force->GetPoint();
		// for(int i=0;i<aps.size();i++)
		// {
		std::cout << "_forcename:\n" << _force->GetName() << std::endl;
		std::cout << _force->GetForce() << std::endl;
			Eigen::Vector3d p1 = aps[0];
			mRI->pushMatrix();
			mRI->translate(p1);
			mRI->drawSphere(0.01*sqrt(1000/1000.0));
			mRI->popMatrix();
			std::cout << "p1-------:  " << p1 << std::endl;
		// }
		// for(int i=0;i<aps.size();i++)
		// {
			mRI->setPenColor(color2);
			Eigen::Vector3d p2 = aps[1];
			mRI->pushMatrix();
			mRI->translate(p2);
			mRI->drawSphere(0.01*sqrt(1000/1000.0));
			mRI->popMatrix();
			std::cout << "p2-------:  " << p2 << std::endl;
		// }
		// std::cout << _force->GetForce() << std::endl;
		///////////////////////////draw force
	    for(int i=0;i<aps.size()-1;i++)
		{	
			double radius = 0.005; //2cm radius 0.005
			double heightN = 5.0e-3;//1mm per N 5.0e-3
			double coneHt = 2.0e-2; //cone height
			Eigen::Vector3d pos = aps[i]; 
			Eigen::Vector4d force_color; 
			force_color << 0.678431, 0.478431, 0.478431,1.0;  //red
			mRI->setPenColor(force_color);
			mRI->pushMatrix();
			mRI->translate(pos);
			mRI->drawSphere(radius);
			mRI->popMatrix();
			Eigen::Vector3d pos2 = pos + _force->GetForce()* heightN;
			Eigen::Vector3d u(0, 0, 1);
			Eigen::Vector3d v = pos2 - pos;
			Eigen::Vector3d mid = 0.5 * (pos + pos2);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) = axis(2);
			w_bracket(0, 2) = axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) = axis(0);
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(radius, len);
			mRI->popMatrix();
			T.translation() = pos2;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCone(2* radius, coneHt);
			mRI->popMatrix();
		}
		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i];
			Eigen::Vector3d p1 = aps[i+1];
			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);
			
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.01*sqrt(1000/1000.0),len);
			mRI->popMatrix();
		}
		
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}
void
Window::
DrawBushingforce()
{
	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
    // std::cout << "----bushingforcehere------  "  << std::endl;
	for(int i=0; i<forces.size(); ++i)
	{
		auto& _force = forces[i];
		if (_force->GetName().find("bushingforce")==std::string::npos)
		{
			continue;
		}
		double a = 0;
		Eigen::Vector4d color1(1, 0.0,1.0,1.0);// purple --exoskeleton
		Eigen::Vector4d color2(0.0, 0.0, 1.0, 1.0);//blue --human
		
		auto aps =_force->GetPoint();
		// for(int i=0;i<aps.size();i++)
		// {
			mRI->setPenColor(color1);
			Eigen::Vector3d p1 = aps[0];
			mRI->pushMatrix();
			mRI->translate(p1);
			mRI->drawSphere(0.01*sqrt(1000/1000.0));
			mRI->popMatrix();
		// }
		// for(int i=0;i<aps.size();i++)
		// {
			mRI->setPenColor(color2);
			Eigen::Vector3d p2 = aps[1];
			mRI->pushMatrix();
			mRI->translate(p2);
			mRI->drawSphere(0.01*sqrt(1000/1000.0));
			mRI->popMatrix();
		// }
		// std::cout << _force->GetForce() << std::endl;
		///////////////////////////draw force
	    for(int i=0;i<aps.size()-1;i++)
		{	
			double radius = 0.005; //2cm radius 0.005
			double heightN = 5.0e-3;//1mm per N 5.0e-3
			double coneHt = 2.0e-2; //cone height
			Eigen::Vector3d pos = aps[i]; 
			Eigen::Vector4d force_color; 
			force_color << 0.678431, 0.478431, 0.478431,1.0;  //red
			// std::cout << _force->GetForce() << std::endl;
			mRI->setPenColor(force_color);
			mRI->pushMatrix();
			mRI->translate(pos);
			mRI->drawSphere(radius);
			mRI->popMatrix();
			// std::cout << "_forcename:\n" << _force->GetName() << std::endl;
			// std::cout << _force->GetForce() << std::endl;
			Eigen::Vector3d pos2 = pos + _force->GetForce()* heightN;
			Eigen::Vector3d u(0, 0, 1);
			Eigen::Vector3d v = pos2 - pos;
			Eigen::Vector3d mid = 0.5 * (pos + pos2);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) = axis(2);
			w_bracket(0, 2) = axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) = axis(0);
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(radius, len);
			mRI->popMatrix();
			T.translation() = pos2;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCone(2* radius, coneHt);
			mRI->popMatrix();
		
		}
		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i];
			Eigen::Vector3d p1 = aps[i+1];
			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);
			
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.01*sqrt(1000/1000.0),len);
			mRI->popMatrix();
		}
		
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}

void
Window::
DrawJointConstraint()
{
	Eigen::Vector4d color_exo(1.0,0.0,0.0,1.0); ;// purple --exoskeleton
	Eigen::Vector4d color_human(1.0, 1.0, 0.0, 0.0);//blue --human
	
	Eigen::Isometry3d t1 = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("exo_leg_l")->getTransform();
	Eigen::Isometry3d t2 = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getTransform();

	Eigen::Vector3d shift1(0.085,-0.26,0.08);
	Eigen::Vector3d shift2(0.01,0,0.068);
	t1.translation() = t1*shift1;
	t2.translation() = t2*shift2;
	mRI->setPenColor(color_exo);
	Eigen::Vector3d p1 = t1.translation();
	mRI->pushMatrix();
	mRI->translate(p1);
	mRI->drawSphere(0.02*sqrt(1000/1000.0));
	mRI->popMatrix();

	mRI->setPenColor(color_human);
	Eigen::Vector3d p2 = t2.translation();
	mRI->pushMatrix();
	mRI->translate(p2);
	mRI->drawSphere(0.02*sqrt(1000/1000.0));
	mRI->popMatrix();

	Eigen::Isometry3d t3 = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("exo_leg_r")->getTransform();
	Eigen::Isometry3d t4 = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getTransform();

	Eigen::Vector3d shift11(0.085,-0.26,-0.08);
	Eigen::Vector3d shift22(-0.02,0,-0.07);
	t3.translation() = t3*shift11;
	t4.translation() = t4*shift22;
	mRI->setPenColor(color_exo);
	Eigen::Vector3d p3 = t3.translation();
	mRI->pushMatrix();
	mRI->translate(p3);
	mRI->drawSphere(0.02*sqrt(1000/1000.0));
	mRI->popMatrix();

	mRI->setPenColor(color_human);
	Eigen::Vector3d p4 = t4.translation();
	mRI->pushMatrix();
	mRI->translate(p4);
	mRI->drawSphere(0.02*sqrt(1000/1000.0));
	mRI->popMatrix();
}


void
Window::
Reset()
{
	mEnv->Reset();
}
void
Window::
SetFocusing()
{
	if(mFocus)
	{
		//if(mEnv->GetWorld()->getNumSkeletons() == 0) return;
		if(mEnv->GetWorld()->getNumSkeletons() == 1) mTrans = -mEnv->GetWorld()->getSkeleton(0)->getRootBodyNode()->getCOM();
		else if(mEnv->GetWorld()->getNumSkeletons() > 1){
			std::string name = mEnv->GetWorld()->getSkeleton(0)->getName();
			boost::to_lower(name);
			if(name!= "ground") mTrans = -mEnv->GetWorld()->getSkeleton(0)->getRootBodyNode()->getCOM();
			else mTrans = -mEnv->GetWorld()->getSkeleton(1)->getRootBodyNode()->getCOM();
		}

		//mTrans = -mEnv->GetWorld()->getSkeleton("NJIT_BME_EXO_Model")->getRootBodyNode()->getCOM();   // load NJIT_BME_EXO_Model
		//mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();              // load Human MASS model

		mTrans[1] -= 0.3;

		mTrans *=1000.0;
		
	}
}

np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}


Eigen::VectorXd
Window::
GetActionFromNN()
{
	p::object get_action;
	get_action= nn_module.attr("get_action");
	Eigen::VectorXd state;
	if (mEnv->GetUseHumanNN())
		state = mEnv->GetFullObservation().head(mEnv->GetNumFullObservation()-mEnv->GetNumHumanObservation());
	else
		state = mEnv->GetFullObservation();
	p::tuple shape = p::make_tuple(state.rows());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape,dtype);
	
	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];
	p::object temp = get_action(state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetNumAction());
	if(mEnv->GetUseSymmetry()){
		action.resize(mEnv->GetNumAction()/2);
	}
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];
	return action;
}



Eigen::VectorXd
Window::
GetActionFromHumanNN()
{
	p::object get_humanaction;
	get_humanaction= human_nn_module.attr("get_humanaction");
	Eigen::VectorXd state;
	state = mEnv->GetFullObservation().tail(mEnv->GetNumHumanObservation());
	p::tuple shape = p::make_tuple(state.rows());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape,dtype);
	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];
	p::object temp = get_humanaction(state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetNumHumanAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];
	return action;
}



Eigen::VectorXd
Window::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}
	p::object get_activation = muscle_nn_module.attr("get_activation");
	Eigen::VectorXd dt = mEnv->GetDesiredTorques().head(mEnv->GetNumHumanAction());
	np::ndarray mt_np = toNumPyArray(mt);
	np::ndarray dt_np = toNumPyArray(dt);
	p::object temp = get_activation(mt_np,dt_np);
	np::ndarray activation_np = np::from_object(temp);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	float* srcs = reinterpret_cast<float*>(activation_np.get_data());
	for(int i=0;i<activation.rows();i++)
		activation[i] = srcs[i];
	return activation;
}

void
Window::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}
void
Window::
DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());
	// std::cout << bn->getName() << "  " << bn->getRelativeTransform().translation() << std::endl; 
	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();

}

void
Window::
DrawSkeleton(const SkeletonPtr& skel)
{	
	DrawBodyNode(skel->getRootBodyNode());
}

void
Window::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(va && !va->isHidden()){
		mRI->pushMatrix();
		mRI->transform(sf->getRelativeTransform());
		if(mDrawShape) DrawShape(sf->getShape().get(),va->getRGBA());
		mRI->popMatrix();
	}

	const auto& ca = sf->getCollisionAspect();
	if(ca){
		mRI->pushMatrix();
		mRI->transform(sf->getRelativeTransform());
		// Eigen::Vector4d color(0.0,1.0,1.0,1.0);
		// Eigen::Vector4d color(0.7450,0.9686,0.9333,1.0);
		// Eigen::Vector4d color(0.0,1.0,1.0,1.0);
		Eigen::Vector4d color(0.329,0.835,0.694,1.0); 

		if(mDrawCollision) DrawCollisionShape(sf->getShape().get(), color);
		mRI->popMatrix();
	}
			
	if(sf->getName() == "r_foot_ground_ShapeNode_0"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_0.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_1"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_1.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_2"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_2.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "r_foot_ground_ShapeNode_3"){
		std::ofstream output_file("./r_foot_ground_ShapeNode_3.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_0"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_0.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_1"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_1.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_2"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_2.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
	if(sf->getName() == "l_foot_ground_ShapeNode_3"){
		std::ofstream output_file("./l_foot_ground_ShapeNode_3.txt", std::ios_base::app);
		output_file << mEnv->GetWorld()->getTime() <<" " << sf->getTransform().translation()[0] <<" " << sf->getTransform().translation()[1] <<" " << sf->getTransform().translation()[2] << "\n";
	}
}

void
Window::
DrawCollisionShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);

	if (shape->is<BoxShape>())
	{
		const auto* box = static_cast<const BoxShape*>(shape);
		mRI->drawSphere(box->getSize()[1]);
	}
	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);
	
	if(mDrawOBJ == false)
	{
		// if (shape->is<BoxShape>())
		// {
		// 		const auto* box = static_cast<const BoxShape*>(shape);
		// 		mRI->drawCube(box->getSize());
		// }
		// else if (shape->is<CylinderShape>())
		// {
		// 		const auto* cylinder = static_cast<const CylinderShape*>(shape);
		// 		mRI->drawCylinder(cylinder->getRadius(), cylinder->getHeight());
		// }	
	}
	else
	{   
		
		if (shape->is<BoxShape>())
		{
			const auto* box = static_cast<const BoxShape*>(shape);
			Eigen::Vector3d talus_size(0.0756,0.0498,0.1570);
			
			if (box->getSize() == talus_size)
			{
				std::cout << box->getSize() << std::endl;
				Eigen::Vector4d color1(1.0,0.0,1.0,1.0); 
				mRI->setPenColor(color1);
				mRI->drawSphere(0.798);
				
			}
		}
		if(shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			glDisable(GL_COLOR_MATERIAL);
			mRI->drawMesh(mesh->getScale(), mesh->getMesh());
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
		}
		// if(mDrawCollision==true)
	    // {
		// 	if (shape->is<SphereShape>())
		// 	{
		// 		const auto* sphere = static_cast<const SphereShape*>(shape);
		// 		mRI->drawSphere(sphere->getRadius());
		// 	}
	    // }
	}
	// if(mDrawOBJ == true)
	// {
	// 	std::cout << "hrerewrewrwwqrwqewqewq" << std::endl;
	// 	if(shape->is<MeshShape>())
	// 	{
	// 		const auto& mesh = static_cast<const MeshShape*>(shape);
	// 		glDisable(GL_COLOR_MATERIAL);
	// 		mRI->drawMesh(mesh->getScale(), mesh->getMesh());
	// 		// float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			
	// 		// this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
	// 	}
	// 	else{
	// 		if (!mEnv->GetUsehumanobjvisual())
	// 		{
	// 			// if (shape->is<BoxShape>())
	// 			// {
	// 			// 		const auto* box = static_cast<const BoxShape*>(shape);
	// 			// 		mRI->drawCube(box->getSize());
	// 			// }
	// 			// else if (shape->is<CylinderShape>())
	// 			// {
	// 			// 		const auto* cylinder = static_cast<const CylinderShape*>(shape);
	// 			// 		mRI->drawCylinder(cylinder->getRadius(), cylinder->getHeight());
	// 			// }	
	// 		}
	// 	}
	// }
     
	glDisable(GL_COLOR_MATERIAL);
}


void 
Window::
DrawArrow(Eigen::Vector3d pos, Eigen::Vector3d force, Eigen::Vector4d color, double radius,double heightN, double coneHt)
{
	mRI->setPenColor(color);
	mRI->pushMatrix();
	mRI->translate(pos);
	mRI->drawSphere(radius);
	mRI->popMatrix();

	Eigen::Vector3d pos2 = pos + force * heightN;
	Eigen::Vector3d u(0, 0, 1);
	Eigen::Vector3d v = pos2 - pos;
	Eigen::Vector3d mid = 0.5 * (pos + pos2);
	double len = v.norm();
	v /= len;
	Eigen::Isometry3d T;
	T.setIdentity();
	Eigen::Vector3d axis = u.cross(v);
	axis.normalize();
	double angle = acos(u.dot(v));
	Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
	w_bracket(0, 1) = -axis(2);
	w_bracket(1, 0) = axis(2);
	w_bracket(0, 2) = axis(1);
	w_bracket(2, 0) = -axis(1);
	w_bracket(1, 2) = -axis(0);
	w_bracket(2, 1) = axis(0);

	Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(angle)) * w_bracket + (1.0 - cos(angle)) * w_bracket * w_bracket;
	T.linear() = R;
	T.translation() = mid;
	mRI->pushMatrix();
	mRI->transform(T);
	mRI->drawCylinder(radius, len);
	mRI->popMatrix();


	T.translation() = pos2;
	mRI->pushMatrix();
	mRI->transform(T);
	mRI->drawCone(2* radius, coneHt);
	mRI->popMatrix();
}

// void
// Window::
// PlotFigure()
// {
// 	t = mEnv->GetWorld()->getTime();
// 	Character* mCharacter = mEnv->GetCharacter();
// 	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1]-mEnv->GetGround()->getRootBodyNode()->getCOM()[1];
// 	Eigen::Vector6d root_pos = mCharacter->GetSkeleton()->getPositions().segment<6>(0);
// 	Eigen::Isometry3d cur_root_inv = mCharacter->GetSkeleton()->getRootBodyNode()->getWorldTransform().inverse();

// 	Eigen::Vector3d root_v = mCharacter->GetSkeleton()->getBodyNode(0)->getCOMLinearVelocity();
// 	double root_v_norm = root_v.norm();

// 	Eigen::Vector3d foot_l =  mCharacter->GetSkeleton()->getBodyNode("l_foot")->getWorldTransform().translation();
// 	Eigen::Vector3d foot_r =  mCharacter->GetSkeleton()->getBodyNode("r_foot")->getWorldTransform().translation();
// 	double pos_foot_l =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("l_foot_ground")->getCOM()(1);
// 	double pos_foot_r =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("r_foot_ground")->getCOM()(1);
// 	// std::cout << "----------" << pos_foot_l << "  " << pos_foot_r << std::endl;
// 	foot_left_Forward_vector.push_back(foot_l(0));
// 	foot_left_Height_vector.push_back(foot_l(1));

// 	foot_right_Forward_vector.push_back(foot_r(0));
// 	foot_right_Height_vector.push_back(foot_r(1));

// 	Eigen::VectorXd p_cur_human = mCharacter->GetSkeleton()->getPositions().tail(mCharacter->GetHumandof());
// 	double hip_joint_angle = p_cur_human[15];
// 	hip_joint_angle_vector.push_back(hip_joint_angle);
// 	//////////////////////plot the error 
// 	double plot_time = mEnv->GetWorld()->getTime();
// 	Eigen::Vector3d skel_COM = mCharacter->GetSkeleton()->getCOM();
// 	Eigen::VectorXd pos0 = mEnv->GetCharacter()->GetSkeleton()->getPositions();
// 	const std::vector<Force*> forces = mEnv->GetCharacter()->GetForces();
// 	double hip_force =0;
// 	double femur_force_l1=0;
// 	double femur_force_l2=0;
// 	double femur_force_r1=0;
// 	double femur_force_r2=0;
// 	double tibia_force_l1=0;
// 	double tibia_force_l2=0;
// 	double tibia_force_r1=0;
// 	double tibia_force_r2=0;			
// 	double hip_torque =0;
// 	double femur_torque_l1=0;
// 	double femur_torque_l2=0;
// 	double femur_torque_r1=0;
// 	double femur_torque_r2=0;
// 	double tibia_torque_l1=0;
// 	double tibia_torque_l2=0;
// 	double tibia_torque_r1=0;
// 	double tibia_torque_r2=0;			
// 	for(int i=0; i<forces.size(); ++i)
// 	{
// 		auto& _force = forces[i];
// 		if (_force->GetName().find("bushingforce")!=std::string::npos)
// 		{
// 			if (_force->GetName().find("hip")!=std::string::npos)
// 			{
// 				hip_force = _force->GetForce().norm();
// 				hip_torque = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
// 			{
// 				femur_force_l1 = _force->GetForce().norm();
// 				femur_torque_l1 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
// 			{
// 				femur_force_l2 = _force->GetForce().norm();
// 				femur_torque_l2 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
// 			{
// 				femur_force_r1 = _force->GetForce().norm();
// 				femur_torque_r1 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
// 			{
// 				femur_force_r2 = _force->GetForce().norm();
// 				femur_torque_r2 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
// 			{
// 				tibia_force_l1 = _force->GetForce().norm();
// 				tibia_torque_l1 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
// 			{
// 				tibia_force_l2 = _force->GetForce().norm();
// 				tibia_torque_l2 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
// 			{
// 				tibia_force_r1 = _force->GetForce().norm();
// 				tibia_torque_r2 = _force->GetTorque().norm();
// 			}
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
// 			{
// 				tibia_force_r2 = _force->GetForce().norm();
// 				tibia_torque_r2 = _force->GetTorque().norm();
// 			}
// 		}
		
// 		if (_force->GetName().find("springforce")!=std::string::npos)
// 		{
// 			if (_force->GetName().find("hip")!=std::string::npos)
// 					hip_force = _force->GetForce().norm();
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
// 					femur_force_l1 = _force->GetForce().norm();
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
// 					femur_force_l2 = _force->GetForce().norm();
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
// 					femur_force_r1 = _force->GetForce().norm();
// 			if ((_force->GetName().find("femur")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
// 					femur_force_r2 = _force->GetForce().norm();
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l1")!=std::string::npos))
// 					tibia_force_l1 = _force->GetForce().norm();
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("l2")!=std::string::npos))
// 					tibia_force_l2 = _force->GetForce().norm();
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r1")!=std::string::npos))
// 					tibia_force_r1 = _force->GetForce().norm();
// 			if ((_force->GetName().find("tibia")!=std::string::npos) && (_force->GetName().find("r2")!=std::string::npos))
// 					tibia_force_r2 = _force->GetForce().norm();
// 		}
// 	}
// 	hip_force_vector.push_back(hip_force);
// 	hip_torque_vector.push_back(hip_torque);
// 	double femur_force_l = sqrt(pow(femur_force_l1,2) +pow(femur_force_l2,2));
// 	double femur_force_r = sqrt(pow(femur_force_r1,2) +pow(femur_force_r2,2));
// 	double tibia_force_l = sqrt(pow(tibia_force_l1,2) +pow(tibia_force_l2,2));
// 	double tibia_force_r = sqrt(pow(tibia_force_r1,2) +pow(tibia_force_r2,2));
// 	double femur_torque_l = sqrt(pow(femur_torque_l1,2) +pow(femur_torque_l2,2));
// 	double femur_torque_r = sqrt(pow(femur_torque_r1,2) +pow(femur_torque_r2,2));
// 	double tibia_torque_l = sqrt(pow(tibia_torque_l1,2) +pow(tibia_torque_l2,2));
// 	double tibia_torque_r = sqrt(pow(tibia_torque_r1,2) +pow(tibia_torque_r2,2));
	
// 	femur_force_vector_l.push_back(femur_force_l);
// 	femur_force_vector_r.push_back(femur_force_r);
// 	tibia_force_vector_l.push_back(tibia_force_l);
// 	tibia_force_vector_r.push_back(tibia_force_r);
	
// 	femur_torque_vector_l.push_back(femur_torque_l);
// 	femur_torque_vector_r.push_back(femur_torque_r);
// 	tibia_torque_vector_l.push_back(tibia_torque_l);
// 	tibia_torque_vector_r.push_back(tibia_torque_r);

// 	double r = mEnv->GetHumanReward();
// 	double r1= mEnv->GetReward();

// 	// Eigen::VectorXd muscle_activation = mEnv->GetActivationLevels();

// 	// muscle_activation_vector.push_back(muscle_activation(100)); 

// 	std::tuple<double,double,double,double,double,Eigen::VectorXd,Eigen::VectorXd,double,double,double,double>tmp = mEnv->GetRenderReward_Error();
// 	double pos_reward = std::get<0>(tmp);
// 	double vel_reward = std::get<1>(tmp);
// 	double ee_reward = std::get<2>(tmp);
// 	double root_reward = std::get<3>(tmp);
// 	double cop_left_reward = std::get<4>(tmp);
// 	double cop_right_reward = std::get<5>(tmp);

// 	Eigen::VectorXd torque =  std::get<6>(tmp);

// 	double hip_l_tar = std::get<7>(tmp);
// 	double knee_l_tar = std::get<8>(tmp);
// 	double ankle_l_tar = std::get<9>(tmp);
// 	double foot_l_tar = std::get<10>(tmp);

// 	double hip_l_cur = std::get<11>(tmp);
// 	double knee_l_cur = std::get<12>(tmp);
// 	double ankle_l_cur = std::get<13>(tmp);
// 	double foot_l_cur = std::get<14>(tmp);
// 	// double cop_left_error = std::get<15>(tmp);
// 	// double cop_right_error = std::get<16>(tmp);

// 	pos_reward_vector.push_back(pos_reward); 
// 	vel_reward_vector.push_back(vel_reward); 
// 	ee_reward_vector.push_back(ee_reward); 
// 	root_reward_vector.push_back(root_reward); 
// 	cop_left_reward_vector.push_back(cop_left_reward); 
// 	cop_right_reward_vector.push_back(cop_right_reward); 


// 	skel_COM_Forward_vector.push_back(skel_COM(0));
// 	skel_COM_Height_vector.push_back(skel_COM(1));
// 	skel_COM_Lateral_vector.push_back(skel_COM(2));


// 	if(mEnv->GetUseSymmetry()){
// 		for(int i; i<mEnv->GetNumAction()/2; i++)
// 			torque_vectors[i]->push_back(torque[i]); 
// 	}
// 	else{
// 		for(int i; i<mEnv->GetNumAction(); i++)
// 			torque_vectors[i]->push_back(torque[i]); 
// 	}

// 	// if (mEnv->GetUseHumanNN())
// 	// {
// 	// 	for(int i = mEnv->GetNumAction(); i<mEnv->GetNumAction()+mEnv->GetNumHumanAction(); i++)
// 	// 			human_torque_vectors[i]->push_back(torque[i]); 
// 	// }

// 	hip_l_tar_vector.push_back(hip_l_tar);
// 	knee_l_tar_vector.push_back(knee_l_tar);
// 	ankle_l_tar_vector.push_back(ankle_l_tar);
// 	foot_l_tar_vector.push_back(foot_l_tar);


// 	hip_l_cur_vector.push_back(hip_l_cur*180/3.14);
// 	knee_l_cur_vector.push_back(knee_l_cur*180/3.14);
// 	ankle_l_cur_vector.push_back(ankle_l_cur*180/3.14);
// 	foot_l_cur_vector.push_back(foot_l_cur*180/3.14);

// 	// cop_left_error_vector.push_back(cop_left_error); 
// 	// cop_right_error_vector.push_back(cop_right_error); 

// 	double cop_left_error = 0;
// 	double cop_right_error = 0;
// 	Eigen::VectorXd cop_error = Eigen::VectorXd::Zero(2);
// 	if ((cop_left_reward==0)||(cop_right_reward==0))
// 	{
// 		// cop_right_error = log(exp(1/cop_right_reward))/40;
// 		cop_total_reward_vector.push_back(cop_right_reward+cop_left_reward);
// 	}
// 	else if ((cop_right_reward!=0) && (cop_left_reward!=0))
// 	{
// 		// std::cout << "cop_left_reward :  " << cop_left_reward << std::endl;
// 		cop_left_error = log(exp(1/cop_left_reward))/40;
// 		cop_right_error = log(exp(1/cop_right_reward))/40;
// 		cop_error << cop_left_error, cop_right_error;
// 		cop_total_reward_vector.push_back(exp(-40*cop_error.squaredNorm()));			
// 	}



// 	time_vector.push_back(plot_time);
	
// 	std::map<std::string, std::string> a0 = {{"color","black"}, {"linestyle","--"},{"label","pos_reward"}};
// 	std::map<std::string, std::string> a1 = {{"color","magenta"}, {"linestyle",":"},{"label","ee_reward"}};
// 	std::map<std::string, std::string> a2 = {{"color","yellow"}, {"label","root_reward"}};
// 	std::map<std::string, std::string> a3 = {{"color","red"},{"marker","+"}, {"label","CoP_left_reward"}};
// 	std::map<std::string, std::string> a4 = {{"color","green"}, {"label","CoP_right_reward"}};
// 	std::map<std::string, std::string> a_sum = {{"color","green"}, {"label","CoP_reward"}};

// 	std::map<std::string, std::string> a5 = {{"color","black"}, {"label","hip_l_tar"}};
// 	std::map<std::string, std::string> a6 = {{"color","blue"}, {"label","knee_l_tar"}};
// 	std::map<std::string, std::string> a7 = {{"color","red"}, {"label","ankle_l_tar"}};
// 	std::map<std::string, std::string> a8 = {{"color","magenta"}, {"label","foot_l_tar"}};

// 	std::map<std::string, std::string> a9 = {{"color","blue"}, {"linewidth","1.5"}, {"label","hip"}};
// 	std::map<std::string, std::string> a10 = {{"color","red"},  {"linewidth","1.5"},{"linestyle","-."}, {"label","knee"}};
// 	std::map<std::string, std::string> a11 = {{"color","black"},  {"linewidth","1.5"},{"linestyle","--"}, {"label","ankle dorsi/plantar"}};
// 	std::map<std::string, std::string> a12 = {{"color","cyan"},  {"linewidth","1.5"},{"linestyle","--"}, {"label","ankle inversion/eversion"}};
// 	std::map<std::string, std::string> a13 = {{"color","magenta"}, {"label","COP_left_error"}};
// 	std::map<std::string, std::string> a14 = {{"color","yellow"}, {"label","COP_right_error"}};

// 	std::map<std::string, std::string> a15 = {{"color","black"}, {"linewidth","1"},{"label","hip"}};
// 	std::map<std::string, std::string> a16 = {{"color","blue"}, {"linestyle","-."}, {"label","knee"}};
// 	std::map<std::string, std::string> a17 = {{"color","red"}, {"linestyle","--"}, {"label","ankle"}};
// 	std::map<std::string, std::string> a18 = {{"color","magenta"}, {"linestyle","--"}, {"label","foot"}};

// 	std::map<std::string, std::string> a19 = {{"color","white"},{"linewidth","1"},{"label","hip_joint"}};
// 	std::map<std::string, std::string> a20 = {{"color","blue"}, {"linestyle","-."},{"label","action-knee"}};
// 	std::map<std::string, std::string> a21 = {{"color","red"}, {"linestyle","--"}, {"label","action-ankle"}};
// 	std::map<std::string, std::string> a22 = {{"color","magenta"}, {"linestyle","--"}, {"label","action-foot"}};
	

// 	std::map<std::string, std::string> a23 = {{"color","blue"}, {"linewidth","1.5"},{"label","hip"}};
// 	std::map<std::string, std::string> a24 = {{"color","red"},{"linewidth","1.5"},{"linestyle","-."},{"label","femur"}};
// 	std::map<std::string, std::string> a25 = {{"color","red"}, {"linewidth","1.5"},{"linestyle","-."},{"label","femur_force_r"}};
// 	std::map<std::string, std::string> a26 = {{"color","black"}, {"linewidth","1.5"}, {"linestyle","--"},{"label","tibia"}};
// 	std::map<std::string, std::string> a27 = {{"color","black"}, {"linewidth","1.5"}, {"linestyle","--"},{"label","tibia_force_r"}};

// 	std::map<std::string, std::string> a28 = {{"color","red"}, {"linewidth","1"},{"label","skel_COM_XY"}};
// 	std::map<std::string, std::string> a29 = {{"color","black"}, {"linewidth","1"},{"label","skel_COM_height"}};
// 	std::map<std::string, std::string> a30 = {{"color","blue"}, {"linewidth","1"},{"label","skel_COM_lateral"}};

// 	std::map<std::string, std::string> a31 = {{"color","blue"},{"linewidth","1.5"},{"linestyle","-."},{"label","hip_torque"}};
// 	std::map<std::string, std::string> a32 = {{"color","red"}, {"linewidth","1.5"},{"linestyle","-."},{"label","femur_torque"}};
// 	std::map<std::string, std::string> a33 = {{"color","black"}, {"linewidth","1.5"},{"label","tibia_torque"}};

// 	std::map<std::string, std::string> a34 = {{"color","blue"},{"linewidth","1.5"},{"linestyle","-."},{"label","hip_cur_joint_angle"}};
// 	std::map<std::string, std::string> a_human_hip_joint = {{"color","blue"}, {"linewidth","1"},{"label","hip_joint_human"}};

// 	std::map<std::string, std::string> a123 = {{"fontsize","10"}};

// 	// const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
// 	// std::map<std::string, std::vector<double>>::iterator it;
// 	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	// {
// 	// 	it->second.push_back(0.0);
// 	// }
// 	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	// {
// 	// 	int i=0;
// 	// 	for(auto muscle : muscles) 
// 	// 	{
// 	// 		// std::cout << muscle->GetMuscleUnitName() << std::endl;
// 	// 		if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
// 	// 		{
// 	// 			i=i+1;
// 	// 			it->second.back() += muscle->GetMuscleUnitactivation();
// 	// 		}
// 	// 	}
// 	// 	it->second.back() = it->second.back()/i;
// 	// }

// 	const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
// 	std::map<std::string, std::vector<double>>::iterator it;

// 	for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	{
// 		it->second.push_back(0.0);
// 	}
// 	for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	{
// 		int i=0;
// 		for(auto muscle : muscles) 
// 		{
// 			// std::cout << muscle->GetMuscleUnitName() << std::endl;
// 			if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
// 			{
// 				i=i+1;
// 				it->second.back() += muscle->GetMuscleUnitactivation();
// 			}
// 		}
// 		it->second.back() = it->second.back()/i;
// 	}

// 	if (mDrawFigure)
// 	{ 
// 		plt::figure(0);
// 		plt::clf();
// 		plt::subplot(4,1,1);
// 		// plt::title("Real-time Reward");
// 		// plt::xlabel("Time/s");
// 		plt::ylabel("Reward");
// 		plt::plot(time_vector,pos_reward_vector, a0);
// 		plt::plot(time_vector,ee_reward_vector,a1);
// 		plt::plot(time_vector,root_reward_vector,a2);
// 		std::map<std::string, std::string> loc = {{"loc","upper left"}};
// 		plt::legend(loc);


// 		plt::subplot(4,1,2);
// 		// plt::title("CoP Reward");
// 		// plt::xlabel("Time/s");
// 		plt::ylabel("CoP Reward");
// 		// plt::plot(time_vector,cop_left_reward_vector,a3);
// 		// plt::plot(time_vector,cop_right_reward_vector,a4);
// 		plt::plot(time_vector,cop_total_reward_vector,a_sum);
// 		plt::legend(loc);


// 		// plt::subplot(4,1,3);
// 		// // plt::title("Joint angle tracking");
// 		// // plt::xlabel("Time/s");
// 		// plt::ylabel("Angle tracking");
// 		// plt::plot(time_vector,hip_l_tar_vector,a5);
// 		// plt::plot(time_vector,knee_l_tar_vector,a6);
// 		// plt::plot(time_vector,ankle_l_tar_vector,a7);
// 		// plt::plot(time_vector,foot_l_tar_vector,a8);

// 		// plt::plot(time_vector,hip_l_cur_vector,a9);
// 		// plt::plot(time_vector,knee_l_cur_vector,a10);
// 		// plt::plot(time_vector,ankle_l_cur_vector,a11);
// 		// plt::plot(time_vector,foot_l_cur_vector,a12);
// 		// plt::legend(loc);


// 		plt::subplot(4,1,3);
// 		// plt::title("Joint torque");
// 		// plt::xlabel("Time/s");
// 		plt::ylabel("Joint Torque/N*m");
// 		plt::plot(time_vector,*(torque_vectors[0]),a15);
// 		plt::plot(time_vector,*(torque_vectors[1]), a16);
// 		plt::plot(time_vector,*(torque_vectors[2]), a17);
// 		plt::plot(time_vector,*(torque_vectors[3]), a18);
// 		plt::legend(loc);

// 		plt::subplot(4,1,4);
// 		plt::xlabel("Time/s");
// 		plt::ylabel("Action from NN/rad");
// 		plt::plot(time_vector, action_hip_vector, a15);
// 		plt::plot(time_vector, action_knee_vector, a16);
// 		plt::plot(time_vector, action_ankle_vector, a17);
// 		plt::plot(time_vector, action_foot_vector, a18);
// 		plt::legend(loc);


// 		if (mDrawBushingforce)
// 		{
// 			plt::figure(1);
// 			plt::clf();
// 			// plt::subplot(3,1,1);
// 			plt::xlabel("Time/s");
// 			plt::ylabel("Human strap force/N");
// 			plt::title("Human strap force");
// 			plt::plot(time_vector,hip_force_vector,a23);
// 			// plt::legend(loc);
// 			// plt::subplot(3,1,2);
// 			plt::plot(time_vector,femur_force_vector_l,a24);
// 			// plt::plot(time_vector,femur_force_vector_r,a17);
// 			// plt::legend(loc);
// 			// plt::subplot(3,1,3);
// 			plt::plot(time_vector,tibia_force_vector_l,a26);
// 			// plt::plot(time_vector,tibia_force_vector_l,a19);
// 			plt::legend(loc);
// 		}

// 		// plt::figure(2);
// 		// plt::clf();
// 		// plt::xlabel("Time/s", a123);
// 		// plt::ylabel("Joint angle/rad", a123);
// 		// plt::plot(time_vector, action_hip_vector, a9);
// 		// plt::plot(time_vector, action_knee_vector, a10);
// 		// plt::plot(time_vector, action_ankle_vector, a11);
// 		// plt::plot(time_vector, action_foot_vector, a12);
// 		// plt::legend(loc);


// 		// plt::figure(3);
// 		// plt::clf();
// 		// plt::xlabel("Time/s", a123);
// 		// plt::ylabel("Joint torque/Nm", a123);
// 		// plt::plot(time_vector,*(torque_vectors[0]),a9);
// 		// plt::plot(time_vector,*(torque_vectors[1]), a10);
// 		// plt::plot(time_vector,*(torque_vectors[2]), a11);
// 		// plt::plot(time_vector,*(torque_vectors[3]), a12);
// 		// plt::legend(loc);

// 		// plt::figure(4);
// 		// plt::clf();
// 		// plt::xlabel("Time/s");
// 		// plt::ylabel("Human strap torque/N");
// 		// plt::plot(time_vector,hip_torque_vector,a31);
// 		// plt::plot(time_vector,femur_torque_vector_l,a32);
// 		// plt::plot(time_vector,tibia_torque_vector_l,a33);
// 		// plt::legend(loc);
	

// 		// plt::figure(2);
// 		// plt::clf();
// 		// // plt::subplot(3,1,1);
// 		// plt::title("COM position");
// 		// plt::xlabel("X/m");
// 		// plt::ylabel("Y/m");
// 		// plt::plot(skel_COM_Forward_vector,skel_COM_Height_vector,a26);
// 		// // plt::plot(time_vector,skel_COM_Height_vector,a27);
// 		// // plt::plot(time_vector,skel_COM_Lateral_vector,a28);
// 		// plt::legend(loc);


// 		// plt::figure(2);
// 		// plt::clf();
// 		// // plt::title("hip joint torque");
// 		// plt::xlabel("Time/s");
// 		// plt::ylabel("hip joint torque/N*m");
// 		// plt::plot(time_vector,*(torque_vectors[9]),a_human_hip_joint);
// 		// plt::plot(time_vector,*(torque_vectors[0]),a15);
// 		// plt::legend(loc);

// 		// plt::figure(3);
// 		// plt::clf();
// 		// plt::xlabel("Time/s");
// 		// plt::ylabel("hip joint action");
// 		// plt::plot(time_vector,action_human_hip_vector,a_human_hip_joint);
// 		// plt::plot(time_vector, hip_joint_angle_vector, a34);
// 		// plt::legend(loc);


// 		plt::figure(4);
// 		plt::clf();
// 		// plt::xlabel("Time/s");
// 		// plt::ylabel("muscle_activation");
// 		int i=1;
// 		for (it = muscle_plot.begin(); it != muscle_plot.end(); it++){
// 			// if ((it->first.find("L_Flexor_Hallucis") != std::string::npos) || (it->first.find("L_Vastus_Intermedius")!= std::string::npos) || (it->first.find("L_Tibialis_Posterior")!= std::string::npos))
// 			// 	continue;
// 			plt::subplot(muscle_plot.size(),1,i);
// 			// if (it->first.find("L_Rectus_Femoris") != std::string::npos)
// 			plt::ylabel("Muscle activation");
// 			// if (it->first.find("L_iliacus") != std::string::npos)
// 			plt::xlabel("Time/s");
// 			plt::plot(time_vector,it->second, muscle_plot_legend[it->first]);
// 			i++;
// 			plt::legend(loc);
// 		}
// 		plt::legend(loc);
// 		plt::show();
// 		plt::pause(0.00001); 
// 	}
// 	// 	if(plot_time > 22.0){
// 	// 		std::cout << "plot_time:" << plot_time << std::endl;
// 	// 		std::vector<std::string> names;
// 	// 		std::vector<std::vector<double> > values;
// 	// 		names.push_back("time_vector"); 
// 	// 		values.push_back(time_vector); 
// 	// 		names.push_back("hip_l_cur_vector"); 
// 	// 		values.push_back(hip_l_cur_vector); 
// 	// 		names.push_back("knee_l_cur_vector"); 
// 	// 		values.push_back(knee_l_cur_vector); 
// 	// 		names.push_back("ankle_l_cur_vector"); 
// 	// 		values.push_back(ankle_l_cur_vector); 
// 	// 		names.push_back("foot_l_cur_vector"); 
// 	// 		values.push_back(foot_l_cur_vector); 
// 	// 		names.push_back("hip_torque"); 
// 	// 		values.push_back(*(torque_vectors[0])); 
// 	// 		names.push_back("knee_torque"); 
// 	// 		values.push_back(*(torque_vectors[1])); 
// 	// 		names.push_back("ankle_torque"); 
// 	// 		values.push_back(*(torque_vectors[2])); 
// 	// 		names.push_back("foot_torque"); 
// 	// 		values.push_back(*(torque_vectors[3])); 			
// 	// 		// names.push_back("hip_action"); 
// 	// 		// values.push_back(action_hip_vector); 
// 	// 		// names.push_back("knee_action"); 
// 	// 		// values.push_back(action_knee_vector); 
// 	// 		// names.push_back("ankle_action"); 
// 	// 		// values.push_back(action_ankle_vector); 
// 	// 		names.push_back("hip_force"); 
// 	// 		values.push_back(hip_force_vector); 
// 	// 		names.push_back("femur_force"); 
// 	// 		values.push_back(femur_force_vector_l); 
// 	// 		names.push_back("tibia_force"); 
// 	// 		values.push_back(tibia_force_vector_l); 
// 	// 	for(int i=0; i<names.size();i++){
// 	// 		std::ofstream output_file("./"+names[i]+".txt");
// 	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
// 	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
// 	// 			double v = *it; 
// 	// 			output_file << std::to_string(v) << '\n';
// 	// 		}
// 	// 	}
// 	// 	// std::exit(EXIT_FAILURE); 
// 	// }



// 	// const std::vector<Muscle*>& muscles = mEnv->GetCharacter()->GetMuscles();
// 	// std::map<std::string, std::vector<double>>::iterator it;

// 	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	// {
// 	// 	it->second.push_back(0.0);
// 	// }
// 	// for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	// {
// 	// 	int i=0;
// 	// 	for(auto muscle : muscles) 
// 	// 	{
// 	// 		// std::cout << muscle->GetMuscleUnitName() << std::endl;
// 	// 		if (muscle->GetMuscleUnitName().find(it->first) != std::string::npos)
// 	// 		{
// 	// 			i=i+1;
// 	// 			it->second.back() += muscle->GetMuscleUnitactivation();
// 	// 		}
// 	// 	}
// 	// 	it->second.back() = it->second.back()/i;
// 	// }
// 	// if(plot_time>22.0)
// 	// {
// 	// 	std::vector<std::string> names;
// 	// 	std::vector<std::vector<double> > values;
// 	// 	for (it = muscle_plot.begin(); it != muscle_plot.end(); it++)
// 	// 	{
// 	// 		names.push_back(it->first); 
// 	// 		values.push_back(it->second); 
// 	// 	}
// 	// 	// names.push_back("cop_right_reward"); 
// 	// 	// values.push_back(cop_right_reward_vector); 
// 	// 	// std::cout << "-----------------\n"; 
// 	// 	// std::cout << values[1][0] << std::endl; 
// 	// 	std::time_t now = time(0);
// 	// 	// Convert now to tm struct for local timezone
// 	// 	std::tm* localtm = localtime(&now);

// 	// 	std::string currenttime = std::asctime(localtm);
// 	// 	for(int i=0; i<names.size();i++){
// 	// 		std::cout << names[i] << std::endl; 
// 	// 		std::ofstream output_file("./result3/"+names[i]+".txt");
// 	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
// 	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
// 	// 			double v = *it; 
// 	// 			output_file << std::to_string(v) << '\n';
// 	// 		}
// 	// 	}
// 	// 	// std::exit(EXIT_FAILURE); 
// 	// }
	

// 	// if(plot_time>6.6)
// 	// {
// 	// 	// std::cout << "safdsafdsafdsafdaf" << std::endl;
// 	// 	std::vector<std::string> names;
// 	// 	std::vector<std::vector<double> > values;
// 	// 	names.push_back("time_vector"); 
// 	// 	values.push_back(time_vector); 
// 	// 	// names.push_back("t_vector"); 
// 	// 	// values.push_back(t_vector); 
// 	// 	names.push_back("ee_reward"); 
// 	// 	values.push_back(ee_reward_vector); 
// 	// 	names.push_back("pos_reward"); 
// 	// 	values.push_back(pos_reward_vector); 
// 	// 	names.push_back("cop_total_reward"); 
// 	// 	values.push_back(cop_total_reward_vector); 
// 	// 	// names.push_back("cop_right_reward"); 
// 	// 	// values.push_back(cop_right_reward_vector); 
// 	// 	// std::cout << "-----------------\n"; 
// 	// 	// std::cout << values[1][0] << std::endl; 

// 	// 	std::time_t now = time(0);

// 	// 	// Convert now to tm struct for local timezone
// 	// 	std::tm* localtm = localtime(&now);


// 	//     std::string currenttime = std::asctime(localtm);
// 	// 	for(int i=0; i<names.size();i++){
// 	// 		std::cout << names[i] << std::endl; 
// 	// 		std::ofstream output_file("./result_case4/"+currenttime+"_"+names[i]+".txt");
// 	// 		// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
// 	// 		for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
// 	// 			double v = *it; 
				
// 	// 			output_file << std::to_string(v) << '\n';
// 	// 		}
// 	// 	}
// 	// 	std::exit(EXIT_FAILURE); 
// 	// }

// }



// draw contact forces
void 
Window::
DrawContactForces(collision::CollisionResult& results)
{
	plt::ion();
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
    auto& ees = mEnv->GetCharacter()->GetEndEffectors();
	double radius = 0.01; //2cm radius
	double heightN = 1.5e-3;//1mm per N
	double coneHt = 2.0e-2; //cone height
    Eigen::Vector3d pos_Root = mEnv->GetCharacter()->GetSkeleton()->getRootBodyNode()->getCOM();
	Eigen::Vector3d pos_foot_r = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TalusR")->getCOM();
	Eigen::Vector3d pos_foot_l = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TalusL")->getCOM();
	pos_foot_l(1) =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FootThumbL")->getCOM()[1];
	pos_foot_r(1) =  mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FootThumbR")->getCOM()[1];
	// std::cout << "----------" << pos_foot_l(1) << "  " << pos_foot_r(1) << std::endl;
	Eigen::Vector3d geo_center_target_left = pos_foot_l;
	Eigen::Vector3d geo_center_target_right = pos_foot_r;
	Eigen::Vector3d geo_center_target = (pos_foot_l + pos_foot_r)/2;
    // Eigen::Vector4d color1(1.0,0.0,1.0,1.0); 
	// mRI->setPenColor(color1);
	// mRI->pushMatrix();
	// mRI->translate(geo_center_target_left);
	// mRI->drawSphere(radius*2);
	// mRI->popMatrix();

	// mRI->setPenColor(color1);
	// mRI->pushMatrix();
	// mRI->translate(geo_center_target_right);
	// mRI->drawSphere(radius*2);
	// mRI->popMatrix();


	Eigen::Vector3d pos = Eigen::Vector3d::Zero();
	Eigen::Vector3d force = Eigen::Vector3d::Zero();
    std::vector<constraint::ContactConstraintPtr>  mContactConstraints;

	// store all the pos and force 
	std::vector<Eigen::Vector3d> all_pos;
	std::vector<Eigen::Vector3d> all_force;
	std::vector<Eigen::Vector3d> all_pos_left;
	std::vector<Eigen::Vector3d> all_pos_right;
	std::vector<Eigen::Vector3d> all_force_left;
	std::vector<Eigen::Vector3d> all_force_right;

	Eigen::Vector4d color; 
	Eigen::Vector3d left_pos, left_force, right_pos, right_force;
	left_pos.setZero(); left_force.setZero(); right_pos.setZero(); right_force.setZero(); 
	for(int i = 0; i < results.getNumContacts(); ++i) 
	{
		auto& contact = results.getContact(i);
		mContactConstraints.clear();
		mContactConstraints.push_back(
				std::make_shared<constraint::ContactConstraint>(contact, mEnv->GetWorld()->getTimeStep()));
		auto pos = contact.point;
		auto force = contact.force;

		auto shapeFrame1 = const_cast<dynamics::ShapeFrame*>(
			contact.collisionObject1->getShapeFrame());
		auto shapeFrame2 = const_cast<dynamics::ShapeFrame*>(
			contact.collisionObject2->getShapeFrame());
	DART_SUPPRESS_DEPRECATED_BEGIN
		auto body1 = shapeFrame1->asShapeNode()->getBodyNodePtr();
		auto body2 = shapeFrame2->asShapeNode()->getBodyNodePtr();
	DART_SUPPRESS_DEPRECATED_END

		for (auto& contactConstraint : mContactConstraints)
		{
			if (body1->getName()=="TalusL"||body1->getName()=="FootThumbL"||body1->getName() =="FootPinkyL")
			{
				all_pos_left.push_back(pos);
				all_force_left.push_back(force);
			}
			else if(body1->getName()=="TalusR"||body1->getName()=="FootThumbR"||body1->getName() =="FootPinkyR")
			{
				all_pos_right.push_back(pos);
				all_force_right.push_back(force);
			}
			else
			{
				// std::cout << body1->getName() << std::endl;
				// std::cout << "-----Warning: contact force not on foot-------" << std::endl;
			}
		}
		all_pos.push_back(pos);
		all_force.push_back(force);
		for (const auto& contactConstraint : mContactConstraints)
		{
			if (body1->getName()=="TalusL"||body1->getName()=="FootThumbL"||body1->getName() =="FootPinkyL")
			{
				color << 1.0,0.0,0.0,1.0;  //red
				left_pos += pos;
				left_force += force;
			}
			else if(body1->getName()=="TalusR"||body1->getName()=="FootThumbR"||body1->getName() =="FootPinkyR")
			{
				color << 0.0,1.0,0.0,1.0;  //
				right_pos += pos;
				right_force += force;
			}
			else
			{
				// std::cout << body1->getName() << std::endl;
				// std::cout << "-----Warning: contact force not on foot-------" << std::endl;
			}
		}
		// if (!mDrawCompositionforces)
		//    DrawArrow(pos, force, color, radius, heightN, coneHt);
        
	}

//////////////////////////////////////////////// calculate the COP(center of pressure)
	Eigen::Vector3d unitV;
	unitV << 0, 1, 0;    // unit normal vector  
	Eigen::Matrix3f A;
	Eigen::Vector3f b;
	Eigen::Vector3d p, f;

    //////////////////////////////////////////// first method  -- calculate COP of both foot 
	// Eigen::Vector3d COP;
	// Eigen::Vector3d p_cross_f;
	// double f_sum = 0; 
	// p_cross_f.setZero();

	// for(int i=0; i<all_pos.size(); i++){
	// 	p = all_pos[i];
	// 	double f_scalar = all_force[i].dot(unitV);
	// 	f_sum += f_scalar; 
	// 	p_cross_f += p.cross(f_scalar * unitV);
	// }
	// if(all_pos.size() != 0){
	// 	COP = -p_cross_f.cross(unitV) / f_sum;
	// 	COP(1) = geo_center_target(1);	
	// 	//std::cout << "COP_right:\n" << COP_right << std::endl;	
	// 	Eigen::Vector4d color4(0.0,0.0,1.0,1.0);  
	// 	mRI->setPenColor(color4);
	// 	mRI->pushMatrix();
	// 	mRI->translate(COP);
	// 	mRI->drawSphere(radius*2);
	// 	mRI->popMatrix();
	// }

	//////////////////////////////////////////// first method  -- calculate COP of each foot
	Eigen::Vector3d COP_left;
	COP_left = geo_center_target_left;
	//std::cout << "geo_center_target_left   "  << geo_center_target_left << std::endl;
	Eigen::Vector3d p_cross_f_left;
	double f_sum_left = 0; 
	p_cross_f_left.setZero();

	for(int i=0; i<all_pos_left.size(); i++){
		p = all_pos_left[i];
		double f_scalar_left = all_force_left[i].dot(unitV);
		f_sum_left += f_scalar_left; 
		p_cross_f_left += p.cross(f_scalar_left * unitV);
	}
	if(all_pos_left.size()!= 0){
		COP_left = -p_cross_f_left.cross(unitV) / f_sum_left;
		COP_left(1) = geo_center_target_left(1);
		
		Eigen::Vector4d color4(0.0,1.0,0.0,1.0);  //red
		mRI->setPenColor(color4);
		mRI->pushMatrix();
		mRI->translate(COP_left);
		mRI->drawSphere(radius*2);
		mRI->popMatrix();
	}
	else 
		COP_left.setZero();

	Eigen::Vector3d COP_right;
	COP_right = geo_center_target_right;
	Eigen::Vector3d p_cross_f_right;
	double f_sum_right = 0; 
	p_cross_f_right.setZero();

	for(int i=0; i<all_pos_right.size(); i++){
		p = all_pos_right[i];
		double f_scalar_right = all_force_right[i].dot(unitV);
		f_sum_right += f_scalar_right; 
		p_cross_f_right += p.cross(f_scalar_right * unitV);
	}
	if(all_pos_right.size() != 0){
		COP_right = -p_cross_f_right.cross(unitV) / f_sum_right;
		COP_right(1) = geo_center_target_right(1);	
		//std::cout << "COP_right:\n" << COP_right << std::endl;	
		Eigen::Vector4d color4(1.0,0.0,0.0,1.0);  
		mRI->setPenColor(color4);
		mRI->pushMatrix();
		mRI->translate(COP_right);
		mRI->drawSphere(radius*2);
		mRI->popMatrix();
	}
	else
		COP_right.setZero();


	if (COP_left(0)!=0)
		cop_left_Forward_vector.push_back(COP_left(0)); 
	else
		cop_left_Forward_vector.push_back(nan("")); 
	
	if (COP_left(1)!=0)	
		cop_left_Height_vector.push_back(COP_left(1));
	else
		cop_left_Height_vector.push_back(nan("")); 

	if (COP_left(2)!=0)	
		cop_left_Lateral_vector.push_back(COP_left(2));
	else
		cop_left_Lateral_vector.push_back(nan("")); 


	if (COP_right(0)!=0)
		cop_right_Forward_vector.push_back(COP_right(0));
	else
		cop_right_Forward_vector.push_back(nan("")); 


	if (COP_right(1)!=0)	 
		cop_right_Height_vector.push_back(COP_right(1));
	else
		cop_right_Height_vector.push_back(nan("")); 

	if (COP_right(2)!=0)	
		cop_right_Lateral_vector.push_back(COP_right(2));
	else
		cop_right_Lateral_vector.push_back(nan("")); 



	// if (mDrawCompositionforces)
	// {
	left_pos /= 4;
	right_pos /= 4;
	color << 0.835, 0.694, 0.329;  //red
	DrawArrow(COP_left, left_force, color, radius, heightN, coneHt);
	// color << 0.0,1.0,0.0,1.0;  //green
	DrawArrow(COP_right, right_force, color, radius, heightN, coneHt);
	// }
	double plot_time = mEnv->GetWorld()->getTime();
	t_vector.push_back(plot_time);
	std::map<std::string, std::string> loc = {{"loc","upper left"}};
	std::map<std::string, std::string> a0_0 = {{"color","blue"},{"label","contact_force_forward"}};
	std::map<std::string, std::string> a0_1 = {{"color","red"},{"label","contact_force_height"}};
	std::map<std::string, std::string> a0_2 = {{"color","black"},{"label","contact_force_lateral"}};

	std::map<std::string, std::string> a1_0 = {{"color","red"}, {"linewidth","1"},{"label","COP_left"}};
	std::map<std::string, std::string> a1_1 = {{"color","green"}, {"linewidth","1"},{"label","COP_right"}};

	contact_force_vector_l_forward.push_back(left_force(0));
	contact_force_vector_l_height.push_back(left_force(1));
	contact_force_vector_l_lateral.push_back(left_force(2));
	contact_force_vector_r_forward.push_back(right_force(0));
	contact_force_vector_r_height.push_back(right_force(1));
	contact_force_vector_r_lateral.push_back(right_force(2));

	if (mDrawFigure)
	{ 
		plt::figure(2);
		plt::clf();
		plt::subplot(2,1,1);
		plt::title("Contact force_foot");
		// plt::xlabel("Time/s");
		plt::ylabel("Force/N");
		plt::plot(t_vector,contact_force_vector_l_height,a0_1);
		plt::plot(t_vector,contact_force_vector_r_height,a0_2);
		plt::legend(loc);
		plt::show();
		plt::pause(0.00001); 
	}

	if(plot_time > 11.0){
			std::cout << "time------------" << std::endl;
			std::vector<std::string> names;
			std::vector<std::vector<double> > values;
			names.push_back("contact_force_vector_l_height"); 
			values.push_back(contact_force_vector_l_height); 
			names.push_back("contact_force_vector_r_height"); 
			values.push_back(contact_force_vector_r_height); 


			std::time_t now = time(0);
			// Convert now to tm struct for local timezone
			std::tm* localtm = localtime(&now);
			std::string currenttime = std::asctime(localtm);

			for(int i=0; i<names.size();i++){
				std::ofstream output_file("./"+names[i]+".txt");
				// for (std::vector<int>::iterator it = myvector.begin() ; it != myvector.end(); ++it)
				for(std::vector<double>::iterator it = values[i].begin(); it != values[i].end(); ++it) {
					double v = *it; 
					output_file << std::to_string(v) << '\n';
				}
			}
			std::exit(EXIT_FAILURE); 
		}

	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

   //////////////////////////////////////////// second method
	// A.setZero();
	// b.setZero();
	// for(int i=0; i<all_pos_left.size(); i++){
	// 	p = all_pos_left[i];
	// 	f = all_force_left[i].dot(unitV) * unitV;

	// 	A(0, 0) += 0;
	// 	A(0, 1) += -f(2);
	// 	A(0, 2) += f(1);

	// 	A(1, 0) += -f(1);
	// 	A(1, 1) += f(0);
	// 	A(1, 2) += 0;

	// 	A(2, 0) += unitV(0);
	// 	A(2, 1) += unitV(1);
	// 	A(2, 2) += unitV(2);

	// 	b(0) += p(2) * f(1) - p(1)* f(2);
	// 	b(1) += p(1) * f(0) - p(0)* f(1);
	// 	b(2) += 0; //p.dot(unitV); 
	// }

	// if(all_pos_left.size() != 0)
	// {
	// 	Eigen::Vector3f COP_left1 = A.colPivHouseholderQr().solve(b);
	// 	std::cout << "COP_left1:\n" << COP_left1 << std::endl; 
	// 	COP_left1(1) = -0.88;
	// 	Eigen::Vector4d color4(0.0,0.8,0.0,1.0);  //red
	// 	mRI->setPenColor(color4);
	// 	mRI->pushMatrix();
	// 	mRI->translate(COP_left1.cast<double>());
	// 	mRI->drawSphere(radius*2);
	// 	mRI->popMatrix();
	// }

}


void Window::
DrawEndEffectors()
{
	auto bvh = mEnv->GetCharacter()->GetBVH();
	auto& ees = mEnv->GetCharacter()->GetEndEffectors();

	Eigen::VectorXd ee_diff(ees.size()*3);
	Eigen::VectorXd com_diff;

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	double radius = 0.015; //2cm radius
	Eigen::Vector4d color(0.0,1.0,0.0,1.0);//green
	//std::cout << "------------ end t = " << t << std::endl; 
	for(int i =0;i<ees.size();i++) {

		mRI->setPenColor(color);
		mRI->pushMatrix();
		mRI->translate(ees[i]->getCOM());
		mRI->drawSphere(radius);
		mRI->popMatrix();
	}

	if(mDrawEndEffectorTargets) {

		Eigen::Vector4d color2(0.0,0.0,1.0,1.0);//blue
		auto skel = mEnv->GetCharacter()->GetSkeleton();
		com_diff = skel->getCOM();
 
		double t = mEnv->GetWorld()->getTime();

		// ee posotion based on BVH
		Character* mCharacter = mEnv->GetCharacter();
        auto pos0 = skel->getPositions();
		std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = mCharacter->GetTargetPosAndVel(t,1.0/mEnv->GetControlHz());
		auto targetPositions = std::get<0>(pv);
		auto targetVelocities = std::get<1>(pv);
		auto targetEE_pos = std::get<2>(pv);
		mCharacter->targetEE_pos = targetEE_pos; 
		mCharacter->GetSkeleton()->setPositions(targetPositions); // set position
		mCharacter->GetSkeleton()->setVelocities(targetVelocities); //set velocities
		mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);

	    // auto pos0 = skel->getPositions();

        auto ees = mCharacter->GetEndEffectors();

		for(int i =0;i<ees.size();i++) {

			mRI->setPenColor(color2);
			mRI->pushMatrix();
			mRI->translate(ees[i]->getCOM());
			mRI->drawSphere(radius);
			mRI->popMatrix();
		}

		skel->setPositions(pos0); //changed the state back
		skel->computeForwardKinematics(true,false,false);

	}

	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}


void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	for(auto muscle : muscles)
	{
		auto aps = muscle->GetAnchors();
		bool lower_body = true;
		double a = muscle->activation;
		// Eigen::Vector3d color(0.7*(3.0*a),0.2,0.7*(1.0-3.0*a));
		// Eigen::Vector4d color(0.3+(2.0*a),0.1,1.0,1.0);//0.7*(1.0-3.0*a));
		Eigen::Vector3d color(0.7,0.7,0.7);
		// glColor3f(1.0,0.0,0.362);
		// glColor3f(0.0,0.0,0.0);
		mRI->setPenColor(color);
		for(int i=0;i<aps.size();i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			mRI->pushMatrix();
			mRI->translate(p);
			mRI->drawSphere(0.005*sqrt(muscle->f0/1000.0));
			mRI->popMatrix();
		}
			
		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			Eigen::Vector3d p1 = aps[i+1]->GetPoint();

			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);

			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.005*sqrt(muscle->f0/1000.0),len);
			mRI->popMatrix();
		}
for(int i=0;i<aps.size()-1;i++)
			{
				
				Eigen::Vector3d color1;
				color1 << 0.2549,0.2117,0.5725;
				if (muscle->f0 ==1792.950000)
					color1 << 0.2549,0.2117,0.5725;
				if (muscle->f0 ==1127.700000|| muscle->f0 ==705.200000)
					color1 << 0.98823529411,0.23921568627,0.23137254902;
				mRI->setPenColor(color1);
				Eigen::Vector3d p = aps[i]->GetPoint();
				Eigen::Vector3d p1 = aps[i+1]->GetPoint();
				Eigen::Vector3d u(0,0,1);
				Eigen::Vector3d v = p-p1;
				Eigen::Vector3d mid = 0.5*(p+p1);
				double len = v.norm();
				v /= len;
				Eigen::Isometry3d T;
				T.setIdentity();
				Eigen::Vector3d axis = u.cross(v);
				axis.normalize();
				double angle = acos(u.dot(v));
				Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
				w_bracket(0, 1) = -axis(2);
				w_bracket(1, 0) =  axis(2);
				w_bracket(0, 2) =  axis(1);
				w_bracket(2, 0) = -axis(1);
				w_bracket(1, 2) = -axis(0);
				w_bracket(2, 1) =  axis(0);
				Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
				T.linear() = R;
				T.translation() = mid;
				mRI->pushMatrix();
				mRI->transform(T);
				mRI->drawCylinder(0.008*sqrt(1792.950000/1000.0),len/2);
				mRI->popMatrix();
					
			}
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}
void
Window::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y) 
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}
void
Window::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);
    
    // update transform

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
        	for (i = 0; i < face->mNumIndices; i++)
        	{
        		int index = face->mIndices[i];

        		v[0] = (&mesh->mVertices[index].x)[0];
        		v[1] = (&mesh->mVertices[index].x)[1];
        		v[2] = (&mesh->mVertices[index].x)[2];
        		v = M*v;
        		double h = v[1]-y;
        		
        		v += h*dir;
        		
        		v[1] = y+0.001;
        		glVertex3f(v[0],v[1],v[2]);
        	}
            glEnd();
        }

    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }

}


void
Window::
DrawGround(double y)
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);
	double width = 0.005;
	int count = 0;
	glBegin(GL_QUADS);
	for(double x = -100.0;x<100.01;x+=1.0)
	{
		for(double z = -100.0;z<100.01;z+=1.0)
		{
			if(count%2==0)
				glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);			
			else
				glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);
			count++;
			glVertex3f(x,y,z);
			glVertex3f(x+1.0,y,z);
			glVertex3f(x+1.0,y,z+1.0);
			glVertex3f(x,y,z+1.0);
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);
}

