#include "EnvManager.h"
#include "DARTHelper.h"
#include <omp.h>

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumEnvs);
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment());
		MASS::Environment* env = mEnvs.back();
     
		env->Initialize(meta_file,false);
        

		// env->ResetInitialState(env->GetCharacter()->GetInitialState());
		// env->SetUseMuscle(false);
		// env->SetControlHz(30);
		// env->SetSimulationHz(600);
		// env->SetRewardParameters(0.65,0.1,0.15,0.1);

		// MASS::Character* character = new MASS::Character();
		// character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"),false);
		// if(env->GetUseMuscle())
		// 	character->LoadMuscles(std::string(MASS_ROOT_DIR)+std::string("/data/muscle.xml"));

		// character->LoadBVH(std::string(MASS_ROOT_DIR)+std::string("/data/motion/walk.bvh"),true);
		
		// double kp = 300.0;
		// character->SetPDParameters(kp,sqrt(2*kp));
		// env->SetCharacter(character);
		// env->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

		// env->Initialize();
	}
}
int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}

int
EnvManager::
GetNumFullObservation()
{
	return mEnvs[0]->GetNumFullObservation();
}

int
EnvManager::
GetNumHumanObservation()
{
	return mEnvs[0]->GetNumHumanObservation();
}


int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}


int
EnvManager::
GetNumHumanAction()
{
	return mEnvs[0]->GetNumHumanAction();
}



int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}
int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}
int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}
bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}

bool
EnvManager::
UseHumanNetwork()
{
	return mEnvs[0]->GetUseHumanNN();
}


bool
EnvManager::
UseSymmetry()
{
	return mEnvs[0]->GetUseSymmetry();
}

void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}
void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}
bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}
np::ndarray 
EnvManager::
GetState(int id)
{
	return toNumPyArray(mEnvs[id]->GetState());
}

np::ndarray 
EnvManager::
GetAction(int id)
{
	return toNumPyArray(mEnvs[id]->GetAction());
}

np::ndarray 
EnvManager::
GetHumanAction(int id)
{
	return toNumPyArray(mEnvs[id]->GetHumanAction());
}


np::ndarray
EnvManager::
GetActions()
{
	Eigen::MatrixXd actions(mNumEnvs,this->GetNumAction());
	for (int id = 0;id<mNumEnvs;++id)
	{
		actions.row(id) = mEnvs[id]->GetAction().transpose();
	}

	return toNumPyArray(actions);
}


np::ndarray
EnvManager::
GetHumanActions()
{
	Eigen::MatrixXd actions(mNumEnvs,this->GetNumHumanAction());
	for (int id = 0;id<mNumEnvs;++id)
	{
		actions.row(id) = mEnvs[id]->GetHumanAction().transpose();
	}

	return toNumPyArray(actions);
}





void 
EnvManager::
SetAction(np::ndarray np_array, int id)
{
	mEnvs[id]->SetAction(toEigenVector(np_array));
}

double 
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

double 
EnvManager::
GetHumanReward(int id)
{
	return mEnvs[id]->GetHumanReward();
}



np::ndarray
EnvManager::
GetFullObservation(int id)
{
	return toNumPyArray(mEnvs[id]->GetFullObservation());
}


void
EnvManager::
Steps(int num, int doneStep)
{
	int totalStep = this->GetNumSteps();

#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++){
			mEnvs[id]->ProcessAction(j+doneStep, totalStep);
			mEnvs[id]->Step();
		}
	}
}
void
EnvManager::
StepsAtOnce()
{
	int num = this->GetNumSteps();
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
		{
			// set mAction as the interpolation of PrevAction, Current Action; 
			mEnvs[id]->ProcessAction(j, num);
			mEnvs[id]->Step();
		}
	}
}
void
EnvManager::
Resets(bool RSI)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}
np::ndarray
EnvManager::
IsEndOfEpisodes()
{
	std::vector<bool> is_end_vector(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		is_end_vector[id] = mEnvs[id]->IsEndOfEpisode();
	}

	return toNumPyArray(is_end_vector);
}
np::ndarray
EnvManager::
GetStates()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState());
	for (int id = 0;id<mNumEnvs;++id)
	{
		states.row(id) = mEnvs[id]->GetState().transpose();
	}

	return toNumPyArray(states);
}

np::ndarray
EnvManager::
GetFullObservations()
{
	Eigen::MatrixXd obs(mNumEnvs, this->GetNumFullObservation());
	for (int id = 0;id<mNumEnvs;++id)
	{
		obs.row(id) = mEnvs[id]->GetFullObservation().transpose();
	}
	return toNumPyArray(obs);
}

void 
EnvManager::
UpdateStateBuffers()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateStateBuffer();
	}
}


void
EnvManager::
SetActions(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(action.row(id).transpose());
	}
}

void
EnvManager::
SetHumanActions(np::ndarray np_array)
{
	Eigen::MatrixXd action_human = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetHumanAction(action_human.row(id).transpose());
	}
}




void
EnvManager::
UpdateActionBuffers(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateActionBuffer(action.row(id).transpose());
	}
}


void
EnvManager::
UpdateHumanActionBuffers(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->UpdateHumanActionBuffer(action.row(id).transpose());
	}
}




np::ndarray
EnvManager::
GetRewards()
{
	std::vector<float> rewards(mNumEnvs);

	for (int id = 0;id<mNumEnvs;++id)
	{
		rewards[id] = mEnvs[id]->GetReward();
	}
	return toNumPyArray(rewards);
}



np::ndarray
EnvManager::
GetMuscleTorques()
{
	std::vector<Eigen::VectorXd> mt(mNumEnvs);

#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mt[id] = mEnvs[id]->GetMuscleTorques();
	}
	return toNumPyArray(mt);
}
np::ndarray
EnvManager::
GetDesiredTorques()
{
	std::vector<Eigen::VectorXd> tau_des(mNumEnvs);
	
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		tau_des[id] = mEnvs[id]->GetDesiredTorques();
	}
	return toNumPyArray(tau_des);
}

void
EnvManager::
SetActivationLevels(np::ndarray np_array)
{
	std::vector<Eigen::VectorXd> activations =toEigenVectorVector(np_array);
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations[id]);
}

p::list
EnvManager::
GetMuscleTuples()
{
	p::list all;
	for (int id = 0; id < mNumEnvs; ++id)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			p::list t;
			t.append(toNumPyArray(tps[j].JtA));
			t.append(toNumPyArray(tps[j].tau_des));
			t.append(toNumPyArray(tps[j].L));
			t.append(toNumPyArray(tps[j].b));
			all.append(t);
		}
		tps.clear();
	}

	return all;
}
using namespace boost::python;

BOOST_PYTHON_MODULE(pymss)
{
	Py_Initialize();
	np::initialize();
	class_<EnvManager>("EnvManager",init<std::string,int>())
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetNumHumanAction",&EnvManager::GetNumHumanAction)
		.def("GetNumFullObservation",&EnvManager::GetNumFullObservation)
		.def("GetNumHumanObservation",&EnvManager::GetNumHumanObservation)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("GetAction",&EnvManager::GetAction)
		.def("GetActions",&EnvManager::GetActions)
		.def("GetHumanAction",&EnvManager::GetHumanAction)
		.def("GetHumanActions",&EnvManager::GetHumanActions)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("UseHumanNetwork",&EnvManager::UseHumanNetwork)
		.def("UseSymmetry",&EnvManager::UseSymmetry)
		.def("Step",&EnvManager::Step)
		.def("Reset",&EnvManager::Reset)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("GetState",&EnvManager::GetState)
		.def("SetAction",&EnvManager::SetAction)
		.def("GetReward",&EnvManager::GetReward)
		.def("GetHumanReward",&EnvManager::GetHumanReward)
		.def("Steps",&EnvManager::Steps)
		.def("StepsAtOnce",&EnvManager::StepsAtOnce)
		.def("Resets",&EnvManager::Resets)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetStates",&EnvManager::GetStates)
		.def("GetFullObservation",&EnvManager::GetFullObservation)
		.def("GetFullObservations",&EnvManager::GetFullObservations)
		.def("SetActions",&EnvManager::SetActions)
		.def("SetHumanActions",&EnvManager::SetHumanActions)
		.def("GetRewards",&EnvManager::GetRewards)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetNumActiveMuscles",&EnvManager::GetNumActiveMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("GetMuscleTuples",&EnvManager::GetMuscleTuples)
		.def("UpdateStateBuffers",&EnvManager::UpdateStateBuffers)
		.def("UpdateHumanActionBuffers",&EnvManager::UpdateHumanActionBuffers)
		.def("UpdateActionBuffers",&EnvManager::UpdateActionBuffers);
		
}