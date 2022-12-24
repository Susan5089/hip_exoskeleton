#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "NumPyHelper.h"
class EnvManager
{
public:
	EnvManager(std::string meta_file,int num_envs);

	int GetNumState();
	int GetNumAction();
	int GetNumHumanAction();
	int GetNumFullObservation();
	int GetNumHumanObservation();

	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();
	bool UseHumanNetwork();
	bool UseSymmetry();

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);
	np::ndarray GetState(int id);
	void SetAction(np::ndarray np_array, int id);
	
	double GetReward(int id);
	double GetHumanReward(int id);
	np::ndarray GetAction(int id); 
	np::ndarray GetHumanAction(int id); 
	np::ndarray GetFullObservation(int id);

	void Steps(int num, int donestep);
	void StepsAtOnce();
	void Resets(bool RSI);
	np::ndarray IsEndOfEpisodes();
	np::ndarray GetStates();
	np::ndarray GetFullObservations();
	np::ndarray GetActions();
	np::ndarray GetHumanActions();
	void SetActions(np::ndarray np_array);
	void SetHumanActions(np::ndarray np_array);
	np::ndarray GetRewards();
	void UpdateStateBuffers();
	void UpdateActionBuffers(np::ndarray np_array);
	void UpdateHumanActionBuffers(np::ndarray np_array);
	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	int GetNumActiveMuscles(){return mEnvs[0]->GetCharacter()->GetActiveMuscleNum();}
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	void SetActivationLevels(np::ndarray np_array);
	
	p::list GetMuscleTuples();
private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif