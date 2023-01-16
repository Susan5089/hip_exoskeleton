#include "Window.h"
#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
namespace p = boost::python;
namespace np = boost::python::numpy;
int main(int argc,char** argv)
{	

	unsigned seed = (unsigned)time(NULL)%100;
	std::cout << "Seed: " << seed << std::endl; 
	std::srand(seed);   
	// std::srand(94);  
	
	MASS::Environment* env = new MASS::Environment();
	if(argc==1)
	{
		std::cout<<"Provide Metadata.txt"<<std::endl;
		return 0;
	}
	env->Initialize(std::string(argv[1]),true);
	std::tuple<Eigen::VectorXd,Eigen::VectorXd,std::map<std::string,Eigen::Vector3d>> pv = env->GetCharacter()->GetTargetPosAndVel(0,1.0/100);
	Eigen::VectorXd mTargetPositions = std::get<0>(pv);
	
	std::vector<MASS::Environment*> mEnvs;
	mEnvs.push_back(env);

	// int offset_x[2] = {-3, 3};
	// for(int _i=0; _i<2; _i++){
	// 	mTargetPositions[3] += offset_x[_i];
	// 	MASS::Environment* env = new MASS::Environment();
	// 	env->Initialize(std::string(argv[1]),true);
	// 	env->Initialize();
	// 	env->GetCharacter()->GetSkeleton()->setPositions(mTargetPositions);
	// 	mTargetPositions[3] -= offset_x[_i];
	// 	mEnvs.push_back(env);
	// }

	// int offset_y[4] = {-6, 6};
	// for(int _i=0; _i<2; _i++){
	// 	mTargetPositions[5] += offset_y[_i];
	// 	MASS::Environment* env = new MASS::Environment();
	// 	env->Initialize(std::string(argv[1]),true);
	// 	env->Initialize();
	// 	env->GetCharacter()->GetSkeleton()->setPositions(mTargetPositions);
	// 	mTargetPositions[5] -= offset_y[_i];
	// 	mEnvs.push_back(env);
	// }
	
	std::cout << "mEnvs size: " << mEnvs.size() << std::endl; 
	std::cout << "Seed: " << seed << std::endl; 

	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::Window* window;
	
	std::cout << "argc:  "<< argc << std::endl;
	if(argc == 2)
	{
		std::cout << "Seed:111111 "  << std::endl; 
		window = new MASS::Window(mEnvs);
		std::cout << "Seed:2222222 " << std::endl; 
	}
	else
	{
		if(!env->GetUseMuscleNN() && env->GetUseHumanNN())
		{
			if(argc!=4){
				std::cout<<"Please provide two networks"<<std::endl;
				return 0;
			}
			window = new MASS::Window(mEnvs ,argv[2],argv[3]);
		}
		else if(env->GetUseMuscleNN() && env->GetUseHumanNN())
		{
			if(argc!=5){
				std::cout<<"Please provide three networks"<<std::endl;
				return 0;
			}
			window = new MASS::Window(mEnvs, argv[2],argv[3],argv[4]);
		}
		else
		{
			if(argc!=3)
			{
				std::cout<<"Please provide the network"<<std::endl;
				return 0;
			}
			window = new MASS::Window(mEnvs, argv[2]);
		}
	}
	// if(argc==1)
	// 	window = new MASS::Window(env);
	// else if (argc==2)
	// 	window = new MASS::Window(env,argv[1]);
	// else if (argc==3)
	// 	window = new MASS::Window(env,argv[1],argv[2]);
	window->initWindow(1920,1080,"gui");
	glutMainLoop();
}
