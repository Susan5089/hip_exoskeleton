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
	std::cout << "Seed: " << seed << std::endl; 
	// env->ResetInitialState(env->GetCharacter()->GetInitialState());
	// if(argc==3)
	// 	env->SetUseMuscle(true);
	// else
	// 	env->SetUseMuscle(false);
	// env->SetControlHz(30);
	// env->SetSimulationHz(600);

	// MASS::Character* character = new MASS::Character();
	// character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"),true);
	// if(env->GetUseMuscle())
	// 	character->LoadMuscles(std::string(MASS_ROOT_DIR)+std::string("/data/muscle.xml"));
	// character->LoadBVH(std::string(MASS_ROOT_DIR)+std::string("/data/motion/walk.bvh"),true);
	
	// double kp = 300.0;
	// character->SetPDParameters(kp,sqrt(2*kp));
	// env->SetCharacter(character);
	// env->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	// env->Initialize();

	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::Window* window;
	
	std::cout << "argc:  "<< argc << std::endl;
	if(argc == 2)
	{
		std::cout << "Seed:111111 "  << std::endl; 
		window = new MASS::Window(env);
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
			window = new MASS::Window(env,argv[2],argv[3]);
		}
		else if(env->GetUseMuscleNN() && env->GetUseHumanNN())
		{
			if(argc!=5){
				std::cout<<"Please provide three networks"<<std::endl;
				return 0;
			}
			window = new MASS::Window(env,argv[2],argv[3],argv[4]);
		}
		else
		{
			if(argc!=3)
			{
				std::cout<<"Please provide the network"<<std::endl;
				return 0;
			}
			window = new MASS::Window(env,argv[2]);
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
