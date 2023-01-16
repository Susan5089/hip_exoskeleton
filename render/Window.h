#ifndef __MASS_WINDOW_H__
#define __MASS_WINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS
{
class Environment;
class Muscle;
class Window : public dart::gui::Win3D
{
public:
	Window(std::vector<MASS::Environment*> mEnvs);
	Window(std::vector<MASS::Environment*> mEnvs, const std::string& nn_path);
	Window(std::vector<MASS::Environment*> mEnvs, const std::string& nn_path,const std::string& human_muscle_nn_path);  // for muscle path or human path
	Window(std::vector<MASS::Environment*> mEnvs, const std::string& nn_path,const std::string& human_nn_path, const std::string& muscle_nn_path);


	void draw() override;
	void keyboard(unsigned char _key, int _x, int _y) override;
	void displayTimer(int _val) override;
	Eigen::Vector3d geo_center_target, geo_center_target_left, geo_center_target_right;

private:
	void SetFocusing();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawCollisionShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawArrow(Eigen::Vector3d pos, Eigen::Vector3d force, Eigen::Vector4d color, double radius,double heightN, double coneHt); 
	void DrawExternalforce();
	void DrawSpringforce();
	void DrawBushingforce();
	void DrawJointConstraint();
	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawGround(double y);
	void PlotFigure();
	void Step();
	void Reset();
	

	// Eigen::VectorXd GetActionFromNN();
	// Eigen::VectorXd GetActionFromHumanNN();
	// Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt);

	Eigen::VectorXd GetActionFromNN(MASS::Environment* mEnv);
	Eigen::VectorXd GetActionFromHumanNN(MASS::Environment* mEnv);
	Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd& mt, Environment* mEnv);

	p::object mm,mns,sys_module,nn_module,human_nn_module,muscle_nn_module;
	float t; 

	std::vector<Environment*> mEnvs;
	bool mFocus;
	bool mSimulating;
	bool mDrawMuscles;
	bool mDrawShape;
	bool mDrawOBJ;
	bool mDrawCollision;
	bool mDrawContact;
	bool mDrawEndEffectors;
	bool mDrawEndEffectorTargets;
	bool mDrawShadow;
	bool mNNLoaded;
	bool mHumanNNLoaded;
	bool mMuscleNNLoaded;
	bool mDrawSpringforce;
	bool mDrawBushingforce;
	bool mDrawFigure;
	bool mDrawCompositionforces;
	bool mDrawground;
	Eigen::Affine3d mViewMatrix;

	std::vector<double> time_vector;
	std::vector<double> t_vector;
	std::vector<double> tt;


};
};


#endif
