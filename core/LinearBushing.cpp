#include "LinearBushing.h"
#include <iostream>
#include "Environment.h"
#include <Eigen/Dense>
#include <Eigen/Geometry> 
#include "DARTHelper.h"


namespace details {

    typedef double Real;
    typedef Eigen::Vector3d Vec3;
    typedef Eigen::Matrix3d Mat3;

	Real square(Real v) { return v * v; }

	//code from Simbody Rotation.cpp function convertThreeAxesBodyFixedRotationToThreeAngles 
	//------------------------------------------------------------------------------
	// Calculate angles ONLY for a three-angle, three-axes, body-fixed, ijk rotation 
	// sequence where i != j and j != k.
	//------------------------------------------------------------------------------
	Vec3 convertThreeAxesBodyFixedRotationToThreeAngles
		(const Mat3& R, int axis1, int axis2, int axis3)
		{
			// Ensure this method has proper arguments.
			assert(axis1 !=axis2 && axis1 != axis3);

			const int i = axis1;
			const int j = axis2;
			const int k = axis3;

			// Need to know if using a forward or reverse cyclical.
			Real plusMinus = 1.0, minusPlus = -1.0;
			auto prev_axis = (axis1 + 2) % 3;
			if (prev_axis == axis2) { plusMinus = -1.0, minusPlus = 1.0; }

			// Calculate theta2 using lots of information in the rotation matrix.
			Real Rsum = std::sqrt((square(R(i,i)) + square(R(i,j))
				+ square(R(j,k)) + square(R(k,k))) / 2);
			// Rsum = abs(cos(theta2)) is inherently positive.
			Real theta2 = std::atan2(plusMinus*R(i,k), Rsum);
			Real theta1, theta3;

			// There is a "singularity" when cos(theta2) == 0
			Real Eps = 1.0e-6;
			if (Rsum > 4 * Eps) {
				theta1 = std::atan2(minusPlus*R(j,k), R(k,k));
				theta3 = std::atan2(minusPlus*R(i,j), R(i,i));
			}
			else if (plusMinus*R(i,k) > 0) {
				const Real spos = R(j,i) + plusMinus*R(k,j);  // 2*sin(theta1 + plusMinus*theta3)
				const Real cpos = R(j,j) + minusPlus*R(k,i);  // 2*cos(theta1 + plusMinus*theta3)
				const Real theta1PlusMinusTheta3 = std::atan2(spos, cpos);
				theta1 = theta1PlusMinusTheta3;  // Arbitrary split
				theta3 = 0;                      // Arbitrary split
			}
			else {
				const Real sneg = plusMinus*(R(k,j) + minusPlus*R(j,i));  // 2*sin(theta1 + minusPlus*theta3)
				const Real cneg = R(j,j) + plusMinus*R(k,i);              // 2*cos(theta1 + minusPlus*theta3)
				const Real theta1MinusPlusTheta3 = std::atan2(sneg, cneg);
				theta1 = theta1MinusPlusTheta3;  // Arbitrary split
				theta3 = 0;                      // Arbitrary split
			}

			// Return values have the following ranges:
			// -pi   <=  theta1  <=  +pi
			// -pi/2 <=  theta2  <=  +pi/2   (Rsum is inherently positive)
			// -pi   <=  theta3  <=  +pi
			return Vec3(theta1, theta2, theta3);
		}
}

namespace MASS
{
    LinearBushing::LinearBushing()
     :Force()
    {
        //In the Master theis by Pujals 2017
        //Simulation of the assistance of an exoskeleton on lower limbs joints using Opensim
        //https://upcommons.upc.edu/handle/2117/110512
        //the following parameters were used
        //Vec3 transStiffness(10000), rotStiffness(1000), transDamping(0.1), rotDamping(0);
        mTranslationStiffness.setConstant(10000);
        mRotationStiffness.setConstant(1000);
        mTranslationDamping.setConstant(0.1);
        mRotationDamping.setZero();

        //default to zero so no force will be produced
        //mTranslationStiffness.setZero();
        //mRotationStiffness.setZero();
        //mTranslationDamping.setZero();
        //mRotationDamping.setZero();
    }

    LinearBushing::~LinearBushing()
    {
    }


    Force* LinearBushing::CreateForce()
    {
        return new LinearBushing;
    }

    // update:
	void LinearBushing::Update() 
    {
        std::cout << mBody1->getName() << std::endl;
        std::cout << mBody2->getName() << std::endl;
        Eigen::Isometry3d trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Isometry3d trf2 = mBody2->getTransform() * mFrame2;
        std::cout << "mbody1: frame: " << trf1.translation() << std::endl;
        std::cout << "mbody2: frame: " << trf2.translation() << std::endl;
        Eigen::Isometry3d rel = trf1.inverse() * trf2;
        Eigen::Vector3d tr = rel.translation();
        
		//euler angle may not be reliable or unique, 
		//the range for all three values are [-Pi, Pi]. 
		//In some cases, the Euler angles are close to Pi instead of zero for near identity matrix
		//Eigen::Vector3d ea = rel.rotation().eulerAngles(0,1,2);

		//When the first and last axis is not equal, the system is a Tait-Bryan system
		//the beta angle will be in the range [-PI/2, PI/2].
		//typedef Eigen::EulerSystem<Eigen::EULER_X, Eigen::EULER_Y, Eigen::EULER_Z> MyEulerSystem;
		//Eigen::EulerAngles<double, MyEulerSystem> ea2(rel.R());
		//Eigen::Vector3d ea1(ea2.alpha(), ea2.beta(), ea2.gamma());
		//still don't work well

		//std::cout << "Euler angle " << ea << std::endl;
		Eigen::Matrix3d R = rel.rotation();
        Eigen::Vector3d ea = details::convertThreeAxesBodyFixedRotationToThreeAngles(R, 0, 1, 2);
		// std::cout << "Euler angle " << ea << std::endl;

        mForce1  = mTranslationStiffness.array() * tr.array();
        mTorque1 = mRotationStiffness.array() * ea.array();
        // if (mTranslationDamping.maxCoeff()>0)
            // std::cout << "mForce1:\n " << mForce1 << std::endl;
        Eigen::Vector3d gv1 = mBody1->getLinearVelocity(mFrame1.translation());
        Eigen::Vector3d gv2 = mBody2->getLinearVelocity(mFrame2.translation());
        Eigen::Vector3d grelv = gv2- gv1;
        // if (mTranslationDamping.maxCoeff()>0)
            // std::cout << "grelv:     "  << grelv << std::endl;
        Eigen::Vector3d relv = trf1.inverse().rotation() * grelv;
        // if (mTranslationDamping.maxCoeff()>0)
            // std::cout << "relv:     "  << relv << std::endl;
        Eigen::Vector3d frelv = mTranslationDamping.array() * relv.array();
        // if (mTranslationDamping.maxCoeff()>0)
            // std::cout << "frelv:     "  << frelv << std::endl;
        mForce1 += frelv;

        //need to think how to do the torque damping
        Eigen::Vector3d gr1= mBody1->getAngularVelocity();
        Eigen::Vector3d gr2= mBody2->getAngularVelocity();
        Eigen::Vector3d grelr = gr2 - gr1;
        Eigen::Vector3d relr = trf1.inverse().rotation() * grelr;
        Eigen::Vector3d trelr = mRotationDamping.array() * relr.array();
        // std::cout << "relr: \n "  << relr << std::endl;
        //is the consistent with euler angle velocity?
        mTorque1 += trelr;

        Eigen::Vector3d gforce = trf1.rotation() * mForce1;
        Eigen::Vector3d gtorque = trf1.rotation() * mTorque1;

        //convert the global force and torque to the local frame
        mForce2 = -(trf2.inverse().rotation() * gforce);
        mTorque2 = -(trf2.inverse().rotation() * gtorque);

        Force::Update(); 
    };

    Eigen::Vector3d LinearBushing::GetForce()
    {
        auto trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Vector3d gforce = trf1.rotation()* mForce1;  // rotation 
        return gforce;
    }

    Eigen::Vector3d LinearBushing::GetTorque()
    {
        auto trf1 = mBody1->getTransform() * mFrame1;
        Eigen::Vector3d gtorque = trf1.rotation() * mTorque1;        // rotation 
        return gtorque;
    }

    std::vector<Eigen::Vector3d> LinearBushing::GetPoint()
    {
        std::vector<Eigen::Vector3d> Pos; 
        auto trf1 = mBody1->getTransform() * mFrame1;
        auto trf2 = mBody2->getTransform() * mFrame2;
        Pos.push_back(trf1.translation());
        Pos.push_back(trf2.translation());
        return Pos;
    }

    void LinearBushing::ApplyForceToBody() 
    {   
        //update(); //make sure the force is computed

        bool isPosLocal = true; 
        bool isForceLocal = true;

        mBody1->addExtForce(mFrame1.rotation() * mForce1, mFrame1.translation(), isForceLocal, isPosLocal);
        mBody1->addExtTorque(mFrame1.rotation() * mTorque1, isForceLocal);

        mBody2->addExtForce(mFrame2.rotation() * mForce2, mFrame2.translation(), isForceLocal, isPosLocal);
        mBody2->addExtTorque(mFrame2.rotation() * mTorque2, isForceLocal);
    };

    void LinearBushing::ReadFromXml(TiXmlElement& inp) 
    {   
        
        Force::ReadFromXml(inp);
        mTranslationStiffness = string_to_vector3d(inp.Attribute("mTranslationStiffness"));
        mTranslationDamping = string_to_vector3d(inp.Attribute("mTranslationDamping"));
        mRotationStiffness = string_to_vector3d(inp.Attribute("mRotationStiffness"));
        mRotationDamping =string_to_vector3d(inp.Attribute("mRotationDamping"));
        std::vector<Eigen::Vector6d> frm;
        
        for(TiXmlElement* Frame = inp.FirstChildElement("Frame");Frame!=nullptr;Frame = Frame->NextSiblingElement("Frame"))
        {
            mbodyNodeNames.push_back(Frame->Attribute("body"));
			frm.push_back(string_to_vector6d(Frame->Attribute("frame")));
        }

        // std::cout << ""
        //3 translations followed by 3 Euler angles (X/Y/Z)
        Eigen::Vector6d frm1 = frm[0];
        Eigen::Vector6d frm2 = frm[1];
        mFrame1 = Eigen::AngleAxisd(frm1[3], Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(frm1[4], Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(frm1[5], Eigen::Vector3d::UnitZ()); 
        mFrame1.translation() = frm1.head(3);

        mFrame2 = Eigen::AngleAxisd(frm2[3], Eigen::Vector3d::UnitX())
             * Eigen::AngleAxisd(frm2[4], Eigen::Vector3d::UnitY())
             * Eigen::AngleAxisd(frm2[5], Eigen::Vector3d::UnitZ()); 
        mFrame2.translation() = frm2.head(3);

    };

    void LinearBushing::SetBodyNode(dart::dynamics::SkeletonPtr mSkeleton)
    {
        for (int i=0; i<mSkeleton->getNumBodyNodes(); i++)
		    std::cout << mSkeleton->getBodyNode(i)->getName() << std::endl;
        mBody1 = mSkeleton->getBodyNode(mbodyNodeNames[0]);
        mBody2 = mSkeleton->getBodyNode(mbodyNodeNames[1]);
        std::string a = mBody2->getName(); 
	    std::string b = mBody1->getName(); 
        Update();
    }
}