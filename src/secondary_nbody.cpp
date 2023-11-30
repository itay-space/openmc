#include "openmc/secondary_nbody.h"

#include <cmath> // for log

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/nuclide.h"

namespace openmc {

//==============================================================================
// NBodyPhaseSpace implementation
//==============================================================================

NBodyPhaseSpace::NBodyPhaseSpace(hid_t group)
{
  read_attribute(group, "n_particles", n_bodies_);
  read_attribute(group, "total_mass", mass_ratio_);
  read_attribute(group, "atomic_weight_ratio", A_);
  read_attribute(group, "q_value", Q_);
}

void NBodyPhaseSpace::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
  mu = uniform_distribution(-1., 1., seed);

  // Determine E_max parameter
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * (A_ / (A_ + 1.0) * E_in + Q_);

  // x is essentially a Maxwellian distribution
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 3:
    y = maxwell_spectrum(1.0, seed);
    break;
  case 4:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    y = -std::log(r1 * r2 * r3);
    break;
  case 5:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    r4 = prn(seed);
    r5 = prn(seed);
    r6 = prn(seed);
    y = -std::log(r1 * r2 * r3 * r4) -
        std::log(r5) * std::pow(std::cos(PI / 2.0 * r6), 2);
    break;
  default:
    throw std::runtime_error {"N-body phase space with >5 bodies."};
  }

  // Now determine v and E_out
  double v = x / (x + y);
  E_out = E_max * v;
}

void NBodyPhaseSpace::get_pdf(
  double det_pos[3],double E_in,double& E_out, uint64_t* seed , Particle &p,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
// By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
 

  // Determine E_max parameter
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * (A_ / (A_ + 1.0) * E_in + Q_);

  // x is essentially a Maxwellian distribution
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 3:
    y = maxwell_spectrum(1.0, seed);
    break;
  case 4:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    y = -std::log(r1 * r2 * r3);
    break;
  case 5:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    r4 = prn(seed);
    r5 = prn(seed);
    r6 = prn(seed);
    y = -std::log(r1 * r2 * r3 * r4) -
        std::log(r5) * std::pow(std::cos(PI / 2.0 * r6), 2);
    break;
  default:
    throw std::runtime_error {"N-body phase space with >5 bodies."};
  }

  // Now determine v and E_out
  double v = x / (x + y);
  E_out = E_max * v;




const auto& nuc {data::nuclides[p.event_nuclide()]};
const auto& rx {nuc->reactions_[p.event_index_mt()]};
double A = nuc->awr_;
 Direction u_lab {det_pos[0]-p.r().x,  // towards the detector
                   det_pos[1]-p.r().y,
                   det_pos[2]-p.r().z};
Direction u_lab_unit = u_lab/u_lab.norm(); // normalize
double m1= p.getMass()/1e6; // mass of incoming particle in MeV
double m2= m1*A; // mass of target 
double E1_tot = p.E_last()/1e6 + m1; // total Energy of incoming particle in MeV
double p1_tot = std::sqrt(E1_tot*E1_tot  - m1*m1); // total momenta of incoming particle in MeV
Direction p1=p1_tot*p.u_last(); // 3 momentum of incoming particle
Direction p2= p.v_t() * m2 /C_LIGHT; //3 momentum of target in lab 
double E2_tot = std::sqrt(p2.norm()*p2.norm() + m2*m2); // 
double E_cm = E1_tot + E2_tot;
Direction p_cm = p1 + p2;
double p_tot_cm = p_cm.norm();
double mu_lab = u_lab_unit.dot(p_cm) /  ( p_tot_cm ) ;  // between cm and p3

   if (rx->scatter_in_cm_) {
   // std::cout << "E_out_cm in nbody" << E_out << std::endl;
    double cond =E_out * (A+1)*(A+1) + E_in * (mu_lab*mu_lab - 1);
    
   
  //if (cond<0) {std::cout << "cond < 0" << std::endl;}
   
   if ( cond >= 0)
   {
    double  E_lab1 = (E_out * (A+1)*(A+1) - 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
    double  mu_cm1 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab1))*std::sqrt(E_lab1/E_out);
    double pdf_mu1_cm =0.5; // center of mass
    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab1);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(pdf_mu1_cm);
    double deriv = sqrt(E_lab1 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab1));
    double pdf_mu1_lab = pdf_mu1_cm * deriv;
    pdfs_lab.push_back(pdf_mu1_lab);
   // std::cout << "pdf_mu1_lab " << pdf_mu1_lab << std::endl;
  // std::cout << "E_lab1: " << E_lab1 << std::endl;
   // std::cout << "mu_cm1: " << mu_cm1 << std::endl;
   // std::cout << "pdf_mu1_cm: " << pdf_mu1_cm << std::endl;

    if (cond > 0)
    {
      double  E_lab2 = (E_out * (A+1)*(A+1) + 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
      double  mu_cm2 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab2))*std::sqrt(E_lab2/E_out);
      double pdf_mu2_cm = 0.5; // center of mass
      Particle ghost_particle=Particle();
      ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab2);
      ghost_particles.push_back(ghost_particle);
      pdfs_cm.push_back(pdf_mu2_cm);
      double deriv = sqrt(E_lab2 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab2));
      double pdf_mu2_lab = pdf_mu2_cm * deriv;
      pdfs_lab.push_back(pdf_mu2_lab);
     // std::cout << "pdf_mu2_lab " << pdf_mu2_lab << std::endl;
    //  std::cout << "E_lab2: " << E_lab2 << std::endl;
   // std::cout << "mu_cm2: " << mu_cm2 << std::endl;
   // std::cout << "pdf_mu2_cm: " << pdf_mu2_cm << std::endl;
    
    }

   }
    //std::cout << "E_out_lab in nbody" << E_lab << std::endl;         



   }

   if (!rx->scatter_in_cm_)
   {
   fatal_error("didn't implement lab");
   }


}



} // namespace openmc
